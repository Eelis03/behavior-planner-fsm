"""The traffic simulator that drives one scenario to completion.

Every vehicle is updated synchronously: all decisions are taken from the same
snapshot, then all states are advanced. Updating in place would make the result
depend on the order the vehicles happen to sit in the list, which is exactly the
kind of hidden coupling that turns a seeded run into an unreproducible one.

The behaviour layer runs at :attr:`PlannerConfig.planning_period` and the
trajectory layer at :attr:`SimulationConfig.dt`. Running the state machine at
the integration step makes it chatter between candidates whose costs differ in
the last few bits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from behavior_planner.algorithm.base import BehaviorPolicy, CarFollowingModel, LaneChangeModel
from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.mobil import MobilLaneChangeModel
from behavior_planner.algorithm.trajectory import (
    TrajectoryGenerator,
    TrajectoryPoint,
    ballistic_step,
    lane_change_for,
    longitudinal_acceleration,
)
from behavior_planner.model.config import PlannerConfig
from behavior_planner.model.decision import Decision, DecisionContext, VetoReason
from behavior_planner.model.lateral import LateralProfile
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorEvent, BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot, time_headway, time_to_collision
from behavior_planner.model.vehicle import EGO_ID, LaneChange, Vehicle
from behavior_planner.pipeline.scenarios import Scenario, build_vehicles
from behavior_planner.pipeline.trace import RunTrace, StepRecord

__all__ = ["SimulationConfig", "TrafficSimulator"]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Timing of the simulation loop."""

    dt: float = 0.1
    """Integration step, in seconds."""

    traffic_decision_period: float = 1.0
    """Interval at which a traffic vehicle consults MOBIL, in seconds.

    Real drivers do not reconsider their lane ten times a second, and letting
    them do so here produces a stream of marginal changes that swamps the ego's
    own decisions.
    """

    def __post_init__(self) -> None:
        """Reject timings the loop cannot honour."""
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.traffic_decision_period <= 0.0:
            raise ValueError(
                f"traffic_decision_period must be positive, got {self.traffic_decision_period}"
            )


@dataclass(frozen=True, slots=True)
class TrafficSimulator:
    """Runs one scenario and returns its trace."""

    policy: BehaviorPolicy
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    car_following: CarFollowingModel = field(default_factory=IntelligentDriverModel)
    lane_change_model: LaneChangeModel | None = None
    policy_name: str = "fsm"

    def run(self, scenario: Scenario, *, steps: int | None = None) -> RunTrace:
        """Simulate ``scenario`` and record every step.

        ``steps`` overrides the scenario duration, which is what the integration
        tests use to run every example script under a reduced budget.
        """
        dt = self.simulation.dt
        total_steps = round(scenario.duration / dt) if steps is None else steps
        if total_steps < 1:
            raise ValueError(f"a run needs at least one step, got {total_steps}")

        road = scenario.road
        traffic_model = self.lane_change_model or MobilLaneChangeModel(self.car_following)
        generator = TrajectoryGenerator(
            car_following=self.car_following, config=self.planner_config
        )

        vehicles = list(build_vehicles(scenario))
        state = BehaviorState.KEEP_LANE
        event = BehaviorEvent.STAY
        records: list[StepRecord] = []
        collisions: set[tuple[int, int]] = set()
        plan_every = max(1, round(self.planner_config.planning_period / dt))
        traffic_every = max(1, round(self.simulation.traffic_decision_period / dt))

        for index in range(total_steps + 1):
            time = index * dt
            snapshot = TrafficSnapshot(road=road, vehicles=tuple(vehicles))
            ego = snapshot.by_id(EGO_ID)

            planned = index % plan_every == 0
            veto = VetoReason.NONE
            if planned:
                context = DecisionContext(
                    road=road,
                    snapshot=snapshot,
                    ego=ego,
                    state=state,
                    config=self.planner_config,
                    time=time,
                )
                decision = self.policy.decide(context)
                veto = _first_veto(decision)
                state, event = decision.state, decision.event
                vehicles[_index_of(vehicles, EGO_ID)] = self._apply_decision(
                    road, ego, state, event
                )
                snapshot = TrafficSnapshot(road=road, vehicles=tuple(vehicles))
                ego = snapshot.by_id(EGO_ID)

            overlaps = snapshot.colliding_pairs()
            collisions.update(overlaps)
            trajectory = self._ego_trajectory(generator, road, snapshot, ego, state, dt)
            records.append(
                self._record(
                    road=road,
                    snapshot=snapshot,
                    ego=ego,
                    state=state,
                    event=event,
                    time=time,
                    acceleration=trajectory[0].acceleration,
                    veto=veto,
                    planned=planned,
                    overlaps=len(overlaps),
                )
            )
            if index == total_steps:
                break

            vehicles = self._advance_all(
                road=road,
                snapshot=snapshot,
                vehicles=vehicles,
                state=state,
                trajectory=trajectory,
                traffic_model=traffic_model,
                consider_changes=index % traffic_every == 0,
                dt=dt,
            )

        return RunTrace(
            scenario=scenario.name,
            policy=self.policy_name,
            dt=dt,
            duration=total_steps * dt,
            seed=scenario.seed,
            records=tuple(records),
            vehicle_count=len(vehicles),
            collision_pairs=tuple(sorted(collisions)),
        )

    def _apply_decision(
        self, road: Road, ego: Vehicle, state: BehaviorState, event: BehaviorEvent
    ) -> Vehicle:
        """Attach, retire or reverse the ego's lateral manoeuvre after a decision."""
        if event is BehaviorEvent.COMMIT:
            return replace(
                ego,
                lane_change=lane_change_for(
                    road, ego, state, self.planner_config.lane_change_duration
                ),
            )
        if event is BehaviorEvent.COMPLETE and ego.lane_change is not None:
            arrived = road.lane_center(ego.lane_change.target_lane)
            return replace(ego, d=arrived, lane_change=None)
        if event is BehaviorEvent.ABORT and ego.lane_change is not None:
            running = ego.lane_change
            return replace(
                ego,
                lane_change=LaneChange(
                    source_lane=running.target_lane,
                    target_lane=running.source_lane,
                    profile=LateralProfile(
                        start=ego.d,
                        target=road.lane_center(running.source_lane),
                        duration=self.planner_config.lane_change_duration,
                    ),
                ),
            )
        return ego

    def _ego_trajectory(
        self,
        generator: TrajectoryGenerator,
        road: Road,
        snapshot: TrafficSnapshot,
        ego: Vehicle,
        state: BehaviorState,
        dt: float,
    ) -> tuple[TrajectoryPoint, ...]:
        """Generate the ego's trajectory for the current manoeuvre."""
        context = DecisionContext(
            road=road,
            snapshot=snapshot,
            ego=ego,
            state=state,
            config=self.planner_config,
        )
        return generator.generate(context, state, dt)

    def _advance_all(
        self,
        *,
        road: Road,
        snapshot: TrafficSnapshot,
        vehicles: list[Vehicle],
        state: BehaviorState,
        trajectory: tuple[TrajectoryPoint, ...],
        traffic_model: LaneChangeModel,
        consider_changes: bool,
        dt: float,
    ) -> list[Vehicle]:
        """Advance every vehicle by ``dt`` from the same snapshot."""
        advanced: list[Vehicle] = []
        for vehicle in vehicles:
            if vehicle.vehicle_id == EGO_ID:
                advanced.append(self._advance_ego(road, vehicle, state, trajectory, dt))
                continue
            advanced.append(
                self._advance_traffic(
                    road=road,
                    snapshot=snapshot,
                    vehicle=vehicle,
                    traffic_model=traffic_model,
                    consider_changes=consider_changes,
                    dt=dt,
                )
            )
        return advanced

    @staticmethod
    def _advance_ego(
        road: Road,
        ego: Vehicle,
        state: BehaviorState,
        trajectory: tuple[TrajectoryPoint, ...],
        dt: float,
    ) -> Vehicle:
        """Move the ego onto the second sample of its generated trajectory.

        A manoeuvre started by an abort runs while the state machine is already
        in lane keeping, so it is retired here rather than by a ``COMPLETE``
        event that the machine would refuse to accept.
        """
        point = trajectory[1]
        moved = ego.moved(s=road.wrap(point.s), d=point.d, speed=point.speed)
        if ego.lane_change is None:
            return moved
        stepped = ego.lane_change.advanced(dt)
        if stepped.is_complete and not state.is_changing:
            arrived = road.lane_center(stepped.target_lane)
            return replace(moved, d=arrived, lane_change=None)
        return replace(moved, lane_change=stepped)

    def _advance_traffic(
        self,
        *,
        road: Road,
        snapshot: TrafficSnapshot,
        vehicle: Vehicle,
        traffic_model: LaneChangeModel,
        consider_changes: bool,
        dt: float,
    ) -> Vehicle:
        """Advance one traffic vehicle under IDM, with MOBIL choosing its lane."""
        subject = vehicle
        if consider_changes and not subject.is_changing_lane:
            target = traffic_model.choose_lane(subject, snapshot)
            if target is not None:
                subject = replace(
                    subject,
                    lane_change=LaneChange(
                        source_lane=subject.assigned_lane(road),
                        target_lane=target,
                        profile=LateralProfile(
                            start=subject.d,
                            target=road.lane_center(target),
                            duration=self.planner_config.lane_change_duration,
                        ),
                    ),
                )

        acceleration = longitudinal_acceleration(
            car_following=self.car_following,
            road=road,
            snapshot=snapshot,
            ego=subject,
            state=BehaviorState.KEEP_LANE,
        )
        speed, s = ballistic_step(
            speed=subject.speed, s=subject.s, acceleration=acceleration, dt=dt, road=road
        )

        maneuver = subject.lane_change
        if maneuver is None:
            return subject.moved(s=s, d=subject.d, speed=speed)
        stepped = maneuver.advanced(dt)
        if stepped.is_complete:
            moved = subject.moved(s=s, d=road.lane_center(stepped.target_lane), speed=speed)
            return replace(moved, lane_change=None)
        return replace(subject.moved(s=s, d=stepped.offset(), speed=speed), lane_change=stepped)

    @staticmethod
    def _record(
        *,
        road: Road,
        snapshot: TrafficSnapshot,
        ego: Vehicle,
        state: BehaviorState,
        event: BehaviorEvent,
        time: float,
        acceleration: float,
        veto: VetoReason,
        planned: bool,
        overlaps: int,
    ) -> StepRecord:
        """Build the trace record for one step.

        The lane recorded is the one the ego's body is physically closest to,
        not the one it has been assigned to, because the metrics describe where
        the vehicle was rather than what it intended. The leader is the nearest
        one in any lane the body overlaps, which during a change means the
        headway is measured against both lanes at once.
        """
        lane = road.nearest_lane(ego.d)
        gap = math.inf
        headway = math.inf
        ttc = math.inf
        for occupied in ego.occupied_lanes(road):
            leader = snapshot.neighbors(ego, occupied).leader
            if leader is None or leader.gap >= gap:
                continue
            gap = leader.gap
            headway = time_headway(gap, ego.speed)
            ttc = time_to_collision(gap, ego.speed - leader.speed)
        return StepRecord(
            time=time,
            state=state,
            event=event,
            lane=lane,
            s=ego.s,
            d=ego.d,
            speed=ego.speed,
            acceleration=acceleration,
            leader_gap=gap,
            time_headway=headway,
            time_to_collision=ttc,
            veto_reason=veto,
            planned=planned,
            collisions=overlaps,
        )


def _first_veto(decision: Decision) -> VetoReason:
    """The first veto reason recorded in ``decision``, or ``NONE``."""
    vetoed = decision.vetoed
    return vetoed[0].verdict.reason if vetoed else VetoReason.NONE


def _index_of(vehicles: list[Vehicle], vehicle_id: int) -> int:
    """Position of ``vehicle_id`` in ``vehicles``."""
    for index, vehicle in enumerate(vehicles):
        if vehicle.vehicle_id == vehicle_id:
            return index
    raise KeyError(f"no vehicle with id {vehicle_id}")
