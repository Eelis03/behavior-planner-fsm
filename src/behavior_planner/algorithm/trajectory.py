"""Trajectory generation for the manoeuvre the behaviour layer has chosen.

The behaviour layer decides which lane; this layer produces the motion that
gets there. The longitudinal profile comes from forward integrating the car
following model against the binding leader, and the lateral profile is the
closed-form minimum jerk transition of :class:`LateralProfile`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from behavior_planner.algorithm.base import CarFollowingModel
from behavior_planner.model.config import PlannerConfig
from behavior_planner.model.decision import DecisionContext
from behavior_planner.model.lateral import LateralProfile
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import LaneNeighbors, TrafficSnapshot
from behavior_planner.model.vehicle import LaneChange, Vehicle

__all__ = [
    "TrajectoryGenerator",
    "TrajectoryPoint",
    "ballistic_step",
    "binding_lanes",
    "lane_change_for",
    "longitudinal_acceleration",
]


def ballistic_step(
    *, speed: float, s: float, acceleration: float, dt: float, road: Road
) -> tuple[float, float]:
    """Constant acceleration step that stops at zero speed instead of reversing.

    A plain constant acceleration update lets a hard braking command drive the
    speed negative, and a negative speed in a car following model is not a small
    error: the vehicle reverses into its follower. When the command would do
    that, the vehicle instead covers exactly the distance needed to reach a
    standstill and stays there, which keeps the speed non-negative for every
    admissible acceleration and every step size.
    """
    next_speed = speed + acceleration * dt
    if next_speed < 0.0:
        distance = 0.5 * speed * speed / abs(acceleration) if acceleration < 0.0 else 0.0
        return 0.0, road.wrap(s + distance)
    distance = speed * dt + 0.5 * acceleration * dt * dt
    return next_speed, road.wrap(s + max(distance, 0.0))


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One sample of a generated trajectory."""

    time: float
    s: float
    d: float
    speed: float
    acceleration: float
    lateral_rate: float


def binding_lanes(road: Road, ego: Vehicle, state: BehaviorState) -> tuple[int, ...]:
    """Lanes whose leaders constrain the ego in ``state``.

    While preparing or executing a change the ego must respect the leader of the
    lane it is leaving and the leader of the lane it is entering: the first
    because it is still there, the second because it is about to be in front.
    Taking the more restrictive of the two is what makes a prepare state useful,
    since it is how the ego drops back to find a gap.
    """
    current = ego.assigned_lane(road)
    if state is BehaviorState.KEEP_LANE and not ego.is_changing_lane:
        return (current,)
    if ego.lane_change is not None:
        source = ego.lane_change.source_lane
        target = ego.lane_change.target_lane
        return tuple(sorted({source, target}))
    neighbour = current + state.lane_offset
    if not road.contains_lane(neighbour):
        return (current,)
    return tuple(sorted({current, neighbour}))


def longitudinal_acceleration(
    *,
    car_following: CarFollowingModel,
    road: Road,
    snapshot: TrafficSnapshot,
    ego: Vehicle,
    state: BehaviorState,
) -> float:
    """Acceleration command for the ego, the strictest over its binding lanes."""
    commands = []
    for lane in binding_lanes(road, ego, state):
        neighbors: LaneNeighbors = snapshot.neighbors(ego, lane)
        leader = neighbors.leader
        gap = math.inf if leader is None else leader.gap
        leader_speed = 0.0 if leader is None else leader.speed
        commands.append(
            car_following.acceleration(
                speed=ego.speed, gap=gap, leader_speed=leader_speed, driver=ego.driver
            )
        )
    return min(commands)


def lane_change_for(
    road: Road, ego: Vehicle, state: BehaviorState, duration: float
) -> LaneChange | None:
    """The manoeuvre that ``state`` starts, or ``None`` if it starts none."""
    if not state.is_changing:
        return None
    source = ego.assigned_lane(road)
    target = source + state.lane_offset
    if not road.contains_lane(target):
        return None
    profile = LateralProfile(start=ego.d, target=road.lane_center(target), duration=duration)
    return LaneChange(source_lane=source, target_lane=target, profile=profile)


@dataclass(frozen=True, slots=True)
class TrajectoryGenerator:
    """Samples the motion the ego will follow over the planning horizon.

    The longitudinal profile is produced by re-evaluating the car following
    model at every sample against a leader assumed to hold its speed, which is
    the same assumption the behaviour layer makes one level up. The lateral
    profile is evaluated in closed form, so the sampling interval affects the
    resolution of the trajectory and nothing else.

    The simulator advances the ego by consuming the first interval of this
    trajectory each step. Generating the whole horizon rather than one step is
    what makes the output usable as a trajectory: it can be plotted, checked
    against comfort limits, or handed to a tracking controller.
    """

    car_following: CarFollowingModel
    config: PlannerConfig

    def generate(
        self,
        context: DecisionContext,
        state: BehaviorState,
        dt: float,
        *,
        horizon: float | None = None,
    ) -> tuple[TrajectoryPoint, ...]:
        """Sample the trajectory for ``state`` at interval ``dt``."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        span = self.config.trajectory_horizon if horizon is None else horizon
        if span <= 0.0:
            raise ValueError(f"horizon must be positive, got {span}")

        road = context.road
        ego = context.ego
        maneuver = ego.lane_change
        acceleration = longitudinal_acceleration(
            car_following=self.car_following,
            road=road,
            snapshot=context.snapshot,
            ego=ego,
            state=state,
        )
        leaders = self._frozen_leaders(context, state)

        points = [
            TrajectoryPoint(
                time=0.0,
                s=ego.s,
                d=ego.d,
                speed=ego.speed,
                acceleration=acceleration,
                lateral_rate=ego.lateral_speed,
            )
        ]
        s = ego.s
        speed = ego.speed
        elapsed = 0.0 if maneuver is None else maneuver.elapsed
        steps = max(1, round(span / dt))
        for index in range(1, steps + 1):
            speed, s = ballistic_step(
                speed=speed, s=s, acceleration=acceleration, dt=dt, road=road
            )
            elapsed += dt
            if maneuver is None:
                d = ego.d
                lateral_rate = 0.0
            else:
                d = maneuver.profile.offset(elapsed)
                lateral_rate = maneuver.profile.rate(elapsed)
            acceleration = self._acceleration_at(s, speed, leaders, index * dt, ego)
            points.append(
                TrajectoryPoint(
                    time=index * dt,
                    s=s,
                    d=d,
                    speed=speed,
                    acceleration=acceleration,
                    lateral_rate=lateral_rate,
                )
            )
        return tuple(points)

    def _frozen_leaders(
        self, context: DecisionContext, state: BehaviorState
    ) -> tuple[tuple[float, float, float], ...]:
        """Leaders of the binding lanes as ``(s, speed, half length)`` triples."""
        found = []
        for lane in binding_lanes(context.road, context.ego, state):
            leader = context.snapshot.neighbors(context.ego, lane).leader
            if leader is not None:
                found.append(
                    (leader.vehicle.s, leader.speed, 0.5 * leader.vehicle.shape.length)
                )
        return tuple(found)

    def _acceleration_at(
        self,
        s: float,
        speed: float,
        leaders: tuple[tuple[float, float, float], ...],
        time: float,
        ego: Vehicle,
    ) -> float:
        """Acceleration at a predicted pose, leaders extrapolated at constant speed."""
        if not leaders:
            return self.car_following.acceleration(
                speed=speed, gap=math.inf, leader_speed=0.0, driver=ego.driver
            )
        commands = []
        for leader_s, leader_speed, half_length in leaders:
            predicted = leader_s + leader_speed * time
            gap = (predicted - s) - half_length - 0.5 * ego.shape.length
            commands.append(
                self.car_following.acceleration(
                    speed=speed, gap=gap, leader_speed=leader_speed, driver=ego.driver
                )
            )
        return min(commands)
