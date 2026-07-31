"""Tier one. The behaviour policy: overtaking, refusing, and staying deterministic."""

from __future__ import annotations

from dataclasses import replace

import pytest

from behavior_planner.algorithm.fsm import is_legal
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner, KeepLaneBaseline
from behavior_planner.algorithm.trajectory import lane_change_for
from behavior_planner.model.config import PlannerConfig
from behavior_planner.model.decision import DecisionContext
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorEvent, BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle
from behavior_planner.pipeline.scenarios import Scenario, VehicleSpec
from behavior_planner.pipeline.suite import run_scenario
from tests.conftest import make_ego, make_vehicle


def context(
    road: Road, ego: Vehicle, *others: Vehicle, state: BehaviorState, config: PlannerConfig
) -> DecisionContext:
    """A decision context holding the ego and the given traffic."""
    return DecisionContext(
        road=road,
        snapshot=TrafficSnapshot(road=road, vehicles=(ego, *others)),
        ego=ego,
        state=state,
        config=config,
    )


def overtaking_scenario(*, blocked: bool) -> Scenario:
    """A slow leader ahead, with the adjacent lane clear or occupied."""
    road = Road(lane_count=3, length=1200.0)
    blockers = tuple(
        VehicleSpec(lane=1, s=offset, speed=20.0, desired_speed=20.0, holds_lane=True)
        for offset in (-30.0, 0.0, 30.0, 60.0, 90.0)
    )
    return Scenario(
        name="blocked" if blocked else "clear",
        description="Overtaking test.",
        road=road,
        ego=VehicleSpec(lane=0, s=0.0, speed=28.0, desired_speed=31.0),
        duration=30.0,
        seed=0,
        scripted=(
            VehicleSpec(lane=0, s=60.0, speed=20.0, desired_speed=20.0, holds_lane=True),
            *(blockers if blocked else ()),
        ),
    )


def test_the_ego_overtakes_a_slow_leader_when_the_adjacent_lane_is_clear() -> None:
    """The reason the planner exists, stated as a test."""
    trace = run_scenario(overtaking_scenario(blocked=False))
    assert trace.collision_count == 0
    assert trace.lane_changes >= 1
    assert BehaviorState.PREPARE_LANE_CHANGE_LEFT in trace.state_sequence
    assert BehaviorState.LANE_CHANGE_LEFT in trace.state_sequence
    assert max(record.lane for record in trace.records) >= 1


def test_the_ego_does_not_overtake_when_the_adjacent_lane_is_blocked() -> None:
    """The same slow leader, with nowhere to go, produces no lane change."""
    trace = run_scenario(overtaking_scenario(blocked=True))
    assert trace.collision_count == 0
    assert trace.lane_changes == 0
    assert {record.lane for record in trace.records} == {0}
    assert BehaviorState.LANE_CHANGE_LEFT not in trace.state_sequence


def test_overtaking_is_what_makes_the_ego_faster() -> None:
    """The clear case beats the blocked case, which is the whole benefit."""
    clear = run_scenario(overtaking_scenario(blocked=False))
    blocked = run_scenario(overtaking_scenario(blocked=True))
    assert clear.distance_travelled > blocked.distance_travelled


def test_every_decision_uses_a_legal_transition(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """The policy can only reach states the machine allows, from every state."""
    ego = make_ego(road, lane=1, s=0.0, speed=24.0, desired_speed=31.0)
    crowd = [
        make_vehicle(road, vehicle_id=index + 1, lane=lane, s=offset, speed=22.0)
        for index, (lane, offset) in enumerate([(0, 40.0), (1, 35.0), (2, -30.0)])
    ]
    for state in BehaviorState:
        subject = ego
        if state.is_changing:
            subject = replace(
                ego,
                lane_change=lane_change_for(road, ego, state, config.lane_change_duration),
            )
        scene = context(road, subject, *crowd, state=state, config=config)
        decision = planner.decide(scene)
        assert is_legal(state, decision.event)


def test_a_committed_manoeuvre_is_not_re_optimised(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """A running lane change continues; it does not re-enter the cost comparison."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    moving = replace(ego, lane_change=replace(maneuver, elapsed=1.0))
    scene = context(road, moving, state=BehaviorState.LANE_CHANGE_LEFT, config=config)
    decision = planner.decide(scene)
    assert decision.state is BehaviorState.LANE_CHANGE_LEFT
    assert decision.event is BehaviorEvent.STAY
    assert decision.candidates == ()


def test_a_finished_manoeuvre_is_retired_with_complete(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """Arrival is signalled by ``COMPLETE``, which is how lane changes are counted."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    arrived = replace(
        ego,
        d=road.lane_center(1),
        lane_change=replace(maneuver, elapsed=config.lane_change_duration),
    )
    scene = context(road, arrived, state=BehaviorState.LANE_CHANGE_LEFT, config=config)
    decision = planner.decide(scene)
    assert decision.event is BehaviorEvent.COMPLETE
    assert decision.state is BehaviorState.KEEP_LANE


def test_an_early_manoeuvre_is_aborted_when_the_gate_objects(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """A change that has barely started is given up if the gap closes."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    moving = replace(ego, lane_change=replace(maneuver, elapsed=0.2))
    intruder = make_vehicle(road, vehicle_id=1, lane=1, s=3.0, speed=28.0)
    scene = context(road, moving, intruder, state=BehaviorState.LANE_CHANGE_LEFT, config=config)
    decision = planner.decide(scene)
    assert decision.event is BehaviorEvent.ABORT
    assert decision.state is BehaviorState.KEEP_LANE


def test_a_late_manoeuvre_is_not_aborted(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """Past the abort limit the ego finishes, because reversing is now the worse option."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    elapsed = config.lane_change_duration * (config.abort_progress_limit + 0.2)
    moving = replace(
        ego, d=maneuver.profile.offset(elapsed), lane_change=replace(maneuver, elapsed=elapsed)
    )
    intruder = make_vehicle(road, vehicle_id=1, lane=1, s=3.0, speed=28.0)
    scene = context(road, moving, intruder, state=BehaviorState.LANE_CHANGE_LEFT, config=config)
    decision = planner.decide(scene)
    assert decision.event is BehaviorEvent.STAY
    assert decision.state is BehaviorState.LANE_CHANGE_LEFT


def test_the_edge_lanes_offer_no_manoeuvre_off_the_road(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """An infeasible successor is dropped before scoring, not vetoed."""
    right = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, right, state=BehaviorState.KEEP_LANE, config=config)
    states = {candidate.state for candidate in planner.decide(scene).candidates}
    assert BehaviorState.PREPARE_LANE_CHANGE_RIGHT not in states

    left = make_ego(road, lane=2, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, left, state=BehaviorState.KEEP_LANE, config=config)
    states = {candidate.state for candidate in planner.decide(scene).candidates}
    assert BehaviorState.PREPARE_LANE_CHANGE_LEFT not in states


def test_the_baseline_policy_never_leaves_its_lane(road: Road, config: PlannerConfig) -> None:
    """The control policy is a control: it satisfies the same interface and does nothing."""
    ego = make_ego(road, lane=1, s=0.0, speed=12.0, desired_speed=31.0)
    blocker = make_vehicle(road, vehicle_id=1, lane=1, s=15.0, speed=10.0)
    scene = context(road, ego, blocker, state=BehaviorState.KEEP_LANE, config=config)
    decision = KeepLaneBaseline().decide(scene)
    assert decision.state is BehaviorState.KEEP_LANE
    assert decision.event is BehaviorEvent.STAY


def test_a_run_is_reproducible_from_its_seed() -> None:
    """Two runs of the same scenario agree in every recorded field."""
    scenario = overtaking_scenario(blocked=False)
    first = run_scenario(scenario)
    second = run_scenario(scenario)
    assert first.records == second.records
    assert first.collision_pairs == second.collision_pairs


def test_different_seeds_give_different_traffic() -> None:
    """The seed is not decorative: it selects the random fill."""
    from behavior_planner.pipeline.scenarios import RandomFill

    base = Scenario(
        name="seeded",
        description="Random fill.",
        road=Road(lane_count=3, length=1200.0),
        ego=VehicleSpec(lane=0, s=0.0, speed=22.0, desired_speed=31.0),
        duration=20.0,
        seed=1,
        fill=RandomFill(lanes=(0, 1, 2), per_lane=8, speed_mean=22.0, speed_gradient=3.0),
    )
    first = run_scenario(base)
    second = run_scenario(replace(base, seed=2))
    assert first.records != second.records


def test_a_run_needs_at_least_one_step() -> None:
    """A zero length run has no trace, so it raises instead of producing one."""
    with pytest.raises(ValueError, match="at least one step"):
        run_scenario(overtaking_scenario(blocked=False), steps=0)
