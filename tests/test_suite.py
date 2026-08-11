"""Tier one. Invariants that must hold across the whole scenario suite."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import pytest

from behavior_planner.algorithm.fsm import is_legal
from behavior_planner.analysis.metrics import scenario_metrics, suite_metrics
from behavior_planner.model.decision import VetoReason
from behavior_planner.model.states import BehaviorState
from behavior_planner.pipeline.scenarios import Scenario, build_vehicles, standard_suite
from behavior_planner.pipeline.simulator import SimulationConfig
from behavior_planner.pipeline.suite import build_planner, run_baseline, run_scenario, run_suite
from behavior_planner.pipeline.trace import RunTrace

SUITE = standard_suite()
NAMES = [scenario.name for scenario in SUITE]


@pytest.fixture(scope="module")
def traces() -> tuple[RunTrace, ...]:
    """Every scenario run once under the default planner."""
    return run_suite(SUITE)


@pytest.fixture(scope="module")
def baselines() -> tuple[RunTrace, ...]:
    """Every scenario run once under the lane keeping control policy."""
    return run_suite(SUITE, baseline=True)


def test_the_suite_produces_no_collisions(traces: tuple[RunTrace, ...]) -> None:
    """The requirement the whole safety layer exists to meet."""
    for trace in traces:
        assert trace.collision_pairs == (), trace.scenario
    assert suite_metrics(traces).total_collisions == 0


def test_the_control_policy_also_produces_no_collisions(
    baselines: tuple[RunTrace, ...],
) -> None:
    """Collision freedom comes from the driver models too, not only from the planner."""
    assert suite_metrics(baselines).total_collisions == 0


@pytest.mark.parametrize("index", range(len(SUITE)))
def test_every_recorded_state_transition_is_legal(traces: tuple[RunTrace, ...], index: int) -> None:
    """No run reaches a state the transition table does not allow."""
    trace = traces[index]
    for previous, current in zip(trace.records, trace.records[1:], strict=False):
        if current.planned:
            assert is_legal(previous.state, current.event)


@pytest.mark.parametrize("index", range(len(SUITE)))
def test_speeds_are_non_negative_and_below_a_physical_bound(
    traces: tuple[RunTrace, ...], index: int
) -> None:
    """The integrator never reverses the ego and never runs it away."""
    trace = traces[index]
    for record in trace.records:
        assert record.speed >= 0.0
        assert record.speed < 40.0


@pytest.mark.parametrize("index", range(len(SUITE)))
def test_the_ego_stays_on_the_carriageway(traces: tuple[RunTrace, ...], index: int) -> None:
    """Lateral offset never leaves the outer lane centres."""
    road = SUITE[index].road
    trace = traces[index]
    for record in trace.records:
        assert -0.01 <= record.d <= road.lane_center(road.lane_count - 1) + 0.01
        assert road.contains_lane(record.lane)


@pytest.mark.parametrize("index", range(len(SUITE)))
def test_time_headway_and_time_to_collision_are_non_negative(
    traces: tuple[RunTrace, ...], index: int
) -> None:
    """A negative time would mean the metric had been computed on an overlap."""
    for record in traces[index].records:
        assert record.time_headway >= 0.0
        assert record.time_to_collision >= 0.0


def test_the_suite_exercises_both_lane_change_directions(
    traces: tuple[RunTrace, ...],
) -> None:
    """The suite is not one sided: the ego goes out and comes back."""
    seen: set[BehaviorState] = set()
    for trace in traces:
        seen.update(trace.state_sequence)
    assert BehaviorState.LANE_CHANGE_LEFT in seen
    assert BehaviorState.LANE_CHANGE_RIGHT in seen
    assert BehaviorState.PREPARE_LANE_CHANGE_LEFT in seen
    assert BehaviorState.PREPARE_LANE_CHANGE_RIGHT in seen


def test_the_suite_exercises_the_safety_gate(traces: tuple[RunTrace, ...]) -> None:
    """At least one scenario is refused a manoeuvre, or the gate is untested by the suite."""
    raised = {reason for trace in traces for reason in trace.veto_reasons}
    assert raised - {VetoReason.NONE}


def test_the_planner_is_never_slower_than_the_control_by_more_than_a_little(
    traces: tuple[RunTrace, ...], baselines: tuple[RunTrace, ...]
) -> None:
    """Lane changing pays on average and never costs much in the cases where it does not.

    A prepared change that is refused costs the ego some speed, because
    preparing means dropping back. Bounding that loss is a real requirement: a
    planner that hunts for gaps it never gets is worse than one that stays put.
    """
    for planned, control in zip(traces, baselines, strict=True):
        gain = planned.distance_travelled / control.distance_travelled
        assert gain > 0.97, planned.scenario
    total_planned = sum(trace.distance_travelled for trace in traces)
    total_control = sum(trace.distance_travelled for trace in baselines)
    assert total_planned > total_control


def test_overtaking_scenarios_beat_the_control_substantially(
    traces: tuple[RunTrace, ...], baselines: tuple[RunTrace, ...]
) -> None:
    """Where a lane change is available and useful, the planner takes it."""
    planned = {trace.scenario: trace for trace in traces}
    control = {trace.scenario: trace for trace in baselines}
    for name in ("slow_leader", "slow_right_lane"):
        ratio = planned[name].distance_travelled / control[name].distance_travelled
        assert ratio > 1.2, name


def test_the_blocked_scenario_produces_no_lane_change(
    traces: tuple[RunTrace, ...],
) -> None:
    """With every adjacent gap occupied at matching speed, the ego holds its lane."""
    blocked = next(trace for trace in traces if trace.scenario == "blocked_overtake")
    assert blocked.lane_changes == 0
    assert {record.lane for record in blocked.records} == {0}


def test_free_flow_reaches_the_desired_speed() -> None:
    """On an empty road the ego converges to its own free flow speed."""
    scenario = next(item for item in SUITE if item.name == "free_flow")
    trace = run_scenario(scenario.with_duration(180.0))
    desired = scenario.ego.desired_speed
    assert trace.records[-1].speed == pytest.approx(desired, rel=1e-3)
    assert trace.lane_changes == 0
    assert set(trace.state_sequence) == {BehaviorState.KEEP_LANE}


@pytest.mark.parametrize("index", range(len(SUITE)))
def test_each_scenario_is_reproducible(index: int) -> None:
    """Two runs of the same scenario agree record for record."""
    scenario = SUITE[index].with_duration(20.0)
    assert run_scenario(scenario).records == run_scenario(scenario).records


@pytest.mark.parametrize("index", range(len(SUITE)))
def test_vehicle_placement_is_reproducible(index: int) -> None:
    """The seeded fill depends on the seed and on nothing else."""
    scenario = SUITE[index]
    assert build_vehicles(scenario) == build_vehicles(scenario)


def test_placement_leaves_the_declared_clearance() -> None:
    """No run starts in a conflict, which would corrupt the headway metric."""
    for scenario in SUITE:
        vehicles = build_vehicles(scenario)
        road = scenario.road
        for first in vehicles:
            for second in vehicles:
                if first.vehicle_id >= second.vehicle_id:
                    continue
                if road.nearest_lane(first.d) != road.nearest_lane(second.d):
                    continue
                separation = abs(road.separation(first.s, second.s))
                assert separation > scenario.shape.length, scenario.name


def test_a_shorter_run_is_a_prefix_of_a_longer_one() -> None:
    """The step budget truncates a run rather than changing it."""
    scenario = next(item for item in SUITE if item.name == "slow_leader")
    short = run_scenario(scenario, steps=100)
    long = run_scenario(scenario, steps=200)
    assert short.records == long.records[: len(short.records)]


def test_the_planning_period_is_honoured() -> None:
    """The behaviour layer runs at its own rate, not at the integration step."""
    scenario = next(item for item in SUITE if item.name == "slow_leader")
    trace = run_scenario(scenario.with_duration(10.0))
    simulation = SimulationConfig()
    from behavior_planner.model.config import PlannerConfig

    expected = round(PlannerConfig().planning_period / simulation.dt)
    planned = [index for index, record in enumerate(trace.records) if record.planned]
    assert planned[0] == 0
    assert all(second - first == expected for first, second in pairwise(planned))


def test_metrics_reduce_a_trace_without_losing_the_collision_count(
    traces: tuple[RunTrace, ...],
) -> None:
    """Every scenario metric is finite or explicitly unbounded, never a silent zero."""
    for trace in traces:
        metrics = scenario_metrics(trace)
        assert metrics.collisions == trace.collision_count
        assert metrics.lane_changes == trace.lane_changes
        assert metrics.mean_speed > 0.0
        assert metrics.minimum_time_headway > 0.0
        assert math.isinf(metrics.minimum_time_to_collision) or (
            metrics.minimum_time_to_collision > 0.0
        )


def test_a_suite_must_use_one_policy(traces: tuple[RunTrace, ...]) -> None:
    """Mixing policies in one aggregate would make the mean speed meaningless."""
    mixed = (traces[0], run_baseline(SUITE[0].with_duration(5.0)))
    with pytest.raises(ValueError, match="one policy"):
        suite_metrics(mixed)


def test_an_ego_off_the_road_is_rejected() -> None:
    """A scenario that cannot be built raises at construction, not at run time."""
    scenario = SUITE[0]
    with pytest.raises(ValueError, match="ego lane"):
        replace(scenario, ego=replace(scenario.ego, lane=99))


def test_the_planner_and_the_gate_are_wired_from_one_configuration() -> None:
    """Changing the configuration changes both layers, and nothing is hard coded."""
    from behavior_planner.model.config import PlannerConfig, SafetyLimits

    config = PlannerConfig(safety=SafetyLimits(minimum_leader_gap=42.0))
    planner = build_planner(config)
    assert planner.gate.limits.minimum_leader_gap == pytest.approx(42.0)  # type: ignore[attr-defined]
    assert planner.cost.config is config.cost  # type: ignore[attr-defined]


def test_scenario_names_are_unique() -> None:
    """The suite is indexed by name in the reports, so the names must be distinct."""
    assert len(NAMES) == len(set(NAMES))


def test_every_scenario_declares_a_description() -> None:
    """A scenario without a stated purpose cannot be judged against its result."""
    for scenario in SUITE:
        assert isinstance(scenario, Scenario)
        assert scenario.description.strip()
        assert scenario.duration > 0.0
