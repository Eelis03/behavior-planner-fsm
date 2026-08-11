"""Tier one and two. How much room the gate left, not only where it refused.

A veto reason says which manoeuvres the gate stopped. It says nothing about the
manoeuvres it allowed, so a gate that permitted every change by a hair and a gate
that permitted them with a lane to spare produce identical traces. The margin the
verdict already carries is the quantity that separates them, and this module
covers its journey from the verdict to the metric.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner, KeepLaneBaseline
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.algorithm.trajectory import lane_change_for
from behavior_planner.analysis.metrics import scenario_metrics, suite_metrics
from behavior_planner.model.config import PlannerConfig
from behavior_planner.model.decision import DecisionContext
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorEvent, BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_scenario, run_suite
from behavior_planner.pipeline.trace import RunTrace
from tests.conftest import make_ego, make_vehicle

SHORT = tuple(scenario.with_duration(20.0) for scenario in standard_suite())


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


def committing_scene(road: Road, config: PlannerConfig) -> tuple[DecisionContext, BehaviorState]:
    """A prepared change the gate permits, with a neighbour close enough to bind.

    An empty target lane leaves the verdict unbounded, which is correct and is
    also the one case in which the margin says nothing, so the left lane here
    carries a leader far enough ahead to be permitted and near enough to be
    measured.
    """
    ego = make_ego(road, lane=0, s=0.0, speed=18.0, desired_speed=31.0)
    blocker = make_vehicle(road, vehicle_id=1, lane=0, s=25.0, speed=15.0, desired_speed=15.0)
    ahead = make_vehicle(road, vehicle_id=2, lane=1, s=120.0, speed=30.0, desired_speed=31.0)
    scene = context(
        road, ego, blocker, ahead, state=BehaviorState.PREPARE_LANE_CHANGE_LEFT, config=config
    )
    return scene, BehaviorState.LANE_CHANGE_LEFT


@pytest.fixture(scope="module")
def traces() -> tuple[RunTrace, ...]:
    """Short runs of every scenario under the default planner."""
    return run_suite(SHORT)


def test_a_decision_that_moves_nothing_sideways_is_unbounded(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """The gate rules on lateral motion, so lane keeping has no margin to report."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    decision = planner.decide(scene)
    assert not decision.state.is_changing
    assert math.isinf(decision.gate_margin)


def test_a_committed_change_carries_the_margin_it_was_permitted_by(
    road: Road,
    planner: FiniteStateBehaviorPlanner,
    gate: GapAndDecelerationGate,
    config: PlannerConfig,
) -> None:
    """The number on the decision is the number the gate produced, not a recomputation."""
    scene, change = committing_scene(road, config)
    decision = planner.decide(scene)
    assert decision.state is change
    assert decision.event is BehaviorEvent.COMMIT
    assert math.isfinite(decision.gate_margin)
    assert decision.gate_margin > 0.0
    assert decision.gate_margin == pytest.approx(gate.review(scene, change).margin)


def test_a_running_change_reports_the_margin_of_its_latest_review(
    road: Road,
    planner: FiniteStateBehaviorPlanner,
    gate: GapAndDecelerationGate,
    config: PlannerConfig,
) -> None:
    """The gate rules again every cycle while the manoeuvre runs, and the number survives."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    moving = replace(ego, lane_change=replace(maneuver, elapsed=1.0))
    ahead = make_vehicle(road, vehicle_id=1, lane=1, s=90.0, speed=30.0, desired_speed=31.0)
    scene = context(road, moving, ahead, state=BehaviorState.LANE_CHANGE_LEFT, config=config)
    decision = planner.decide(scene)
    assert decision.event is BehaviorEvent.STAY
    assert math.isfinite(decision.gate_margin)
    assert decision.gate_margin == pytest.approx(
        gate.review(scene, BehaviorState.LANE_CHANGE_LEFT).margin
    )


def test_a_manoeuvre_finished_over_the_gate_s_objection_reports_a_negative_margin(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """Past the abort limit the ego finishes anyway, and the trace has to say so.

    This is the one case in which a decision puts a manoeuvre in force that the
    gate would refuse. Reporting it as unbounded would hide the only situation in
    which the gate is overruled.
    """
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
    assert decision.gate_margin < 0.0


def test_an_abandoned_change_leaves_the_margin_unbounded(
    road: Road, planner: FiniteStateBehaviorPlanner, config: PlannerConfig
) -> None:
    """An abort adopts lane keeping, which is not gated; the veto reason carries the refusal."""
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
    assert math.isinf(decision.gate_margin)
    assert decision.vetoed
    assert decision.vetoed[0].verdict.margin < 0.0


def test_the_control_policy_bounds_nothing(road: Road, config: PlannerConfig) -> None:
    """A policy that never leaves its lane never asks the gate for room."""
    ego = make_ego(road, lane=1, s=0.0, speed=12.0, desired_speed=31.0)
    blocker = make_vehicle(road, vehicle_id=1, lane=1, s=15.0, speed=10.0)
    scene = context(road, ego, blocker, state=BehaviorState.KEEP_LANE, config=config)
    assert math.isinf(KeepLaneBaseline().decide(scene).gate_margin)


def test_the_trace_bounds_a_margin_only_where_the_gate_ruled(
    traces: tuple[RunTrace, ...],
) -> None:
    """A bounded margin implies a planning cycle that adopted a lateral manoeuvre."""
    for trace in traces:
        for record in trace.records:
            if math.isfinite(record.gate_margin):
                assert record.planned, trace.scenario
                assert record.state.is_changing, trace.scenario


def test_the_run_minimum_is_the_smallest_bounded_margin(
    traces: tuple[RunTrace, ...],
) -> None:
    """The run level number is a minimum over the samples, not over the steps."""
    for trace in traces:
        margins = trace.gate_margins()
        assert all(math.isfinite(margin) for margin in margins)
        if margins:
            assert trace.minimum_gate_margin == min(margins)
        else:
            assert math.isinf(trace.minimum_gate_margin)
        assert scenario_metrics(trace).gate_margin_samples == len(margins)


def test_a_run_that_never_changes_lane_has_no_margin_to_report(
    traces: tuple[RunTrace, ...],
) -> None:
    """An empty road produces no manoeuvre, so the gate bounds nothing all run."""
    free_flow = next(trace for trace in traces if trace.scenario == "free_flow")
    assert free_flow.lane_changes == 0
    assert free_flow.gate_margins() == ()
    metrics = scenario_metrics(free_flow)
    assert metrics.gate_margin_samples == 0
    assert math.isinf(metrics.minimum_gate_margin)


def test_the_suite_takes_no_manoeuvre_the_gate_had_come_to_refuse(
    traces: tuple[RunTrace, ...],
) -> None:
    """The gate is never overruled anywhere in the suite, and the margin proves it.

    The collision count says no manoeuvre ended badly. This says none of them
    was still running when the gate changed its mind, which is a stronger claim
    and the one the abort path exists to support.
    """
    metrics = suite_metrics(traces)
    assert metrics.minimum_gate_margin >= 0.0
    assert any(item.gate_margin_samples > 0 for item in metrics.scenarios)
    assert metrics.minimum_gate_margin == min(
        item.minimum_gate_margin for item in metrics.scenarios
    )


def test_the_margin_tightens_when_the_gate_is_relaxed() -> None:
    """The number tracks the thresholds it is normalised by, or it measures nothing.

    Halving the required gaps leaves the same geometry a larger share of the
    requirement, so every manoeuvre the suite takes is reported with more room
    than the default limits give it.
    """
    from behavior_planner.model.config import SafetyLimits

    scenario = next(item for item in SHORT if item.name == "slow_leader")
    strict = run_scenario(scenario)
    relaxed = run_scenario(
        scenario,
        config=PlannerConfig(safety=SafetyLimits(minimum_leader_gap=2.5, minimum_follower_gap=2.5)),
    )
    assert math.isfinite(strict.minimum_gate_margin)
    assert relaxed.minimum_gate_margin > strict.minimum_gate_margin
