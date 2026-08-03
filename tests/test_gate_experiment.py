"""Tier one. The experiment the README opens with, checked rather than quoted.

Every number in the opening section of the README comes from
:func:`gate_weight_sweep`. This module asserts the properties those numbers are
claimed to have, so a change that quietly made the claim false would fail here
rather than leaving the README saying something the code no longer does.
"""

from __future__ import annotations

import pytest

from behavior_planner.model.config import CostWeights, PlannerConfig
from behavior_planner.model.decision import VetoReason
from behavior_planner.model.states import BehaviorState
from behavior_planner.pipeline.gate_experiment import (
    PROGRESS_WEIGHTS,
    blocked_merge_context,
    gate_weight_sweep,
)

SAMPLES = gate_weight_sweep()


def test_the_sweep_covers_three_orders_of_magnitude() -> None:
    """The claim is about the range, so the range is part of the contract."""
    assert PROGRESS_WEIGHTS[0] == CostWeights().progress
    assert PROGRESS_WEIGHTS[-1] == 1000.0 * PROGRESS_WEIGHTS[0]
    assert len(SAMPLES) == len(PROGRESS_WEIGHTS)
    assert list(PROGRESS_WEIGHTS) == sorted(PROGRESS_WEIGHTS)


def test_the_cost_function_prefers_the_lane_change_at_every_weight() -> None:
    """Without this the gate would never be asked, and the experiment would be empty."""
    assert all(sample.cost_prefers_change for sample in SAMPLES)
    assert all(sample.cost_margin > 0.0 for sample in SAMPLES)


def test_the_gate_refuses_the_lane_change_at_every_weight() -> None:
    """The property a cost term could not have, over the whole range."""
    assert all(not sample.allowed for sample in SAMPLES)
    assert {sample.reason for sample in SAMPLES} == {VetoReason.OCCUPIED}


def test_the_planner_never_takes_the_manoeuvre() -> None:
    """The gate is not advisory. The planner follows it, not the ranking."""
    assert all(sample.chosen is not BehaviorState.LANE_CHANGE_LEFT for sample in SAMPLES)


def test_the_cost_preference_grows_while_the_verdict_does_not_move() -> None:
    """The two halves of the claim, measured against each other.

    A weighted sum makes the manoeuvre look better and better as the weight
    rises. The verdict is not a term in that sum, so it does not notice.
    """
    margins = [sample.cost_margin for sample in SAMPLES]
    assert margins == sorted(margins)
    assert margins[-1] / margins[0] > 100.0
    assert len({sample.allowed for sample in SAMPLES}) == 1


def test_the_soft_safety_weight_is_not_what_refuses_the_manoeuvre() -> None:
    """Restoring the soft term must change the ranking and not the verdict.

    Both halves matter. If the term changed nothing it would be dead weight, and
    if the veto moved when the cost changed, the two would not be separate and
    the whole argument would be wrong.
    """
    with_term = gate_weight_sweep(safety_weight=CostWeights().safety)
    assert all(not sample.allowed for sample in with_term)
    assert [sample.reason for sample in with_term] == [sample.reason for sample in SAMPLES]
    assert all(
        soft.lane_change_cost > bare.lane_change_cost
        for soft, bare in zip(with_term, SAMPLES, strict=True)
    )
    assert with_term[0].chosen is not SAMPLES[0].chosen


def test_the_scene_is_the_one_the_claim_describes() -> None:
    """A slow leader ahead in the ego lane and a vehicle beside it, just behind."""
    context = blocked_merge_context(PlannerConfig())
    assert context.state is BehaviorState.PREPARE_LANE_CHANGE_LEFT
    assert context.ego_lane == 0
    others = [
        vehicle
        for vehicle in context.snapshot.vehicles
        if vehicle.vehicle_id != context.ego.vehicle_id
    ]
    leader = next(vehicle for vehicle in others if vehicle.d == context.ego.d)
    beside = next(vehicle for vehicle in others if vehicle.d != context.ego.d)
    assert leader.speed < context.ego.speed
    assert context.road.separation(context.ego.s, leader.s) > 0.0
    assert context.road.separation(context.ego.s, beside.s) < 0.0


def test_a_sweep_over_no_weights_is_rejected() -> None:
    """An empty sweep proves nothing and must not look like a passing experiment."""
    with pytest.raises(ValueError, match="at least one"):
        gate_weight_sweep(())
