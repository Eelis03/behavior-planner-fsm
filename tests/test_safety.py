"""Tier one. The safety gate, including its independence from the cost function.

The central test in this module constructs the conflict the gate exists to
resolve: a situation in which the cost function ranks a lane change first and
the gate refuses it anyway.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from behavior_planner.algorithm.cost import WeightedCostModel
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.model.config import CostConfig, CostWeights, PlannerConfig, SafetyLimits
from behavior_planner.model.decision import DecisionContext, VetoReason
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle
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


def test_the_gate_ignores_states_with_no_lateral_motion(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """Lane keeping and preparing move nothing sideways, so there is nothing to veto."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    beside = make_vehicle(road, vehicle_id=1, lane=0, s=1.0, speed=28.0)
    scene = context(road, ego, beside, state=BehaviorState.KEEP_LANE, config=config)
    for state in (
        BehaviorState.KEEP_LANE,
        BehaviorState.PREPARE_LANE_CHANGE_RIGHT,
        BehaviorState.PREPARE_LANE_CHANGE_LEFT,
    ):
        verdict = gate.review(scene, state)
        assert verdict.allowed
        assert verdict.reason is VetoReason.NONE


def test_a_clear_lane_is_permitted(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """With nothing in the target lane there is no objection and no bound on the margin."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert verdict.allowed
    assert math.isinf(verdict.margin)


def test_a_change_off_the_road_is_vetoed(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """The rightmost lane has nothing to its right."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_RIGHT)
    assert not verdict.allowed
    assert verdict.reason is VetoReason.OFF_ROAD


def test_a_vehicle_abreast_of_the_ego_is_reported_as_occupied(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """A negative gap is not a small gap, and gets its own veto reason."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    beside = make_vehicle(road, vehicle_id=1, lane=1, s=2.0, speed=28.0)
    scene = context(road, ego, beside, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert not verdict.allowed
    assert verdict.reason is VetoReason.OCCUPIED


def test_a_leader_inside_the_minimum_gap_is_vetoed(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """The required leader gap grows with the ego's speed."""
    ego = make_ego(road, lane=0, s=0.0, speed=30.0, desired_speed=31.0)
    required = config.safety.minimum_leader_gap + config.safety.leader_headway * ego.speed
    ahead = make_vehicle(road, vehicle_id=1, lane=1, s=required + 4.0, speed=30.0)
    scene = context(road, ego, ahead, state=BehaviorState.KEEP_LANE, config=config)
    assert not gate.review(scene, BehaviorState.LANE_CHANGE_LEFT).allowed

    roomy = replace(ahead, s=road.wrap(required + 40.0))
    scene = context(road, ego, roomy, state=BehaviorState.KEEP_LANE, config=config)
    assert gate.review(scene, BehaviorState.LANE_CHANGE_LEFT).allowed


def test_a_follower_inside_the_minimum_gap_is_vetoed(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """The gate protects the vehicle behind as well as the vehicle in front."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    behind = make_vehicle(road, vehicle_id=1, lane=1, s=-12.0, speed=28.0)
    scene = context(road, ego, behind, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert not verdict.allowed
    assert verdict.reason in {VetoReason.FOLLOWER_GAP, VetoReason.FOLLOWER_DECELERATION}


def test_a_closing_leader_is_vetoed_on_time_to_collision(
    road: Road, gate: GapAndDecelerationGate, config: PlannerConfig
) -> None:
    """A gap that satisfies the distance rule can still fail the time rule."""
    ego = make_ego(road, lane=0, s=0.0, speed=32.0, desired_speed=33.0)
    required = config.safety.minimum_leader_gap + config.safety.leader_headway * ego.speed
    ahead = make_vehicle(road, vehicle_id=1, lane=1, s=required + 12.0, speed=15.0)
    scene = context(road, ego, ahead, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert not verdict.allowed
    assert verdict.reason is VetoReason.LEADER_TIME_TO_COLLISION


def test_a_closing_follower_is_vetoed_on_time_to_collision(
    road: Road, config: PlannerConfig, car_following: object
) -> None:
    """A fast follower far enough back by distance can still be too close in time.

    The deceleration check is relaxed for this case so that the time to
    collision rule is the one under test.
    """
    limits = SafetyLimits(maximum_follower_deceleration=50.0)
    gate = GapAndDecelerationGate(limits=limits, car_following=car_following)  # type: ignore[arg-type]
    ego = make_ego(road, lane=0, s=0.0, speed=16.0, desired_speed=31.0)
    behind = make_vehicle(road, vehicle_id=1, lane=1, s=-40.0, speed=32.0)
    scene = context(road, ego, behind, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert not verdict.allowed
    assert verdict.reason is VetoReason.FOLLOWER_TIME_TO_COLLISION


def test_a_manoeuvre_that_forces_hard_braking_is_vetoed(
    road: Road, config: PlannerConfig, car_following: object
) -> None:
    """The gate holds the ego to the same standard MOBIL holds the traffic to.

    The geometric rules are disabled here so that only the deceleration check
    can fire, which is what makes this test about that check.
    """
    limits = SafetyLimits(
        minimum_follower_gap=0.5,
        follower_headway=0.0,
        minimum_leader_gap=0.5,
        leader_headway=0.0,
        minimum_time_to_collision=0.01,
        maximum_follower_deceleration=3.0,
    )
    gate = GapAndDecelerationGate(limits=limits, car_following=car_following)  # type: ignore[arg-type]
    ego = make_ego(road, lane=0, s=0.0, speed=20.0, desired_speed=31.0)
    behind = make_vehicle(road, vehicle_id=1, lane=1, s=-14.0, speed=24.0)
    scene = context(road, ego, behind, state=BehaviorState.KEEP_LANE, config=config)
    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert not verdict.allowed
    assert verdict.reason is VetoReason.FOLLOWER_DECELERATION


def test_the_gate_vetoes_a_change_the_cost_function_ranks_first(
    road: Road, gate: GapAndDecelerationGate
) -> None:
    """The conflict the gate exists to resolve, constructed exactly.

    The ego is stuck behind a slow leader, so the progress term prefers the
    neighbouring lane, whose own leader is far away. The gap it would move into
    is occupied: a vehicle sits alongside the ego, just behind it, where it
    lowers no cost term but leaves nowhere to go. The soft safety weight is set
    to zero, which is the misconfiguration the gate is the defence against: with
    it the cost function ranks the lane change first, and it is still refused.

    The three assertions are deliberately separate. The first shows the cost
    function prefers the change, the second shows the gate refuses it, and the
    third shows the planner follows the gate rather than the cost.
    """
    config = PlannerConfig(
        cost=CostConfig(
            weights=CostWeights(progress=5.0, safety=0.0, comfort=0.10, lane_preference=0.35)
        )
    )
    cost = WeightedCostModel(config.cost)
    planner = FiniteStateBehaviorPlanner(cost=cost, gate=gate)

    ego = make_ego(road, lane=0, s=0.0, speed=18.0, desired_speed=31.0)
    blocker = make_vehicle(road, vehicle_id=1, lane=0, s=25.0, speed=15.0, desired_speed=15.0)
    beside = make_vehicle(road, vehicle_id=2, lane=1, s=-1.5, speed=26.0, desired_speed=31.0)
    scene = context(
        road, ego, blocker, beside, state=BehaviorState.PREPARE_LANE_CHANGE_LEFT, config=config
    )

    weights = config.cost.weights
    change_cost = cost.evaluate(scene, BehaviorState.LANE_CHANGE_LEFT).weighted_total(weights)
    keep_cost = cost.evaluate(scene, BehaviorState.KEEP_LANE).weighted_total(weights)
    assert change_cost < keep_cost

    verdict = gate.review(scene, BehaviorState.LANE_CHANGE_LEFT)
    assert not verdict.allowed
    assert verdict.reason is VetoReason.OCCUPIED

    decision = planner.decide(scene)
    assert decision.state is not BehaviorState.LANE_CHANGE_LEFT
    assert any(
        candidate.state is BehaviorState.LANE_CHANGE_LEFT and not candidate.admissible
        for candidate in decision.candidates
    )


def test_raising_the_progress_weight_cannot_buy_the_manoeuvre(
    road: Road, gate: GapAndDecelerationGate
) -> None:
    """A veto is not a large penalty, so no weight makes it affordable.

    This is the property that distinguishes a gate from a cost term. If safety
    were a term, there would always be a speed advantage large enough to pay
    for it.
    """
    ego = make_ego(road, lane=0, s=0.0, speed=18.0, desired_speed=31.0)
    blocker = make_vehicle(road, vehicle_id=1, lane=0, s=25.0, speed=15.0, desired_speed=15.0)
    beside = make_vehicle(road, vehicle_id=2, lane=1, s=-1.5, speed=26.0, desired_speed=31.0)

    for progress in (5.0, 50.0, 500.0, 5000.0):
        config = PlannerConfig(
            cost=CostConfig(
                weights=CostWeights(progress=progress, safety=0.0, comfort=0.0, lane_preference=0.0)
            )
        )
        planner = FiniteStateBehaviorPlanner(cost=WeightedCostModel(config.cost), gate=gate)
        scene = context(
            road,
            ego,
            blocker,
            beside,
            state=BehaviorState.PREPARE_LANE_CHANGE_LEFT,
            config=config,
        )
        assert planner.decide(scene).state is not BehaviorState.LANE_CHANGE_LEFT


def test_the_gate_reads_none_of_the_cost_configuration() -> None:
    """The gate's constructor takes limits and a driver model, and nothing else.

    Separation that is only a convention drifts. This checks it structurally.
    """
    fields = {field for field in GapAndDecelerationGate.__dataclass_fields__}
    assert fields == {"limits", "car_following"}


def test_a_verdict_must_agree_with_its_reason() -> None:
    """An allowed verdict carries no reason and a veto must name one."""
    from behavior_planner.model.decision import SafetyVerdict

    with pytest.raises(ValueError, match="no reason"):
        SafetyVerdict(allowed=True, reason=VetoReason.LEADER_GAP)
    with pytest.raises(ValueError, match="must name a reason"):
        SafetyVerdict(allowed=False, reason=VetoReason.NONE)
