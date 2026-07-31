"""Tier one. The weighted cost function."""

from __future__ import annotations

import pytest

from behavior_planner.algorithm.cost import WeightedCostModel, target_lane_for
from behavior_planner.model.config import CostConfig, CostWeights, PlannerConfig
from behavior_planner.model.decision import CostTerms, DecisionContext
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle
from tests.conftest import make_ego, make_vehicle

ALL_STATES = list(BehaviorState)


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


@pytest.mark.parametrize("successor", ALL_STATES)
def test_every_term_is_normalised(
    road: Road, cost: WeightedCostModel, config: PlannerConfig, successor: BehaviorState
) -> None:
    """Each term stays inside ``[0, 1]``, which is what makes the weights comparable."""
    ego = make_ego(road, lane=1, s=0.0, speed=12.0, desired_speed=31.0)
    crowd = [
        make_vehicle(road, vehicle_id=index + 1, lane=lane, s=offset, speed=10.0)
        for index, (lane, offset) in enumerate(
            [(0, 6.0), (0, -6.0), (1, 7.0), (1, -7.0), (2, 6.5), (2, -6.5)]
        )
    ]
    scene = context(road, ego, *crowd, state=BehaviorState.KEEP_LANE, config=config)
    terms = cost.evaluate(scene, successor)
    for name, value in terms.as_dict().items():
        assert 0.0 <= value <= 1.0, name


def test_an_empty_lane_costs_nothing_on_progress(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """With no leader the lane delivers the free flow speed, so the shortfall is zero."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    assert cost.evaluate(scene, BehaviorState.KEEP_LANE).progress == pytest.approx(0.0)


def test_a_slow_leader_raises_the_progress_cost_as_it_approaches(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """The penalty grows linearly as the leader comes inside the horizon."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    previous = -1.0
    for gap in (200.0, 100.0, 60.0, 20.0):
        leader = make_vehicle(road, vehicle_id=1, lane=1, s=gap, speed=18.0)
        scene = context(road, ego, leader, state=BehaviorState.KEEP_LANE, config=config)
        value = cost.evaluate(scene, BehaviorState.KEEP_LANE).progress
        assert value > previous
        previous = value


def test_a_leader_beyond_the_horizon_is_ignored(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """Past the progress horizon a slow vehicle stops influencing the choice."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    beyond = config.cost.progress_horizon + 50.0
    leader = make_vehicle(road, vehicle_id=1, lane=1, s=beyond, speed=5.0)
    scene = context(road, ego, leader, state=BehaviorState.KEEP_LANE, config=config)
    assert cost.evaluate(scene, BehaviorState.KEEP_LANE).progress == pytest.approx(0.0)


def test_a_prepare_state_is_scored_on_both_lanes(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """Preparing collects the mean of the lane being left and the lane being sought.

    This is what stops the machine from parking in a prepare state: preparing
    never scores better on progress than actually arriving.
    """
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    leader = make_vehicle(road, vehicle_id=1, lane=0, s=30.0, speed=18.0)
    scene = context(road, ego, leader, state=BehaviorState.KEEP_LANE, config=config)
    keeping = cost.evaluate(scene, BehaviorState.KEEP_LANE).progress
    preparing = cost.evaluate(scene, BehaviorState.PREPARE_LANE_CHANGE_LEFT).progress
    changing = cost.evaluate(scene, BehaviorState.LANE_CHANGE_LEFT).progress
    assert changing < preparing < keeping
    assert preparing == pytest.approx(0.5 * (keeping + changing))


def test_lingering_in_a_prepare_state_costs_more_than_finishing(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """The comfort term charges preparing more than changing, on purpose."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.PREPARE_LANE_CHANGE_LEFT, config=config)
    preparing = cost.evaluate(scene, BehaviorState.PREPARE_LANE_CHANGE_LEFT).comfort
    changing = cost.evaluate(scene, BehaviorState.LANE_CHANGE_LEFT).comfort
    keeping = cost.evaluate(scene, BehaviorState.KEEP_LANE).comfort
    assert keeping == pytest.approx(0.0)
    assert changing < preparing


def test_lane_preference_measures_distance_from_the_preferred_lane(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """Lane 0 is free, and the cost rises with the lane index on a three lane road."""
    for lane, expected in ((0, 0.0), (1, 0.5), (2, 1.0)):
        ego = make_ego(road, lane=lane, s=0.0, speed=28.0, desired_speed=31.0)
        scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
        terms = cost.evaluate(scene, BehaviorState.KEEP_LANE)
        assert terms.lane_preference == pytest.approx(expected)


def test_a_close_leader_raises_the_soft_safety_term(
    road: Road, cost: WeightedCostModel, config: PlannerConfig
) -> None:
    """The soft term measures the shortfall against the cost function's own gaps."""
    ego = make_ego(road, lane=1, s=0.0, speed=25.0, desired_speed=31.0)
    roomy = make_vehicle(road, vehicle_id=1, lane=1, s=120.0, speed=25.0)
    tight = make_vehicle(road, vehicle_id=1, lane=1, s=12.0, speed=25.0)
    for vehicle, expected_positive in ((roomy, False), (tight, True)):
        scene = context(road, ego, vehicle, state=BehaviorState.KEEP_LANE, config=config)
        value = cost.evaluate(scene, BehaviorState.KEEP_LANE).safety
        assert (value > 0.0) is expected_positive


def test_the_weighted_total_is_the_declared_linear_combination() -> None:
    """No hidden terms: the total is exactly the four weighted values."""
    weights = CostWeights(progress=5.0, safety=1.5, comfort=0.10, lane_preference=0.35)
    terms = CostTerms(progress=0.2, safety=0.3, comfort=0.4, lane_preference=0.5)
    expected = 5.0 * 0.2 + 1.5 * 0.3 + 0.10 * 0.4 + 0.35 * 0.5
    assert terms.weighted_total(weights) == pytest.approx(expected)


def test_target_lane_is_clamped_to_the_road(
    road: Road, config: PlannerConfig
) -> None:
    """Scoring a successor never raises, even where the successor is infeasible."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    assert target_lane_for(scene, BehaviorState.LANE_CHANGE_RIGHT) == 0
    assert target_lane_for(scene, BehaviorState.LANE_CHANGE_LEFT) == 1


def test_negative_or_empty_weights_are_rejected() -> None:
    """A weight vector that cannot rank anything raises at construction."""
    with pytest.raises(ValueError, match="must not be negative"):
        CostWeights(progress=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        CostWeights(progress=0.0, safety=0.0, comfort=0.0, lane_preference=0.0)


def test_an_invalid_cost_configuration_is_rejected() -> None:
    """Every scale the cost function divides by must be positive."""
    with pytest.raises(ValueError, match="progress_horizon"):
        CostConfig(progress_horizon=0.0)
    with pytest.raises(ValueError, match="desired_standstill_gap"):
        CostConfig(desired_standstill_gap=0.0)
