"""Tier one. Trajectory generation for the chosen manoeuvre."""

from __future__ import annotations

from dataclasses import replace

import pytest

from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.trajectory import (
    TrajectoryGenerator,
    ballistic_step,
    binding_lanes,
    lane_change_for,
    longitudinal_acceleration,
)
from behavior_planner.model.config import PlannerConfig
from behavior_planner.model.decision import DecisionContext
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


@pytest.fixture
def generator(
    car_following: IntelligentDriverModel, config: PlannerConfig
) -> TrajectoryGenerator:
    """The default trajectory generator."""
    return TrajectoryGenerator(car_following=car_following, config=config)


def test_lane_keeping_binds_only_the_current_lane(road: Road) -> None:
    """Nothing in a neighbouring lane constrains a vehicle staying where it is."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    assert binding_lanes(road, ego, BehaviorState.KEEP_LANE) == (1,)


def test_preparing_and_changing_bind_both_lanes(road: Road) -> None:
    """The ego respects the leader it still has and the leader it is about to have."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    assert binding_lanes(road, ego, BehaviorState.PREPARE_LANE_CHANGE_LEFT) == (1, 2)
    assert binding_lanes(road, ego, BehaviorState.LANE_CHANGE_RIGHT) == (0, 1)


def test_the_strictest_binding_lane_wins(
    road: Road, car_following: IntelligentDriverModel
) -> None:
    """Preparing behind a slow vehicle in the target lane holds the ego back.

    This is how a prepare state does its job: the ego drops back until a gap
    opens rather than driving past the one it wants.
    """
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    ahead_left = make_vehicle(road, vehicle_id=1, lane=1, s=20.0, speed=12.0)
    snapshot = TrafficSnapshot(road=road, vehicles=(ego, ahead_left))
    keeping = longitudinal_acceleration(
        car_following=car_following,
        road=road,
        snapshot=snapshot,
        ego=ego,
        state=BehaviorState.KEEP_LANE,
    )
    preparing = longitudinal_acceleration(
        car_following=car_following,
        road=road,
        snapshot=snapshot,
        ego=ego,
        state=BehaviorState.PREPARE_LANE_CHANGE_LEFT,
    )
    assert preparing < keeping


def test_a_generated_trajectory_starts_at_the_current_pose(
    road: Road, generator: TrajectoryGenerator, config: PlannerConfig
) -> None:
    """The first sample is the present, not a prediction of it."""
    ego = make_ego(road, lane=1, s=25.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    points = generator.generate(scene, BehaviorState.KEEP_LANE, 0.1)
    assert points[0].s == pytest.approx(ego.s)
    assert points[0].d == pytest.approx(ego.d)
    assert points[0].speed == pytest.approx(ego.speed)


def test_a_trajectory_covers_the_configured_horizon(
    road: Road, generator: TrajectoryGenerator, config: PlannerConfig
) -> None:
    """The sample count follows from the horizon and the interval."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    points = generator.generate(scene, BehaviorState.KEEP_LANE, 0.1)
    assert len(points) == round(config.trajectory_horizon / 0.1) + 1
    assert points[-1].time == pytest.approx(config.trajectory_horizon)


def test_a_trajectory_never_reverses(
    road: Road, generator: TrajectoryGenerator, config: PlannerConfig
) -> None:
    """Braking hard against a stopped obstacle stops the ego, it does not reverse it."""
    ego = make_ego(road, lane=1, s=0.0, speed=30.0, desired_speed=31.0)
    wall = make_vehicle(road, vehicle_id=1, lane=1, s=8.0, speed=0.0, desired_speed=20.0)
    scene = context(road, ego, wall, state=BehaviorState.KEEP_LANE, config=config)
    points = generator.generate(scene, BehaviorState.KEEP_LANE, 0.1, horizon=20.0)
    assert all(point.speed >= 0.0 for point in points)
    assert points[-1].speed == pytest.approx(0.0, abs=1e-9)


def test_a_lane_change_trajectory_follows_the_lateral_profile(
    road: Road, generator: TrajectoryGenerator, config: PlannerConfig
) -> None:
    """The lateral samples are the closed-form profile, evaluated at the sample times."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    moving = replace(ego, lane_change=maneuver)
    scene = context(road, moving, state=BehaviorState.LANE_CHANGE_LEFT, config=config)
    points = generator.generate(
        scene, BehaviorState.LANE_CHANGE_LEFT, 0.1, horizon=config.lane_change_duration
    )
    for point in points:
        assert point.d == pytest.approx(maneuver.profile.offset(point.time))
    assert points[-1].d == pytest.approx(road.lane_center(1))


def test_no_manoeuvre_is_built_for_a_lane_off_the_road(
    road: Road, config: PlannerConfig
) -> None:
    """A change out of the rightmost lane produces nothing rather than an invalid profile."""
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    assert (
        lane_change_for(
            road, ego, BehaviorState.LANE_CHANGE_RIGHT, config.lane_change_duration
        )
        is None
    )


def test_the_lateral_manoeuvre_respects_comfort_limits(
    road: Road, config: PlannerConfig
) -> None:
    """The default duration keeps the lateral acceleration inside a comfortable band.

    Published comfort thresholds for lateral acceleration in normal driving sit
    near 2 m/s squared. The closed form makes this checkable without sampling.
    """
    ego = make_ego(road, lane=0, s=0.0, speed=28.0, desired_speed=31.0)
    maneuver = lane_change_for(
        road, ego, BehaviorState.LANE_CHANGE_LEFT, config.lane_change_duration
    )
    assert maneuver is not None
    assert maneuver.profile.peak_acceleration < 2.0
    assert maneuver.profile.peak_rate < 2.5


def test_a_ballistic_step_wraps_around_the_ring(road: Road) -> None:
    """Arc length stays inside the road, whatever the step size."""
    speed, s = ballistic_step(
        speed=30.0, s=road.length - 1.0, acceleration=0.0, dt=1.0, road=road
    )
    assert speed == pytest.approx(30.0)
    assert 0.0 <= s < road.length
    assert s == pytest.approx(29.0)


def test_a_non_positive_interval_is_rejected(
    road: Road, generator: TrajectoryGenerator, config: PlannerConfig
) -> None:
    """A trajectory with no time step is not a trajectory."""
    ego = make_ego(road, lane=1, s=0.0, speed=28.0, desired_speed=31.0)
    scene = context(road, ego, state=BehaviorState.KEEP_LANE, config=config)
    with pytest.raises(ValueError, match="dt must be positive"):
        generator.generate(scene, BehaviorState.KEEP_LANE, 0.0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        generator.generate(scene, BehaviorState.KEEP_LANE, 0.1, horizon=0.0)
