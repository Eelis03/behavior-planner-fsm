"""Tier one. Properties of the Intelligent Driver Model.

The equilibrium gap and the free flow speed are closed-form consequences of the
model equation, so they are checked against the formula rather than against a
recorded number.
"""

from __future__ import annotations

import math

import pytest

from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.trajectory import ballistic_step
from behavior_planner.model.config import DriverParams, IdmParams
from behavior_planner.model.road import Road


@pytest.fixture
def driver() -> DriverParams:
    """A driver at the published highway calibration."""
    return DriverParams(idm=IdmParams())


def test_free_flow_acceleration_vanishes_at_the_desired_speed(
    car_following: IntelligentDriverModel, driver: DriverParams
) -> None:
    """With no leader the only equilibrium is the desired speed."""
    v0 = driver.idm.desired_speed
    assert car_following.free_acceleration(speed=v0, driver=driver) == pytest.approx(0.0)
    assert car_following.free_acceleration(speed=0.5 * v0, driver=driver) > 0.0
    assert car_following.free_acceleration(speed=1.5 * v0, driver=driver) < 0.0


def test_a_vehicle_in_free_flow_converges_to_its_desired_speed(
    car_following: IntelligentDriverModel, driver: DriverParams, road: Road
) -> None:
    """Integrating with no leader reaches the desired speed from either side."""
    v0 = driver.idm.desired_speed
    for start in (5.0, v0 + 8.0):
        speed, s = start, 0.0
        for _ in range(6000):
            acceleration = car_following.acceleration(
                speed=speed, gap=math.inf, leader_speed=0.0, driver=driver
            )
            speed, s = ballistic_step(
                speed=speed, s=s, acceleration=acceleration, dt=0.1, road=road
            )
        assert speed == pytest.approx(v0, rel=1e-3)


def test_equilibrium_gap_matches_the_closed_form(
    car_following: IntelligentDriverModel, driver: DriverParams
) -> None:
    """The acceleration is zero at the gap the closed form predicts."""
    params = driver.idm
    for speed in (10.0, 20.0, 28.0):
        gap = car_following.equilibrium_gap(speed=speed, params=params)
        expected = (params.minimum_gap + speed * params.time_headway) / math.sqrt(
            1.0 - (speed / params.desired_speed) ** params.acceleration_exponent
        )
        assert gap == pytest.approx(expected)
        acceleration = car_following.acceleration(
            speed=speed, gap=gap, leader_speed=speed, driver=driver
        )
        assert acceleration == pytest.approx(0.0, abs=1e-12)


def test_equilibrium_gap_is_unbounded_at_and_above_the_desired_speed(
    car_following: IntelligentDriverModel, driver: DriverParams
) -> None:
    """No finite gap balances a driver already held back by the free flow term."""
    params = driver.idm
    assert math.isinf(car_following.equilibrium_gap(speed=params.desired_speed, params=params))
    assert math.isinf(
        car_following.equilibrium_gap(speed=params.desired_speed + 5.0, params=params)
    )


def test_following_a_steady_leader_reaches_the_equilibrium_spacing(
    car_following: IntelligentDriverModel, driver: DriverParams, road: Road
) -> None:
    """A follower settles at the closed-form gap behind a constant speed leader."""
    leader_speed = 22.0
    expected = car_following.equilibrium_gap(speed=leader_speed, params=driver.idm)
    speed, s = 15.0, 0.0
    leader_s = 200.0
    for _ in range(20000):
        gap = leader_s - s
        acceleration = car_following.acceleration(
            speed=speed, gap=gap, leader_speed=leader_speed, driver=driver
        )
        speed, s = ballistic_step(
            speed=speed, s=s, acceleration=acceleration, dt=0.05, road=road
        )
        leader_s = road.wrap(leader_s + leader_speed * 0.05)
        if s > leader_s:
            leader_s += road.length
    assert speed == pytest.approx(leader_speed, rel=1e-3)
    assert leader_s - s == pytest.approx(expected, rel=2e-3)


def test_the_model_never_produces_a_negative_speed(
    car_following: IntelligentDriverModel, driver: DriverParams, road: Road
) -> None:
    """A stopped leader at close range brings the follower to rest, not into reverse.

    The integrator, not the model equation, is what guarantees this: the
    acceleration itself is unbounded below as the gap closes.
    """
    speed, s = 30.0, 0.0
    leader_s = 12.0
    for _ in range(4000):
        acceleration = car_following.acceleration(
            speed=speed, gap=leader_s - s, leader_speed=0.0, driver=driver
        )
        speed, s = ballistic_step(
            speed=speed, s=s, acceleration=acceleration, dt=0.1, road=road
        )
        assert speed >= 0.0
    assert speed == pytest.approx(0.0, abs=1e-9)


def test_acceleration_rises_with_the_gap(
    car_following: IntelligentDriverModel, driver: DriverParams
) -> None:
    """More room is never worse, which is what makes the interaction term a repulsion."""
    previous = -math.inf
    for gap in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        acceleration = car_following.acceleration(
            speed=25.0, gap=gap, leader_speed=25.0, driver=driver
        )
        assert acceleration > previous
        previous = acceleration
    assert previous < car_following.acceleration(
        speed=25.0, gap=math.inf, leader_speed=25.0, driver=driver
    )


def test_closing_on_a_leader_is_penalised_more_than_matching_it(
    car_following: IntelligentDriverModel, driver: DriverParams
) -> None:
    """The closing rate term brakes earlier than the distance term alone would."""
    matched = car_following.acceleration(
        speed=25.0, gap=40.0, leader_speed=25.0, driver=driver
    )
    closing = car_following.acceleration(
        speed=25.0, gap=40.0, leader_speed=15.0, driver=driver
    )
    opening = car_following.acceleration(
        speed=25.0, gap=40.0, leader_speed=32.0, driver=driver
    )
    assert closing < matched < opening


def test_desired_gap_is_never_below_the_jam_distance(
    car_following: IntelligentDriverModel, driver: DriverParams
) -> None:
    """The dynamic term is clamped at zero, so ``s_star`` never falls under ``s0``."""
    params = driver.idm
    for leader_speed in (0.0, 25.0, 60.0):
        desired = car_following.desired_gap(
            speed=25.0, leader_speed=leader_speed, params=params
        )
        assert desired >= params.minimum_gap


@pytest.mark.parametrize(
    "field",
    ["desired_speed", "minimum_gap", "maximum_acceleration", "comfortable_deceleration"],
)
def test_non_positive_parameters_are_rejected(field: str) -> None:
    """A parameter the model is not defined on raises at construction."""
    with pytest.raises(ValueError, match=field):
        IdmParams(**{field: 0.0})
