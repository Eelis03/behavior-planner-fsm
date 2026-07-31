"""Tier one. The MOBIL lane change model that drives the traffic."""

from __future__ import annotations

from dataclasses import replace

import pytest

from behavior_planner.algorithm.mobil import MobilLaneChangeModel
from behavior_planner.model.config import DriverParams, IdmParams, MobilParams
from behavior_planner.model.lateral import LateralProfile
from behavior_planner.model.road import Road
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import LaneChange, Vehicle
from tests.conftest import make_vehicle


def snapshot(road: Road, *vehicles: Vehicle) -> TrafficSnapshot:
    """A snapshot holding exactly the given vehicles."""
    return TrafficSnapshot(road=road, vehicles=vehicles)


def test_a_blocked_vehicle_moves_into_a_clear_lane(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """A driver held back by a slow leader takes an empty neighbouring lane."""
    subject = make_vehicle(road, vehicle_id=1, lane=0, s=0.0, speed=30.0, desired_speed=32.0)
    blocker = make_vehicle(road, vehicle_id=2, lane=0, s=25.0, speed=18.0, desired_speed=18.0)
    assert lane_change_model.choose_lane(subject, snapshot(road, subject, blocker)) == 1


def test_an_unobstructed_vehicle_stays_put(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """With nothing in the way there is no incentive above the threshold."""
    subject = make_vehicle(road, vehicle_id=1, lane=1, s=0.0, speed=30.0, desired_speed=30.0)
    assert lane_change_model.choose_lane(subject, snapshot(road, subject)) is None


def test_the_safety_criterion_refuses_a_change_that_would_slam_the_new_follower(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """A large incentive does not buy a merge in front of a close, fast follower.

    The subject is given zero politeness so that the incentive criterion cannot
    be what stops it. The only thing left to refuse the change is the safety
    criterion, which is what this test is about.
    """
    subject = make_vehicle(road, vehicle_id=1, lane=0, s=0.0, speed=18.0, desired_speed=32.0)
    subject = replace(
        subject,
        driver=DriverParams(idm=subject.driver.idm, mobil=MobilParams(politeness=0.0)),
    )
    blocker = make_vehicle(road, vehicle_id=2, lane=0, s=25.0, speed=15.0, desired_speed=15.0)
    tailgater = make_vehicle(road, vehicle_id=3, lane=1, s=-8.0, speed=32.0, desired_speed=32.0)
    scene = snapshot(road, subject, blocker, tailgater)

    assessment = lane_change_model.assess(subject, scene, 1)
    assert assessment.incentive > assessment.threshold
    assert not assessment.safe
    assert not assessment.accepted
    assert lane_change_model.choose_lane(subject, scene) is None


def test_the_safety_criterion_loosens_as_the_follower_falls_back(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """The same merge becomes safe once the follower is far enough behind."""
    subject = make_vehicle(road, vehicle_id=1, lane=0, s=0.0, speed=18.0, desired_speed=32.0)
    blocker = make_vehicle(road, vehicle_id=2, lane=0, s=25.0, speed=15.0, desired_speed=15.0)
    for behind, expected in ((-8.0, False), (-200.0, True)):
        follower = make_vehicle(
            road, vehicle_id=3, lane=1, s=behind, speed=32.0, desired_speed=32.0
        )
        scene = snapshot(road, subject, blocker, follower)
        assert lane_change_model.assess(subject, scene, 1).safe is expected


def test_politeness_reduces_the_incentive_to_cut_in(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """Weighing the cost imposed on others lowers the incentive to impose it."""
    subject = make_vehicle(road, vehicle_id=1, lane=0, s=0.0, speed=22.0, desired_speed=32.0)
    blocker = make_vehicle(road, vehicle_id=2, lane=0, s=40.0, speed=20.0, desired_speed=20.0)
    follower = make_vehicle(road, vehicle_id=3, lane=1, s=-45.0, speed=30.0, desired_speed=30.0)

    incentives = []
    for politeness in (0.0, 0.5, 1.0):
        driver = DriverParams(idm=subject.driver.idm, mobil=MobilParams(politeness=politeness))
        candidate = replace(subject, driver=driver)
        scene = snapshot(road, candidate, blocker, follower)
        incentives.append(lane_change_model.assess(candidate, scene, 1).incentive)
    assert incentives[0] > incentives[1] > incentives[2]


def test_the_right_bias_makes_a_change_to_the_right_cheaper(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """The asymmetric variant lowers the threshold one way and raises it the other."""
    biased = MobilParams(right_bias=0.5)
    subject = make_vehicle(road, vehicle_id=1, lane=1, s=0.0, speed=28.0, desired_speed=30.0)
    subject = replace(
        subject, driver=DriverParams(idm=subject.driver.idm, mobil=biased)
    )
    scene = snapshot(road, subject)
    to_right = lane_change_model.assess(subject, scene, 0)
    to_left = lane_change_model.assess(subject, scene, 2)
    assert to_right.threshold < to_left.threshold
    assert to_left.threshold - to_right.threshold == pytest.approx(2.0 * biased.right_bias)


def test_a_vehicle_already_changing_lane_is_not_given_a_second_manoeuvre(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """One manoeuvre at a time, whatever the incentive says."""
    subject = make_vehicle(road, vehicle_id=1, lane=0, s=0.0, speed=30.0, desired_speed=32.0)
    blocker = make_vehicle(road, vehicle_id=2, lane=0, s=25.0, speed=18.0, desired_speed=18.0)
    running = replace(
        subject,
        lane_change=LaneChange(
            source_lane=0,
            target_lane=1,
            profile=LateralProfile(start=0.0, target=road.lane_center(1), duration=3.0),
        ),
    )
    assert lane_change_model.choose_lane(running, snapshot(road, running, blocker)) is None


def test_an_edge_lane_offers_only_one_candidate(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """A vehicle in the leftmost lane is never offered a lane that does not exist."""
    subject = make_vehicle(road, vehicle_id=1, lane=2, s=0.0, speed=30.0, desired_speed=32.0)
    blocker = make_vehicle(road, vehicle_id=2, lane=2, s=25.0, speed=18.0, desired_speed=18.0)
    assert lane_change_model.choose_lane(subject, snapshot(road, subject, blocker)) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("politeness", 1.5), ("changing_threshold", -1.0), ("safe_deceleration", 0.0)],
)
def test_invalid_parameters_are_rejected(field: str, value: float) -> None:
    """A parameter outside the model's range raises at construction."""
    with pytest.raises(ValueError, match=field):
        MobilParams(**{field: value})


def test_a_polite_leader_pulls_aside_for_a_faster_follower(
    road: Road, lane_change_model: MobilLaneChangeModel
) -> None:
    """The politeness term reproduces courtesy behaviour, which the scenarios rely on.

    The scripted scenarios disable this with an unreachable switching threshold,
    because a leader that yields is not an obstruction.
    """
    slow = Vehicle(
        vehicle_id=1,
        s=0.0,
        d=road.lane_center(0),
        speed=20.0,
        driver=DriverParams(idm=IdmParams(desired_speed=20.0), mobil=MobilParams(politeness=0.5)),
    )
    fast = make_vehicle(road, vehicle_id=2, lane=0, s=-30.0, speed=30.0, desired_speed=32.0)
    assert lane_change_model.choose_lane(slow, snapshot(road, slow, fast)) == 1
