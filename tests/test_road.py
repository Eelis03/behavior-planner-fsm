"""Tier one. The Frenet frame: wrapping, separation and lane occupancy."""

from __future__ import annotations

import pytest

from behavior_planner.model.road import Road


def test_lane_centres_are_evenly_spaced(road: Road) -> None:
    """Lane 0 sits on the reference line and each lane is one width to its left."""
    assert road.lane_center(0) == pytest.approx(0.0)
    assert road.lane_center(1) == pytest.approx(road.lane_width)
    assert road.lane_center(2) == pytest.approx(2.0 * road.lane_width)


def test_an_absent_lane_is_rejected(road: Road) -> None:
    """Asking for the centre of a lane the road does not have raises."""
    assert not road.contains_lane(-1)
    assert not road.contains_lane(road.lane_count)
    with pytest.raises(ValueError, match="not on a road"):
        road.lane_center(road.lane_count)


def test_nearest_lane_rounds_and_clamps(road: Road) -> None:
    """Offsets between centres round to the closer lane, and beyond the edge clamp."""
    assert road.nearest_lane(0.0) == 0
    assert road.nearest_lane(0.4 * road.lane_width) == 0
    assert road.nearest_lane(0.6 * road.lane_width) == 1
    assert road.nearest_lane(-50.0) == 0
    assert road.nearest_lane(50.0) == road.lane_count - 1


def test_wrapping_closes_the_road(road: Road) -> None:
    """Arc length is taken modulo the circumference in both directions."""
    assert road.wrap(0.0) == pytest.approx(0.0)
    assert road.wrap(road.length) == pytest.approx(0.0)
    assert road.wrap(road.length + 30.0) == pytest.approx(30.0)
    assert road.wrap(-30.0) == pytest.approx(road.length - 30.0)


def test_forward_distance_is_never_negative(road: Road) -> None:
    """Going forward from a point always reaches every other point."""
    assert road.forward_distance(100.0, 250.0) == pytest.approx(150.0)
    assert road.forward_distance(250.0, 100.0) == pytest.approx(road.length - 150.0)
    for origin in (0.0, 300.0, road.length - 1.0):
        for target in (0.0, 25.0, 600.0, road.length - 5.0):
            distance = road.forward_distance(origin, target)
            assert 0.0 <= distance < road.length


def test_separation_is_signed_and_bounded(road: Road) -> None:
    """Separation is the shorter way round and carries a sign."""
    half = 0.5 * road.length
    assert road.separation(100.0, 250.0) == pytest.approx(150.0)
    assert road.separation(250.0, 100.0) == pytest.approx(-150.0)
    for origin in (0.0, 700.0):
        for target in (0.0, 400.0, road.length - 2.0):
            assert -half <= road.separation(origin, target) < half


def test_a_centred_vehicle_occupies_exactly_one_lane(road: Road) -> None:
    """A body narrower than a lane and centred in it overlaps nothing else."""
    for lane in road.lanes:
        assert road.occupied_lanes(road.lane_center(lane), 1.8) == (lane,)


def test_a_vehicle_between_lanes_occupies_both(road: Road) -> None:
    """Halfway across a boundary, the body is in the lane it left and the one it enters."""
    boundary = 0.5 * road.lane_width
    assert road.occupied_lanes(boundary, 1.8) == (0, 1)


def test_a_wide_body_occupies_three_lanes(road: Road) -> None:
    """Occupancy follows from the geometry, not from a lane index."""
    assert road.occupied_lanes(road.lane_center(1), 2.5 * road.lane_width) == (0, 1, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [("lane_count", 0), ("lane_width", 0.0), ("length", -1.0), ("speed_limit", 0.0)],
)
def test_a_degenerate_road_is_rejected(field: str, value: float) -> None:
    """A road the simulator cannot use raises at construction."""
    kwargs: dict[str, float] = {"lane_count": 3}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        Road(**kwargs)  # type: ignore[arg-type]
