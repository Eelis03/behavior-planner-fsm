"""Per-lane neighbour queries over a set of vehicles at one instant."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from behavior_planner.model.road import Road
from behavior_planner.model.vehicle import Vehicle

__all__ = [
    "LaneNeighbors",
    "Neighbor",
    "TrafficSnapshot",
    "time_headway",
    "time_to_collision",
]


def time_to_collision(gap: float, closing_speed: float) -> float:
    """Time until two bodies meet at the present closing rate, in seconds.

    ``math.inf`` when they are not closing. A non-positive gap, meaning the
    bodies already touch, returns zero rather than a negative time.
    """
    if closing_speed <= 0.0:
        return math.inf
    if gap <= 0.0:
        return 0.0
    return gap / closing_speed


def time_headway(gap: float, speed: float) -> float:
    """Time to cover ``gap`` at ``speed``, in seconds.

    ``math.inf`` at a standstill, which is correct: a stopped vehicle is not
    about to close any gap. A non-positive gap returns zero.
    """
    if speed <= 0.0:
        return math.inf
    if gap <= 0.0:
        return 0.0
    return gap / speed


@dataclass(frozen=True, slots=True)
class Neighbor:
    """A vehicle found by a neighbour query, with the gap that separates it.

    :attr:`gap` is bumper to bumper along the reference line and is negative
    only when the two bodies overlap longitudinally.
    """

    vehicle: Vehicle
    gap: float

    @property
    def speed(self) -> float:
        """Longitudinal speed of the neighbour."""
        return self.vehicle.speed


@dataclass(frozen=True, slots=True)
class LaneNeighbors:
    """The vehicle immediately ahead and the vehicle immediately behind."""

    leader: Neighbor | None = None
    follower: Neighbor | None = None


@dataclass(frozen=True, slots=True)
class TrafficSnapshot:
    """Every vehicle at one instant, indexed by the lanes each one occupies.

    A vehicle part way through a lane change occupies two lanes and is returned
    by queries against both, so traffic in the lane being entered reacts to it
    from the moment it starts crossing rather than when it arrives.
    """

    road: Road
    vehicles: tuple[Vehicle, ...]
    _by_lane: dict[int, tuple[Vehicle, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the per-lane index, each lane's list sorted by arc length."""
        buckets: dict[int, list[Vehicle]] = {lane: [] for lane in self.road.lanes}
        for vehicle in self.vehicles:
            for lane in vehicle.occupied_lanes(self.road):
                buckets[lane].append(vehicle)
        index = {
            lane: tuple(sorted(members, key=lambda v: v.s)) for lane, members in buckets.items()
        }
        object.__setattr__(self, "_by_lane", index)

    def in_lane(self, lane: int) -> tuple[Vehicle, ...]:
        """Every vehicle whose body overlaps ``lane``, ordered by arc length."""
        if not self.road.contains_lane(lane):
            raise ValueError(f"lane {lane} is not on this road")
        return self._by_lane[lane]

    def by_id(self, vehicle_id: int) -> Vehicle:
        """The vehicle with ``vehicle_id``."""
        for vehicle in self.vehicles:
            if vehicle.vehicle_id == vehicle_id:
                return vehicle
        raise KeyError(f"no vehicle with id {vehicle_id}")

    def neighbors(self, subject: Vehicle, lane: int) -> LaneNeighbors:
        """Closest vehicle ahead of and behind ``subject`` in ``lane``.

        ``subject`` is excluded by identifier, so the query is well defined when
        the subject itself occupies ``lane``.
        """
        leader: Neighbor | None = None
        follower: Neighbor | None = None
        best_ahead = self.road.length
        best_behind = self.road.length
        for other in self.in_lane(lane):
            if other.vehicle_id == subject.vehicle_id:
                continue
            ahead = self.road.forward_distance(subject.s, other.s)
            behind = self.road.length - ahead
            half_bodies = 0.5 * (subject.shape.length + other.shape.length)
            if ahead <= behind:
                if ahead < best_ahead:
                    best_ahead = ahead
                    leader = Neighbor(vehicle=other, gap=ahead - half_bodies)
            elif behind < best_behind:
                best_behind = behind
                follower = Neighbor(vehicle=other, gap=behind - half_bodies)
        return LaneNeighbors(leader=leader, follower=follower)

    def colliding_pairs(self) -> tuple[tuple[int, int], ...]:
        """Every pair of overlapping vehicle identifiers, each pair reported once.

        Two bodies can overlap only if they share a lane, and within a lane only
        if they are adjacent in the arc-length ordering, so testing consecutive
        pairs per lane finds every overlap without the quadratic sweep.
        """
        found: set[tuple[int, int]] = set()
        for lane in self.road.lanes:
            members = self.in_lane(lane)
            count = len(members)
            if count < 2:
                continue
            for index in range(count):
                first = members[index]
                second = members[(index + 1) % count]
                if first.vehicle_id == second.vehicle_id:
                    continue
                if self.overlaps(first, second):
                    low = min(first.vehicle_id, second.vehicle_id)
                    high = max(first.vehicle_id, second.vehicle_id)
                    found.add((low, high))
        return tuple(sorted(found))

    def overlaps(self, first: Vehicle, second: Vehicle) -> bool:
        """True when the two bounding boxes intersect in the Frenet frame."""
        along = abs(self.road.separation(first.s, second.s))
        across = abs(first.d - second.d)
        return along < 0.5 * (first.shape.length + second.shape.length) and across < 0.5 * (
            first.shape.width + second.shape.width
        )
