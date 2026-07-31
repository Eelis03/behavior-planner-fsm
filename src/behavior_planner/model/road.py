"""Straight multi-lane highway expressed in a Frenet style coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Road"]


@dataclass(frozen=True, slots=True)
class Road:
    """A ring shaped highway with ``lane_count`` parallel lanes of equal width.

    A position is the pair ``(s, d)``. ``s`` is arc length along the reference
    line and is measured modulo :attr:`length`, so the road closes on itself and
    the vehicle density stays constant for the whole run. ``d`` is the lateral
    offset from the reference line, positive towards the left. Lane 0 is the
    rightmost lane and its centre lies on ``d = 0``, so the centre of lane ``k``
    is ``k * lane_width`` and a lane change to the left raises the lane index.

    The reference line is straight, so the Frenet frame coincides with a
    Cartesian frame and no curvature correction is applied. The frame is kept
    because every quantity the planner reasons about, gap, headway, lateral
    offset, is naturally expressed in it, and because a curved reference line
    can be substituted without touching the layers above.
    """

    lane_count: int
    lane_width: float = 3.7
    length: float = 1200.0
    speed_limit: float = 33.3

    def __post_init__(self) -> None:
        """Reject a road that cannot be simulated."""
        if self.lane_count < 1:
            raise ValueError(f"lane_count must be at least 1, got {self.lane_count}")
        if self.lane_width <= 0.0:
            raise ValueError(f"lane_width must be positive, got {self.lane_width}")
        if self.length <= 0.0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.speed_limit <= 0.0:
            raise ValueError(f"speed_limit must be positive, got {self.speed_limit}")

    @property
    def lanes(self) -> tuple[int, ...]:
        """Every valid lane index, rightmost first."""
        return tuple(range(self.lane_count))

    def contains_lane(self, lane: int) -> bool:
        """True when ``lane`` is a lane of this road."""
        return 0 <= lane < self.lane_count

    def lane_center(self, lane: int) -> float:
        """Lateral offset of the centre line of ``lane``."""
        if not self.contains_lane(lane):
            raise ValueError(f"lane {lane} is not on a road with {self.lane_count} lanes")
        return lane * self.lane_width

    def nearest_lane(self, d: float) -> int:
        """Index of the lane whose centre is closest to lateral offset ``d``."""
        raw = round(d / self.lane_width)
        return min(max(raw, 0), self.lane_count - 1)

    def wrap(self, s: float) -> float:
        """Map ``s`` into ``[0, length)``."""
        return s % self.length

    def forward_distance(self, origin: float, target: float) -> float:
        """Distance travelled going forward from ``origin`` to ``target``.

        The result lies in ``[0, length)``. It is the quantity a follower needs:
        how far ahead something is, never how far behind.
        """
        return (target - origin) % self.length

    def separation(self, origin: float, target: float) -> float:
        """Signed longitudinal separation of ``target`` from ``origin``.

        The result lies in ``[-length / 2, length / 2)`` and is positive when
        ``target`` is ahead. Use this where the sign matters, for example
        collision testing, and :meth:`forward_distance` where it does not.
        """
        half = 0.5 * self.length
        return (target - origin + half) % self.length - half

    def occupied_lanes(self, d: float, width: float) -> tuple[int, ...]:
        """Every lane whose band overlaps a body of ``width`` centred on ``d``.

        A vehicle part way through a lane change occupies two lanes, and both
        must see it. Lane ``k`` spans ``[k * w - w / 2, k * w + w / 2]``.
        """
        if width <= 0.0:
            raise ValueError(f"width must be positive, got {width}")
        half_body = 0.5 * width
        half_lane = 0.5 * self.lane_width
        lower = d - half_body
        upper = d + half_body
        occupied = [
            lane
            for lane in self.lanes
            for centre in (self.lane_center(lane),)
            if lower < centre + half_lane and upper > centre - half_lane
        ]
        if occupied:
            return tuple(occupied)
        # The body lies entirely outside the carriageway, which the simulator
        # never produces; fall back to the closest lane so callers see one lane.
        return (self.nearest_lane(d),)
