"""Vehicle geometry, kinematic state and in-progress lane changes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from behavior_planner.model.config import DriverParams
from behavior_planner.model.lateral import LateralProfile
from behavior_planner.model.road import Road

__all__ = ["EGO_ID", "LaneChange", "Vehicle", "VehicleShape"]

EGO_ID: int = 0
"""Identifier of the vehicle driven by the behaviour planner."""


@dataclass(frozen=True, slots=True)
class VehicleShape:
    """Bounding box of a vehicle in the Frenet frame."""

    length: float = 5.0
    width: float = 1.8

    def __post_init__(self) -> None:
        """Reject a degenerate body."""
        if self.length <= 0.0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.width <= 0.0:
            raise ValueError(f"width must be positive, got {self.width}")


@dataclass(frozen=True, slots=True)
class LaneChange:
    """A lateral transition that has started and has not yet finished."""

    source_lane: int
    target_lane: int
    profile: LateralProfile
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        """Reject a manoeuvre that does not change lane."""
        if self.source_lane == self.target_lane:
            raise ValueError(f"a lane change must change lane, got lane {self.source_lane} twice")
        if self.elapsed < 0.0:
            raise ValueError(f"elapsed must not be negative, got {self.elapsed}")

    @property
    def progress(self) -> float:
        """Fraction of the transition duration already spent, in ``[0, 1]``."""
        return min(self.elapsed / self.profile.duration, 1.0)

    @property
    def is_complete(self) -> bool:
        """True once the lateral transition has run its full duration."""
        return self.elapsed >= self.profile.duration

    def offset(self) -> float:
        """Current lateral offset of the manoeuvre."""
        return self.profile.offset(self.elapsed)

    def rate(self) -> float:
        """Current lateral velocity of the manoeuvre."""
        return self.profile.rate(self.elapsed)

    def advanced(self, dt: float) -> LaneChange:
        """Return the manoeuvre ``dt`` seconds later."""
        return replace(self, elapsed=self.elapsed + dt)


@dataclass(frozen=True, slots=True)
class Vehicle:
    """One traffic participant at one instant.

    The lateral offset :attr:`d` is redundant with :attr:`lane_change` while a
    manoeuvre runs and is kept in step by the simulator, which recomputes it
    from the manoeuvre's closed-form profile rather than integrating it.
    """

    vehicle_id: int
    s: float
    d: float
    speed: float
    shape: VehicleShape = field(default_factory=VehicleShape)
    driver: DriverParams = field(default_factory=DriverParams)
    lane_change: LaneChange | None = None

    def __post_init__(self) -> None:
        """Reject a state the dynamics can never produce."""
        if self.speed < 0.0:
            raise ValueError(f"speed must not be negative, got {self.speed}")

    @property
    def is_changing_lane(self) -> bool:
        """True while a lateral transition is in progress."""
        return self.lane_change is not None

    @property
    def lateral_speed(self) -> float:
        """Lateral velocity, zero unless a lane change is running."""
        return 0.0 if self.lane_change is None else self.lane_change.rate()

    def assigned_lane(self, road: Road) -> int:
        """Lane this vehicle counts as belonging to for decision purposes.

        A vehicle part way through a lane change is treated as belonging to the
        lane it is entering, because that is the lane whose traffic it must now
        keep pace with.
        """
        if self.lane_change is not None:
            return self.lane_change.target_lane
        return road.nearest_lane(self.d)

    def occupied_lanes(self, road: Road) -> tuple[int, ...]:
        """Every lane whose band this vehicle's body overlaps."""
        return road.occupied_lanes(self.d, self.shape.width)

    def moved(self, *, s: float, d: float, speed: float) -> Vehicle:
        """Return a copy at a new pose, with the manoeuvre untouched."""
        return replace(self, s=s, d=d, speed=max(speed, 0.0))
