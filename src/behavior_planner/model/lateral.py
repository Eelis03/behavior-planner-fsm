"""Closed-form lateral offset profile used by every lane change in the model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = ["LateralProfile", "quintic_acceleration", "quintic_progress", "quintic_rate"]

# Peak values of the normalised derivatives of the quintic, computed once in
# closed form rather than by sampling. See :class:`LateralProfile`.
_PEAK_RATE: Final[float] = 1.875
_PEAK_ACCELERATION: Final[float] = 10.0 / math.sqrt(3.0)
_PEAK_JERK: Final[float] = 60.0


def quintic_progress(tau: float) -> float:
    """Fraction of the lateral displacement completed at normalised time ``tau``.

    The polynomial ``10 t^3 - 15 t^4 + 6 t^5`` is the unique quintic satisfying
    zero rate and zero acceleration at both ends, which is the minimum jerk
    transition between two lateral offsets.
    """
    t = min(max(tau, 0.0), 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def quintic_rate(tau: float) -> float:
    """First derivative of :func:`quintic_progress` with respect to ``tau``."""
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    return tau * tau * (30.0 + tau * (-60.0 + 30.0 * tau))


def quintic_acceleration(tau: float) -> float:
    """Second derivative of :func:`quintic_progress` with respect to ``tau``."""
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    return tau * (60.0 + tau * (-180.0 + 120.0 * tau))


@dataclass(frozen=True, slots=True)
class LateralProfile:
    """A minimum jerk transition from lateral offset ``start`` to ``target``.

    The profile is a pure function of elapsed time, so a vehicle's lateral state
    is recomputed from its manoeuvre rather than integrated. Two vehicles given
    the same manoeuvre therefore follow bit-identical lateral paths, which is
    what makes a seeded run reproducible.

    The peak derivatives are available in closed form, ``1.875 * D / T`` for the
    rate, ``(10 / sqrt(3)) * D / T^2`` for the acceleration and ``60 * D / T^3``
    for the jerk, where ``D`` is the displacement and ``T`` the duration. The
    comfort limits of a manoeuvre can therefore be checked without sampling it.
    """

    start: float
    target: float
    duration: float

    def __post_init__(self) -> None:
        """Reject a profile with a non-positive duration."""
        if self.duration <= 0.0:
            raise ValueError(f"duration must be positive, got {self.duration}")

    @property
    def displacement(self) -> float:
        """Signed lateral distance covered over the whole profile."""
        return self.target - self.start

    def offset(self, elapsed: float) -> float:
        """Lateral offset ``elapsed`` seconds after the manoeuvre started."""
        return self.start + self.displacement * quintic_progress(elapsed / self.duration)

    def rate(self, elapsed: float) -> float:
        """Lateral velocity ``elapsed`` seconds after the manoeuvre started."""
        return self.displacement * quintic_rate(elapsed / self.duration) / self.duration

    def acceleration(self, elapsed: float) -> float:
        """Lateral acceleration ``elapsed`` seconds after the manoeuvre started."""
        scale = self.duration * self.duration
        return self.displacement * quintic_acceleration(elapsed / self.duration) / scale

    @property
    def peak_rate(self) -> float:
        """Largest lateral speed reached, in metres per second."""
        return _PEAK_RATE * abs(self.displacement) / self.duration

    @property
    def peak_acceleration(self) -> float:
        """Largest lateral acceleration reached, in metres per second squared."""
        return _PEAK_ACCELERATION * abs(self.displacement) / (self.duration**2)

    @property
    def peak_jerk(self) -> float:
        """Largest lateral jerk reached, in metres per second cubed."""
        return _PEAK_JERK * abs(self.displacement) / (self.duration**3)
