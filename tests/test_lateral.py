"""Tier one. The minimum jerk lateral profile.

Every quantity here is closed form, so the assertions are exact to floating
point tolerance rather than to a recorded value.
"""

from __future__ import annotations

import math

import pytest

from behavior_planner.model.lateral import (
    LateralProfile,
    quintic_acceleration,
    quintic_progress,
    quintic_rate,
)


def test_the_polynomial_satisfies_its_boundary_conditions() -> None:
    """Zero progress, rate and acceleration at both ends, full progress at the end."""
    assert quintic_progress(0.0) == pytest.approx(0.0)
    assert quintic_progress(1.0) == pytest.approx(1.0)
    for end in (0.0, 1.0):
        assert quintic_rate(end) == pytest.approx(0.0)
        assert quintic_acceleration(end) == pytest.approx(0.0)


def test_the_polynomial_is_symmetric_about_its_midpoint() -> None:
    """The minimum jerk transition spends the same time either side of halfway."""
    for tau in (0.1, 0.25, 0.4):
        assert quintic_progress(tau) + quintic_progress(1.0 - tau) == pytest.approx(1.0)
    assert quintic_progress(0.5) == pytest.approx(0.5)


def test_progress_is_monotone_and_clamped() -> None:
    """The profile never goes backwards, and it saturates outside ``[0, 1]``."""
    previous = -1.0
    for index in range(0, 101):
        value = quintic_progress(index / 100.0)
        assert value >= previous
        previous = value
    assert quintic_progress(-2.0) == pytest.approx(0.0)
    assert quintic_progress(3.0) == pytest.approx(1.0)


def test_a_profile_starts_and_ends_where_it_says() -> None:
    """The offset matches the endpoints and stays inside them throughout."""
    profile = LateralProfile(start=0.0, target=3.7, duration=3.0)
    assert profile.offset(0.0) == pytest.approx(0.0)
    assert profile.offset(3.0) == pytest.approx(3.7)
    assert profile.offset(10.0) == pytest.approx(3.7)
    for step in range(31):
        assert 0.0 <= profile.offset(step / 10.0) <= 3.7


def test_peak_rate_matches_the_closed_form() -> None:
    """The peak lateral speed is ``1.875 * displacement / duration``."""
    profile = LateralProfile(start=0.0, target=3.7, duration=3.0)
    assert profile.peak_rate == pytest.approx(1.875 * 3.7 / 3.0)
    sampled = max(profile.rate(index / 2000.0 * 3.0) for index in range(2001))
    assert sampled == pytest.approx(profile.peak_rate, rel=1e-5)


def test_peak_acceleration_matches_the_closed_form() -> None:
    """The peak lateral acceleration is ``(10 / sqrt(3)) * displacement / duration^2``."""
    profile = LateralProfile(start=0.0, target=3.7, duration=3.0)
    expected = 10.0 / math.sqrt(3.0) * 3.7 / 9.0
    assert profile.peak_acceleration == pytest.approx(expected)
    sampled = max(abs(profile.acceleration(index / 2000.0 * 3.0)) for index in range(2001))
    assert sampled == pytest.approx(profile.peak_acceleration, rel=1e-5)


def test_peak_jerk_matches_the_closed_form() -> None:
    """The peak lateral jerk is ``60 * displacement / duration^3``, reached at the ends."""
    profile = LateralProfile(start=0.0, target=3.7, duration=3.0)
    assert profile.peak_jerk == pytest.approx(60.0 * 3.7 / 27.0)


def test_a_longer_manoeuvre_is_gentler_in_every_derivative() -> None:
    """Doubling the duration halves the peak rate and quarters the peak acceleration."""
    short = LateralProfile(start=0.0, target=3.7, duration=2.0)
    long = LateralProfile(start=0.0, target=3.7, duration=4.0)
    assert long.peak_rate == pytest.approx(0.5 * short.peak_rate)
    assert long.peak_acceleration == pytest.approx(0.25 * short.peak_acceleration)
    assert long.peak_jerk == pytest.approx(0.125 * short.peak_jerk)


def test_a_rightward_profile_mirrors_a_leftward_one() -> None:
    """Direction changes the sign of the offset, not the magnitude of the peaks."""
    left = LateralProfile(start=0.0, target=3.7, duration=3.0)
    right = LateralProfile(start=3.7, target=0.0, duration=3.0)
    assert right.peak_rate == pytest.approx(left.peak_rate)
    assert right.rate(1.5) == pytest.approx(-left.rate(1.5))
    assert right.offset(1.5) + left.offset(1.5) == pytest.approx(3.7)


def test_a_non_positive_duration_is_rejected() -> None:
    """A profile with no duration has no derivatives, so it cannot be built."""
    with pytest.raises(ValueError, match="duration"):
        LateralProfile(start=0.0, target=3.7, duration=0.0)
