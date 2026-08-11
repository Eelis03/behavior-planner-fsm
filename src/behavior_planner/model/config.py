"""Declared configuration for the driver models, the cost function and the gate.

Every tunable number the planner depends on is a field of one of the frozen
dataclasses below, each with its unit and the reason for its default. Nothing in
``algorithm/`` or ``pipeline/`` contains a bare numeric constant that changes
behaviour: a constant that is not here is a mathematical constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CostConfig",
    "CostWeights",
    "DriverParams",
    "IdmParams",
    "MobilParams",
    "PlannerConfig",
    "SafetyLimits",
]


@dataclass(frozen=True, slots=True)
class IdmParams:
    """Intelligent Driver Model parameters for one driver.

    Defaults follow the highway calibration reported by Treiber, Hennecke and
    Helbing (2000): desired speed 120 km/h, safe time gap 1.6 s, jam distance
    2 m, maximum acceleration 0.73 m/s^2, comfortable deceleration 1.67 m/s^2
    and acceleration exponent 4.
    """

    desired_speed: float = 33.3
    """Free flow speed v0, in metres per second."""

    time_headway: float = 1.6
    """Safe time gap T to the leader, in seconds."""

    minimum_gap: float = 2.0
    """Jam distance s0 kept at a standstill, in metres."""

    maximum_acceleration: float = 0.73
    """Acceleration a in free flow, in metres per second squared."""

    comfortable_deceleration: float = 1.67
    """Deceleration b the driver is willing to apply, in metres per second squared."""

    acceleration_exponent: float = 4.0
    """Exponent delta shaping the approach to the desired speed."""

    def __post_init__(self) -> None:
        """Reject parameters outside the range the model is defined on."""
        if self.desired_speed <= 0.0:
            raise ValueError(f"desired_speed must be positive, got {self.desired_speed}")
        if self.time_headway < 0.0:
            raise ValueError(f"time_headway must not be negative, got {self.time_headway}")
        if self.minimum_gap <= 0.0:
            raise ValueError(f"minimum_gap must be positive, got {self.minimum_gap}")
        if self.maximum_acceleration <= 0.0:
            raise ValueError(
                f"maximum_acceleration must be positive, got {self.maximum_acceleration}"
            )
        if self.comfortable_deceleration <= 0.0:
            raise ValueError(
                f"comfortable_deceleration must be positive, got {self.comfortable_deceleration}"
            )
        if self.acceleration_exponent <= 0.0:
            raise ValueError(
                f"acceleration_exponent must be positive, got {self.acceleration_exponent}"
            )


@dataclass(frozen=True, slots=True)
class MobilParams:
    """MOBIL lane change parameters for one driver.

    Defaults sit inside the ranges discussed by Kesting, Treiber and Helbing
    (2007). The politeness factor is deliberately below one half so that traffic
    is neither perfectly altruistic nor purely selfish, which is the regime in
    which the ego vehicle has to negotiate for a gap rather than being handed
    one.
    """

    politeness: float = 0.25
    """Weight p given to the acceleration of the affected neighbours."""

    changing_threshold: float = 0.1
    """Switching threshold delta_a_th, in metres per second squared."""

    safe_deceleration: float = 4.0
    """Largest deceleration b_safe that may be imposed on the new follower."""

    right_bias: float = 0.0
    """Asymmetric keep-right bias a_bias, in metres per second squared.

    Zero selects the symmetric rule of the source paper. A positive value adds
    the bias to a change towards a lower lane index and subtracts it from a
    change towards a higher one, which is the asymmetric European variant.
    """

    def __post_init__(self) -> None:
        """Reject parameters outside the range the model is defined on."""
        if not 0.0 <= self.politeness <= 1.0:
            raise ValueError(f"politeness must lie in [0, 1], got {self.politeness}")
        if self.changing_threshold < 0.0:
            raise ValueError(
                f"changing_threshold must not be negative, got {self.changing_threshold}"
            )
        if self.safe_deceleration <= 0.0:
            raise ValueError(f"safe_deceleration must be positive, got {self.safe_deceleration}")
        if self.right_bias < 0.0:
            raise ValueError(f"right_bias must not be negative, got {self.right_bias}")


@dataclass(frozen=True, slots=True)
class DriverParams:
    """The complete behaviour of one traffic participant."""

    idm: IdmParams = field(default_factory=IdmParams)
    mobil: MobilParams = field(default_factory=MobilParams)


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Relative importance of the four cost terms.

    The terms are each normalised to ``[0, 1]`` before weighting, so a weight is
    directly the penalty charged for the worst case of that term. The defaults
    satisfy two requirements that pull against each other, and the admissible
    window between them is narrow:

    * The ego must overtake. Leaving a lane whose leader is slow has to beat
      staying in it, so ``progress`` times half the speed shortfall must exceed
      ``comfort`` plus ``lane_preference`` times the normalised lane distance.
    * The ego must come back. Once the obstruction is passed, the preference
      term alone has to pay for a further lane change, so ``lane_preference``
      times the normalised lane distance must exceed ``comfort``.

    The second inequality bounds ``comfort`` from above by roughly half of
    ``lane_preference``, and the first then bounds ``lane_preference`` from
    above by the smallest speed advantage that ought to provoke a manoeuvre.
    ``docs/design-notes.md`` records why the window is narrow and what that
    implies about hand-tuned cost functions in general.
    """

    progress: float = 5.0
    """Speed shortfall relative to the free flow speed."""

    safety: float = 1.5
    """Shortfall of the desired gaps, a soft term distinct from the safety gate.

    Setting this to zero does not make the planner unsafe, because the gate is
    not part of the cost. It makes the planner rude: it then proposes merges
    into gaps that are legal but tight, and the gate has to refuse them. That
    difference is visible in the veto reasons the suite records.
    """

    comfort: float = 0.10
    """Lateral effort demanded by the manoeuvre."""

    lane_preference: float = 0.35
    """Distance from the preferred lane."""

    def __post_init__(self) -> None:
        """Reject weights that are negative or uniformly zero."""
        values = (self.progress, self.safety, self.comfort, self.lane_preference)
        if any(value < 0.0 for value in values):
            raise ValueError(f"cost weights must not be negative, got {values}")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one cost weight must be positive")


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Everything the cost function reads.

    The desired gaps below are the cost function's own notion of roomy, and are
    deliberately not the safety gate's thresholds. Sharing them would make the
    gate a stricter copy of a cost term and invite the reading that safety is
    negotiable at a price.
    """

    weights: CostWeights = field(default_factory=CostWeights)

    preferred_lane: int = 0
    """Lane the ego returns to when nothing else is at stake, rightmost by default."""

    progress_horizon: float = 120.0
    """Range over which a leader reduces the speed a lane can deliver, in metres."""

    desired_standstill_gap: float = 5.0
    """Gap the soft safety term wants at zero speed, in metres."""

    desired_leader_headway: float = 1.0
    """Time gap the soft safety term wants to the leader, in seconds."""

    desired_follower_headway: float = 0.6
    """Time gap the soft safety term wants to leave the follower, in seconds."""

    prepare_effort: float = 1.0
    """Normalised lateral effort charged to a prepare state."""

    change_effort: float = 0.9
    """Normalised lateral effort charged to a lane change state.

    Lower than :attr:`prepare_effort` on purpose. A prepare state is a
    transient: the planner has already accepted the cost of a lane change and is
    only waiting for a gap, so lingering there must cost more than finishing.
    """

    def __post_init__(self) -> None:
        """Reject a configuration the cost function cannot normalise."""
        if self.preferred_lane < 0:
            raise ValueError(f"preferred_lane must not be negative, got {self.preferred_lane}")
        if self.progress_horizon <= 0.0:
            raise ValueError(f"progress_horizon must be positive, got {self.progress_horizon}")
        if self.desired_standstill_gap <= 0.0:
            raise ValueError(
                f"desired_standstill_gap must be positive, got {self.desired_standstill_gap}"
            )
        if self.desired_leader_headway < 0.0:
            raise ValueError("desired_leader_headway must not be negative")
        if self.desired_follower_headway < 0.0:
            raise ValueError("desired_follower_headway must not be negative")
        if self.prepare_effort < 0.0 or self.change_effort < 0.0:
            raise ValueError("effort values must not be negative")


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Hard limits enforced by the safety gate.

    None of these numbers appears in the cost function. A manoeuvre that
    violates any one of them is rejected outright, whatever it would have
    scored.
    """

    minimum_leader_gap: float = 5.0
    """Gap to the target lane leader at a standstill, in metres."""

    leader_headway: float = 0.8
    """Additional gap to the leader per unit of ego speed, in seconds."""

    minimum_follower_gap: float = 5.0
    """Gap to the target lane follower at a standstill, in metres."""

    follower_headway: float = 0.5
    """Additional gap to the follower per unit of follower speed, in seconds."""

    minimum_time_to_collision: float = 3.0
    """Smallest permitted time to collision with either neighbour, in seconds."""

    maximum_follower_deceleration: float = 3.0
    """Largest deceleration the manoeuvre may force on the new follower.

    This is the MOBIL safety criterion applied to the ego's own manoeuvre rather
    than to a traffic vehicle's, and it is the reason a merge that is
    geometrically legal but rude is still refused.
    """

    def __post_init__(self) -> None:
        """Reject limits that would disable the gate."""
        values = (
            self.minimum_leader_gap,
            self.minimum_follower_gap,
            self.minimum_time_to_collision,
            self.maximum_follower_deceleration,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError(f"safety limits must be positive, got {values}")
        if self.leader_headway < 0.0 or self.follower_headway < 0.0:
            raise ValueError("safety headways must not be negative")


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """The complete configuration of the behaviour planner."""

    cost: CostConfig = field(default_factory=CostConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    planning_period: float = 0.5
    """Interval between behaviour decisions, in seconds.

    The trajectory layer runs at the simulation step, the behaviour layer at
    this period. Deciding at every step makes the finite state machine chatter
    on numerical noise.
    """

    lane_change_duration: float = 3.5
    """Duration of the lateral transition, in seconds."""

    trajectory_horizon: float = 2.0
    """Length of the trajectory generated for the chosen manoeuvre, in seconds."""

    abort_progress_limit: float = 0.4
    """Fraction of the lateral transition beyond which an abort is refused.

    Reversing a lane change that is more than this far along puts the vehicle
    back across a lane boundary it has already cleared, which is worse than
    finishing.
    """

    def __post_init__(self) -> None:
        """Reject timings that the simulator cannot honour."""
        if self.planning_period <= 0.0:
            raise ValueError(f"planning_period must be positive, got {self.planning_period}")
        if self.lane_change_duration <= 0.0:
            raise ValueError(
                f"lane_change_duration must be positive, got {self.lane_change_duration}"
            )
        if self.trajectory_horizon <= 0.0:
            raise ValueError(f"trajectory_horizon must be positive, got {self.trajectory_horizon}")
        if not 0.0 <= self.abort_progress_limit <= 1.0:
            raise ValueError(
                f"abort_progress_limit must lie in [0, 1], got {self.abort_progress_limit}"
            )
