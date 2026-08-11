"""Records exchanged between the decision layers: context, costs, verdicts.

These are pure data. Keeping them here rather than beside the code that
produces them lets the cost function and the safety gate stay independent of
each other while both remain describable by one trace record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, unique

from behavior_planner.model.config import CostWeights, PlannerConfig
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorEvent, BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle

__all__ = [
    "CandidateScore",
    "CostTerms",
    "Decision",
    "DecisionContext",
    "SafetyVerdict",
    "VetoReason",
]


@unique
class VetoReason(StrEnum):
    """Why the safety gate refused a manoeuvre, or that it did not."""

    NONE = "none"
    """No objection. Present so that an allowed verdict carries a reason too."""

    OFF_ROAD = "off_road"
    """The target lane does not exist."""

    LEADER_GAP = "leader_gap"
    """The gap to the target lane leader is below the hard minimum."""

    FOLLOWER_GAP = "follower_gap"
    """The gap to the target lane follower is below the hard minimum."""

    LEADER_TIME_TO_COLLISION = "leader_time_to_collision"
    """The ego is closing on the target lane leader too quickly."""

    FOLLOWER_TIME_TO_COLLISION = "follower_time_to_collision"
    """The target lane follower is closing on the ego too quickly."""

    FOLLOWER_DECELERATION = "follower_deceleration"
    """The manoeuvre would force the new follower to brake harder than allowed."""

    OCCUPIED = "occupied"
    """A vehicle already overlaps the space the ego would move into."""


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """The safety gate's ruling on one candidate manoeuvre."""

    allowed: bool
    reason: VetoReason = VetoReason.NONE
    margin: float = 0.0
    """Smallest normalised margin across the checks, negative when vetoed."""

    def __post_init__(self) -> None:
        """Reject a verdict whose flag and reason disagree."""
        if self.allowed and self.reason is not VetoReason.NONE:
            raise ValueError(f"an allowed verdict must carry no reason, got {self.reason}")
        if not self.allowed and self.reason is VetoReason.NONE:
            raise ValueError("a veto must name a reason")

    @classmethod
    def allow(cls, margin: float) -> SafetyVerdict:
        """A permitting verdict with the given margin."""
        return cls(allowed=True, reason=VetoReason.NONE, margin=margin)

    @classmethod
    def veto(cls, reason: VetoReason, margin: float) -> SafetyVerdict:
        """A refusing verdict naming ``reason``."""
        return cls(allowed=False, reason=reason, margin=margin)


@dataclass(frozen=True, slots=True)
class CostTerms:
    """The four normalised cost terms scored for one candidate successor.

    Every term lies in ``[0, 1]``, so a weight in :class:`CostWeights` reads
    directly as the penalty charged for the worst case of that term.
    """

    progress: float
    safety: float
    comfort: float
    lane_preference: float

    def weighted_total(self, weights: CostWeights) -> float:
        """Weighted sum of the four terms."""
        return (
            weights.progress * self.progress
            + weights.safety * self.safety
            + weights.comfort * self.comfort
            + weights.lane_preference * self.lane_preference
        )

    def as_dict(self) -> dict[str, float]:
        """The terms keyed by name, for tracing and reporting."""
        return {
            "progress": self.progress,
            "safety": self.safety,
            "comfort": self.comfort,
            "lane_preference": self.lane_preference,
        }


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Everything the behaviour layer is allowed to look at.

    Passing one record rather than a handful of arguments is what lets an
    alternative policy be substituted without changing any call site.
    """

    road: Road
    snapshot: TrafficSnapshot
    ego: Vehicle
    state: BehaviorState
    config: PlannerConfig
    time: float = 0.0

    @property
    def ego_lane(self) -> int:
        """Lane the ego counts as belonging to."""
        return self.ego.assigned_lane(self.road)

    def target_lane(self, successor: BehaviorState) -> int:
        """Lane ``successor`` works towards, which may be off the road.

        A manoeuvre already in progress has a target of its own, and the ego is
        assigned to that target from the moment it commits. Adding the state's
        lane offset a second time would aim one lane further out, which is how a
        change into the rightmost lane comes to look like a change off the road.
        The result is deliberately not clamped so the caller can detect that.
        """
        running = self.ego.lane_change
        if running is not None and successor.is_changing:
            return running.target_lane
        return self.ego_lane + successor.lane_offset


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One successor state, its cost, and the safety gate's ruling on it."""

    state: BehaviorState
    event: BehaviorEvent
    target_lane: int
    terms: CostTerms
    total: float
    verdict: SafetyVerdict

    @property
    def admissible(self) -> bool:
        """True when the gate permits this candidate."""
        return self.verdict.allowed


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one behaviour planning cycle."""

    state: BehaviorState
    event: BehaviorEvent
    target_lane: int
    candidates: tuple[CandidateScore, ...] = ()
    gate_margin: float = math.inf
    """Room the gate left on the state this decision adopts.

    A verdict carries a number as well as a boolean, and the number is how far
    the manoeuvre sat from whichever threshold bound it, as a fraction of that
    threshold. Recording only the boolean leaves a run able to report where the
    gate refused and unable to report how close the manoeuvres it permitted came
    to being refused, which are the same measurement seen from either side.

    ``inf`` where the gate placed no bound, which is every decision that moves
    the ego nowhere sideways and every change into an empty lane. Negative in
    the single case where the gate is overruled: a manoeuvre past
    :attr:`PlannerConfig.abort_progress_limit` is finished rather than reversed,
    whatever the gate has come to think of it.
    """

    @property
    def vetoed(self) -> tuple[CandidateScore, ...]:
        """Candidates the gate refused, in the order they were scored."""
        return tuple(candidate for candidate in self.candidates if not candidate.admissible)
