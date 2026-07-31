"""The interfaces the decision layers are written against.

Every component below is a Protocol, so a car following model, a lane change
model for traffic, a cost function, a safety gate or a whole behaviour policy
can be replaced without touching the simulator. The concrete implementations in
this package are the defaults, not the contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from behavior_planner.model.config import DriverParams
from behavior_planner.model.decision import (
    CostTerms,
    Decision,
    DecisionContext,
    SafetyVerdict,
)
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle

__all__ = [
    "BehaviorPolicy",
    "CarFollowingModel",
    "CostModel",
    "LaneChangeModel",
    "SafetyGate",
]


@runtime_checkable
class CarFollowingModel(Protocol):
    """Longitudinal acceleration of a vehicle given its leader."""

    def acceleration(
        self,
        *,
        speed: float,
        gap: float,
        leader_speed: float,
        driver: DriverParams,
    ) -> float:
        """Acceleration in metres per second squared.

        ``gap`` is bumper to bumper and may be ``math.inf`` when no leader is in
        range, in which case ``leader_speed`` is ignored.
        """
        ...


@runtime_checkable
class LaneChangeModel(Protocol):
    """Lane change decisions for a traffic vehicle, not for the ego."""

    def choose_lane(self, subject: Vehicle, snapshot: TrafficSnapshot) -> int | None:
        """Lane the subject should move into, or ``None`` to stay put."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """Scoring of one candidate successor behaviour state."""

    def evaluate(self, context: DecisionContext, successor: BehaviorState) -> CostTerms:
        """The normalised cost terms of moving to ``successor``."""
        ...


@runtime_checkable
class SafetyGate(Protocol):
    """A veto over candidate manoeuvres, independent of any cost."""

    def review(self, context: DecisionContext, successor: BehaviorState) -> SafetyVerdict:
        """Whether ``successor`` may be entered at all."""
        ...


@runtime_checkable
class BehaviorPolicy(Protocol):
    """The behaviour layer: current situation in, next behaviour state out."""

    def decide(self, context: DecisionContext) -> Decision:
        """Choose the next behaviour state and the event that reaches it."""
        ...
