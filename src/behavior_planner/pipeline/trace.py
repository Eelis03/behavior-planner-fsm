"""The structured record a simulation run produces.

The trace is written once, by the simulator, and read by everything downstream.
Metrics, figures and regression tests all work from it, so a quantity that is
not on the record cannot be reported, and a quantity that is on the record is
reported the same way everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from behavior_planner.model.decision import VetoReason
from behavior_planner.model.states import BehaviorEvent, BehaviorState

__all__ = ["RunTrace", "StepRecord"]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """The state of the ego and its surroundings at one simulation step."""

    time: float
    state: BehaviorState
    event: BehaviorEvent
    lane: int
    s: float
    d: float
    speed: float
    acceleration: float
    leader_gap: float
    """Bumper to bumper gap to the ego's leader, ``inf`` when there is none."""

    time_headway: float
    """Leader gap divided by ego speed, in seconds, ``inf`` when undefined."""

    time_to_collision: float
    """Time to reach the leader at the present closing rate, ``inf`` when opening."""

    veto_reason: VetoReason
    """Why the gate refused a candidate this step, ``NONE`` when it refused none."""

    planned: bool
    """True on the steps where the behaviour layer ran."""

    collisions: int
    """Number of overlapping vehicle pairs anywhere on the road this step."""


@dataclass(frozen=True, slots=True)
class RunTrace:
    """A complete run of one scenario."""

    scenario: str
    policy: str
    dt: float
    duration: float
    seed: int
    records: tuple[StepRecord, ...]
    vehicle_count: int
    collision_pairs: tuple[tuple[int, int], ...] = ()
    """Every distinct pair of vehicles that overlapped at any point in the run."""

    def __post_init__(self) -> None:
        """Reject an empty trace, which no metric is defined on."""
        if not self.records:
            raise ValueError(f"scenario {self.scenario!r} produced no records")

    @property
    def states(self) -> tuple[BehaviorState, ...]:
        """The behaviour state at every step."""
        return tuple(record.state for record in self.records)

    @property
    def state_sequence(self) -> tuple[BehaviorState, ...]:
        """The behaviour states in order, with consecutive repeats collapsed.

        This is the decision sequence a regression test can pin. It changes only
        when a transition is taken, not when a float moves in the last bit.
        """
        collapsed: list[BehaviorState] = []
        for state in self.states:
            if not collapsed or collapsed[-1] is not state:
                collapsed.append(state)
        return tuple(collapsed)

    @property
    def lane_changes(self) -> int:
        """Lateral manoeuvres the ego carried through to the target lane centre.

        Counted from the ``COMPLETE`` events of the planning steps, so an
        aborted change contributes nothing and a change in progress at the end
        of the run is not counted until it arrives.
        """
        return sum(
            1
            for record in self.records
            if record.planned and record.event is BehaviorEvent.COMPLETE
        )

    @property
    def aborted_changes(self) -> int:
        """Planning cycles that gave up a prepared or a running lane change."""
        return sum(
            1 for record in self.records if record.planned and record.event is BehaviorEvent.ABORT
        )

    @property
    def collision_count(self) -> int:
        """Distinct pairs of vehicles that overlapped at any point in the run."""
        return len(self.collision_pairs)

    @property
    def veto_reasons(self) -> tuple[VetoReason, ...]:
        """Every distinct veto reason raised during the run, in first-seen order."""
        seen: list[VetoReason] = []
        for record in self.records:
            if record.veto_reason is not VetoReason.NONE and record.veto_reason not in seen:
                seen.append(record.veto_reason)
        return tuple(seen)

    @property
    def distance_travelled(self) -> float:
        """Arc length covered by the ego over the run, in metres."""
        return sum(record.speed for record in self.records[1:]) * self.dt

    @property
    def minimum_time_headway(self) -> float:
        """Smallest finite time headway observed, ``inf`` if none was."""
        finite = [
            record.time_headway for record in self.records if math.isfinite(record.time_headway)
        ]
        return min(finite) if finite else math.inf

    def finite_times_to_collision(self) -> tuple[float, ...]:
        """Every finite time to collision observed, in step order."""
        return tuple(
            record.time_to_collision
            for record in self.records
            if math.isfinite(record.time_to_collision)
        )
