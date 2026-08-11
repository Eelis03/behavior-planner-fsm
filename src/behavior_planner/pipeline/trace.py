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
from behavior_planner.model.vehicle import EGO_ID

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

    gate_margin: float
    """Room the gate left on the manoeuvre adopted, ``inf`` when it bounded none.

    The gate rules once per planning cycle rather than once per integration step,
    so this is unbounded on every step where the behaviour layer did not run.
    """

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
    def ego_collision_pairs(self) -> tuple[tuple[int, int], ...]:
        """Overlapping pairs the ego was part of.

        Two traffic vehicles colliding with each other and the ego colliding
        with something are different failures with different owners: the first
        is a property of the traffic model, the second is a property of the
        behaviour layer. Reporting one number for both would let either hide
        behind the other.
        """
        return tuple(pair for pair in self.collision_pairs if EGO_ID in pair)

    @property
    def ego_collision_count(self) -> int:
        """Distinct vehicles the ego overlapped at any point in the run."""
        return len(self.ego_collision_pairs)

    @property
    def veto_reasons(self) -> tuple[VetoReason, ...]:
        """Every distinct veto reason raised during the run, in first-seen order."""
        seen: list[VetoReason] = []
        for record in self.records:
            if record.veto_reason is not VetoReason.NONE and record.veto_reason not in seen:
                seen.append(record.veto_reason)
        return tuple(seen)

    @property
    def minimum_gate_margin(self) -> float:
        """Least room the gate left on a manoeuvre the ego adopted, ``inf`` if none.

        The time headway and the time to collision recorded here describe the car
        following model, which governs how closely the ego follows in its own
        lane and over which the gate has no authority. This is the one quantity
        on the record that describes the gate, and it distinguishes a run in
        which no manoeuvre came close to refusal from one in which every
        manoeuvre was taken by a hair.
        """
        margins = self.gate_margins()
        return min(margins) if margins else math.inf

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

    def gate_margins(self) -> tuple[float, ...]:
        """Every bounded gate margin observed, in step order."""
        return tuple(
            record.gate_margin for record in self.records if math.isfinite(record.gate_margin)
        )
