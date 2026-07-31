"""The behaviour finite state machine: an explicit, total, checked transition table.

Two properties are enforced here rather than assumed.

*Totality.* The union of :data:`TRANSITIONS` and :data:`REJECTED` is the whole
product of the state and event alphabets, and the two are disjoint. The check
runs at import, so a pair that is neither declared legal nor declared illegal is
an error the moment the module loads, not a surprise at run time.

*Explicit rejection.* An illegal pair raises :class:`IllegalTransitionError`. It
does not silently return the current state, which is the failure mode that lets
a planner bug hide as a vehicle that mysteriously never changes lane.
"""

from __future__ import annotations

from itertools import product
from typing import Final

from behavior_planner.model.states import BehaviorEvent, BehaviorState

__all__ = [
    "REJECTED",
    "TRANSITIONS",
    "IllegalTransitionError",
    "event_for",
    "is_legal",
    "successors",
    "transition",
]

_S = BehaviorState
_E = BehaviorEvent


class IllegalTransitionError(ValueError):
    """Raised when a state and event pair is not a declared transition."""

    def __init__(self, state: BehaviorState, event: BehaviorEvent) -> None:
        super().__init__(f"event {event.value!r} is not legal in state {state.value!r}")
        self.state = state
        self.event = event


TRANSITIONS: Final[dict[tuple[BehaviorState, BehaviorEvent], BehaviorState]] = {
    # Lane keeping may continue or begin preparing a change in either direction.
    (_S.KEEP_LANE, _E.STAY): _S.KEEP_LANE,
    (_S.KEEP_LANE, _E.REQUEST_LEFT): _S.PREPARE_LANE_CHANGE_LEFT,
    (_S.KEEP_LANE, _E.REQUEST_RIGHT): _S.PREPARE_LANE_CHANGE_RIGHT,
    # A prepared change may continue, commit, or be given up. It may not switch
    # direction: that has to pass back through lane keeping, so the decision to
    # go the other way is taken with the ego properly centred.
    (_S.PREPARE_LANE_CHANGE_LEFT, _E.STAY): _S.PREPARE_LANE_CHANGE_LEFT,
    (_S.PREPARE_LANE_CHANGE_LEFT, _E.COMMIT): _S.LANE_CHANGE_LEFT,
    (_S.PREPARE_LANE_CHANGE_LEFT, _E.ABORT): _S.KEEP_LANE,
    (_S.PREPARE_LANE_CHANGE_RIGHT, _E.STAY): _S.PREPARE_LANE_CHANGE_RIGHT,
    (_S.PREPARE_LANE_CHANGE_RIGHT, _E.COMMIT): _S.LANE_CHANGE_RIGHT,
    (_S.PREPARE_LANE_CHANGE_RIGHT, _E.ABORT): _S.KEEP_LANE,
    # A running change may continue, be abandoned, or be retired on arrival.
    # Abort and complete both land in lane keeping but are distinct events: they
    # differ in which lane the ego ends up in, and the trace has to say which.
    (_S.LANE_CHANGE_LEFT, _E.STAY): _S.LANE_CHANGE_LEFT,
    (_S.LANE_CHANGE_LEFT, _E.ABORT): _S.KEEP_LANE,
    (_S.LANE_CHANGE_LEFT, _E.COMPLETE): _S.KEEP_LANE,
    (_S.LANE_CHANGE_RIGHT, _E.STAY): _S.LANE_CHANGE_RIGHT,
    (_S.LANE_CHANGE_RIGHT, _E.ABORT): _S.KEEP_LANE,
    (_S.LANE_CHANGE_RIGHT, _E.COMPLETE): _S.KEEP_LANE,
}
"""Every legal transition, one entry per pair."""

REJECTED: Final[frozenset[tuple[BehaviorState, BehaviorEvent]]] = frozenset(
    {
        # Nothing to commit, abort or complete while merely keeping lane.
        (_S.KEEP_LANE, _E.COMMIT),
        (_S.KEEP_LANE, _E.ABORT),
        (_S.KEEP_LANE, _E.COMPLETE),
        # A prepare state cannot be re-requested or completed: no lateral motion
        # has happened yet, so there is nothing to complete.
        (_S.PREPARE_LANE_CHANGE_LEFT, _E.REQUEST_LEFT),
        (_S.PREPARE_LANE_CHANGE_LEFT, _E.REQUEST_RIGHT),
        (_S.PREPARE_LANE_CHANGE_LEFT, _E.COMPLETE),
        (_S.PREPARE_LANE_CHANGE_RIGHT, _E.REQUEST_LEFT),
        (_S.PREPARE_LANE_CHANGE_RIGHT, _E.REQUEST_RIGHT),
        (_S.PREPARE_LANE_CHANGE_RIGHT, _E.COMPLETE),
        # A running change cannot be committed again, nor redirected.
        (_S.LANE_CHANGE_LEFT, _E.REQUEST_LEFT),
        (_S.LANE_CHANGE_LEFT, _E.REQUEST_RIGHT),
        (_S.LANE_CHANGE_LEFT, _E.COMMIT),
        (_S.LANE_CHANGE_RIGHT, _E.REQUEST_LEFT),
        (_S.LANE_CHANGE_RIGHT, _E.REQUEST_RIGHT),
        (_S.LANE_CHANGE_RIGHT, _E.COMMIT),
    }
)
"""Every pair that is deliberately not a transition."""


def _check_table_is_total() -> None:
    """Verify that the declared pairs partition the state and event product."""
    universe = set(product(BehaviorState, BehaviorEvent))
    declared = set(TRANSITIONS)
    overlap = declared & REJECTED
    if overlap:
        raise RuntimeError(f"pairs declared both legal and illegal: {sorted(overlap)}")
    missing = universe - declared - REJECTED
    if missing:
        raise RuntimeError(f"transition table is not total, undeclared pairs: {sorted(missing)}")
    extra = (declared | REJECTED) - universe
    if extra:
        raise RuntimeError(
            f"transition table declares pairs outside the alphabets: {sorted(extra)}"
        )


_check_table_is_total()


def is_legal(state: BehaviorState, event: BehaviorEvent) -> bool:
    """True when ``event`` is accepted in ``state``."""
    return (state, event) in TRANSITIONS


def transition(state: BehaviorState, event: BehaviorEvent) -> BehaviorState:
    """Successor of ``state`` under ``event``.

    Raises :class:`IllegalTransitionError` for a rejected pair. The machine has
    no silent no-op: a caller that wants to stay put must say so with
    :attr:`BehaviorEvent.STAY`.
    """
    successor = TRANSITIONS.get((state, event))
    if successor is None:
        raise IllegalTransitionError(state, event)
    return successor


def successors(state: BehaviorState) -> tuple[BehaviorState, ...]:
    """Every state reachable from ``state`` in one transition, ordered.

    The ordering is the declaration order of :class:`BehaviorState`, so the
    candidate list a policy scores does not depend on dictionary iteration.
    """
    reachable = {target for (source, _), target in TRANSITIONS.items() if source is state}
    return tuple(member for member in BehaviorState if member in reachable)


def event_for(state: BehaviorState, successor: BehaviorState) -> BehaviorEvent:
    """The event that carries ``state`` to ``successor``.

    Where two events reach the same successor, which happens for a running lane
    change that may either abort or complete, the caller must pick the event
    itself. This function then raises, because guessing would put the wrong
    label in the trace.
    """
    matches = tuple(
        event
        for (source, event), target in TRANSITIONS.items()
        if source is state and target is successor
    )
    if not matches:
        raise ValueError(f"{successor.value!r} is not a successor of {state.value!r}")
    if len(matches) > 1:
        raise ValueError(
            f"transition {state.value!r} to {successor.value!r} is reachable by "
            f"{sorted(event.value for event in matches)}; the caller must choose"
        )
    return matches[0]
