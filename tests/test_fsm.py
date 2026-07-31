"""Tier one. Invariants of the behaviour finite state machine.

The transition relation is the part of the planner that must be correct by
inspection, so it is tested exhaustively rather than by sampling: every one of
the thirty state and event pairs is exercised.
"""

from __future__ import annotations

from itertools import product

import pytest

from behavior_planner.algorithm.fsm import (
    REJECTED,
    TRANSITIONS,
    IllegalTransitionError,
    event_for,
    is_legal,
    successors,
    transition,
)
from behavior_planner.model.states import BehaviorEvent, BehaviorState

ALL_PAIRS = tuple(product(BehaviorState, BehaviorEvent))


def test_alphabets_have_the_expected_size() -> None:
    """Five behaviour states and six events, so thirty pairs to account for."""
    assert len(BehaviorState) == 5
    assert len(BehaviorEvent) == 6
    assert len(ALL_PAIRS) == 30


def test_transition_table_is_total() -> None:
    """Every state and event pair is declared either legal or illegal, never both."""
    declared = set(TRANSITIONS)
    assert declared.isdisjoint(REJECTED)
    assert declared | REJECTED == set(ALL_PAIRS)


@pytest.mark.parametrize(("state", "event"), ALL_PAIRS)
def test_every_pair_has_defined_behaviour(state: BehaviorState, event: BehaviorEvent) -> None:
    """No pair leaves the machine undefined: it returns a state or it raises.

    This is the totality property. A pair that silently returned ``None`` or
    silently returned ``state`` would pass a weaker test and hide a planner bug.
    """
    if is_legal(state, event):
        successor = transition(state, event)
        assert isinstance(successor, BehaviorState)
        assert successor is TRANSITIONS[(state, event)]
    else:
        with pytest.raises(IllegalTransitionError):
            transition(state, event)


@pytest.mark.parametrize(("state", "event"), sorted(REJECTED, key=lambda pair: str(pair)))
def test_illegal_transitions_raise_rather_than_pass(
    state: BehaviorState, event: BehaviorEvent
) -> None:
    """An illegal pair raises and the exception names both operands."""
    with pytest.raises(IllegalTransitionError) as caught:
        transition(state, event)
    assert caught.value.state is state
    assert caught.value.event is event
    assert state.value in str(caught.value)
    assert event.value in str(caught.value)


@pytest.mark.parametrize("state", list(BehaviorState))
def test_stay_is_legal_and_idempotent_in_every_state(state: BehaviorState) -> None:
    """The machine can always hold its state, and does so explicitly."""
    assert transition(state, BehaviorEvent.STAY) is state


def test_successors_match_the_declared_table() -> None:
    """The successor sets are exactly what the table declares, in enum order."""
    assert successors(BehaviorState.KEEP_LANE) == (
        BehaviorState.KEEP_LANE,
        BehaviorState.PREPARE_LANE_CHANGE_LEFT,
        BehaviorState.PREPARE_LANE_CHANGE_RIGHT,
    )
    assert successors(BehaviorState.PREPARE_LANE_CHANGE_LEFT) == (
        BehaviorState.KEEP_LANE,
        BehaviorState.PREPARE_LANE_CHANGE_LEFT,
        BehaviorState.LANE_CHANGE_LEFT,
    )
    assert successors(BehaviorState.PREPARE_LANE_CHANGE_RIGHT) == (
        BehaviorState.KEEP_LANE,
        BehaviorState.PREPARE_LANE_CHANGE_RIGHT,
        BehaviorState.LANE_CHANGE_RIGHT,
    )
    assert successors(BehaviorState.LANE_CHANGE_LEFT) == (
        BehaviorState.KEEP_LANE,
        BehaviorState.LANE_CHANGE_LEFT,
    )
    assert successors(BehaviorState.LANE_CHANGE_RIGHT) == (
        BehaviorState.KEEP_LANE,
        BehaviorState.LANE_CHANGE_RIGHT,
    )


def test_a_prepare_state_cannot_switch_direction_without_passing_through_keep_lane() -> None:
    """Reversing a prepared change is a two step path, not a one step one."""
    left = BehaviorState.PREPARE_LANE_CHANGE_LEFT
    assert BehaviorState.PREPARE_LANE_CHANGE_RIGHT not in successors(left)
    returned = transition(left, BehaviorEvent.ABORT)
    assert returned is BehaviorState.KEEP_LANE
    assert (
        transition(returned, BehaviorEvent.REQUEST_RIGHT)
        is BehaviorState.PREPARE_LANE_CHANGE_RIGHT
    )


@pytest.mark.parametrize("state", list(BehaviorState))
def test_keep_lane_is_reachable_from_every_state(state: BehaviorState) -> None:
    """There is always a way back to lane keeping, which makes the machine live."""
    assert BehaviorState.KEEP_LANE in successors(state)


def test_event_for_recovers_the_unique_event() -> None:
    """A successor reached by exactly one event is labelled by that event."""
    assert (
        event_for(BehaviorState.KEEP_LANE, BehaviorState.PREPARE_LANE_CHANGE_LEFT)
        is BehaviorEvent.REQUEST_LEFT
    )
    assert (
        event_for(BehaviorState.PREPARE_LANE_CHANGE_RIGHT, BehaviorState.LANE_CHANGE_RIGHT)
        is BehaviorEvent.COMMIT
    )


def test_event_for_refuses_to_guess_an_ambiguous_label() -> None:
    """Abort and complete both land in lane keeping, so the caller must choose."""
    with pytest.raises(ValueError, match="must choose"):
        event_for(BehaviorState.LANE_CHANGE_LEFT, BehaviorState.KEEP_LANE)


def test_event_for_rejects_an_unreachable_successor() -> None:
    """A successor outside the table is an error, not an empty answer."""
    with pytest.raises(ValueError, match="not a successor"):
        event_for(BehaviorState.KEEP_LANE, BehaviorState.LANE_CHANGE_LEFT)


@pytest.mark.parametrize(
    ("state", "offset"),
    [
        (BehaviorState.KEEP_LANE, 0),
        (BehaviorState.PREPARE_LANE_CHANGE_LEFT, 1),
        (BehaviorState.LANE_CHANGE_LEFT, 1),
        (BehaviorState.PREPARE_LANE_CHANGE_RIGHT, -1),
        (BehaviorState.LANE_CHANGE_RIGHT, -1),
    ],
)
def test_lane_offset_matches_the_direction_of_the_state(
    state: BehaviorState, offset: int
) -> None:
    """Left raises the lane index, right lowers it, keep lane leaves it alone."""
    assert state.lane_offset == offset
