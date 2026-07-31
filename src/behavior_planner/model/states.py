"""The behaviour state and event alphabets of the finite state machine."""

from __future__ import annotations

from enum import StrEnum, unique

__all__ = ["BehaviorEvent", "BehaviorState"]


@unique
class BehaviorState(StrEnum):
    """The five behaviour states of a highway lane change planner.

    A prepare state is not a lane change. The vehicle stays laterally inside its
    current lane and adjusts its longitudinal behaviour to open a gap in the
    target lane. Only a change state moves the vehicle across the boundary.
    """

    KEEP_LANE = "keep_lane"
    PREPARE_LANE_CHANGE_LEFT = "prepare_lane_change_left"
    PREPARE_LANE_CHANGE_RIGHT = "prepare_lane_change_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"

    @property
    def lane_offset(self) -> int:
        """Lane index change this state is working towards.

        ``+1`` towards the left, ``-1`` towards the right and ``0`` for
        :attr:`KEEP_LANE`.
        """
        return _LANE_OFFSET[self]

    @property
    def is_preparing(self) -> bool:
        """True for the two prepare states."""
        return self in _PREPARING

    @property
    def is_changing(self) -> bool:
        """True for the two lane change states."""
        return self in _CHANGING


@unique
class BehaviorEvent(StrEnum):
    """The input alphabet of the finite state machine.

    The alphabet is closed: the transition table below is defined on the full
    product of :class:`BehaviorState` and :class:`BehaviorEvent`, so there is no
    input for which the machine's response is undefined.
    """

    STAY = "stay"
    """Remain in the current state for another planning cycle."""

    REQUEST_LEFT = "request_left"
    """Begin preparing a change to the lane on the left."""

    REQUEST_RIGHT = "request_right"
    """Begin preparing a change to the lane on the right."""

    COMMIT = "commit"
    """Convert a prepared change into a lateral manoeuvre."""

    ABORT = "abort"
    """Give up a prepared or running change and return to lane keeping."""

    COMPLETE = "complete"
    """Retire a lateral manoeuvre that has reached the target lane centre."""


_LANE_OFFSET: dict[BehaviorState, int] = {
    BehaviorState.KEEP_LANE: 0,
    BehaviorState.PREPARE_LANE_CHANGE_LEFT: 1,
    BehaviorState.PREPARE_LANE_CHANGE_RIGHT: -1,
    BehaviorState.LANE_CHANGE_LEFT: 1,
    BehaviorState.LANE_CHANGE_RIGHT: -1,
}

_PREPARING: frozenset[BehaviorState] = frozenset(
    {BehaviorState.PREPARE_LANE_CHANGE_LEFT, BehaviorState.PREPARE_LANE_CHANGE_RIGHT}
)

_CHANGING: frozenset[BehaviorState] = frozenset(
    {BehaviorState.LANE_CHANGE_LEFT, BehaviorState.LANE_CHANGE_RIGHT}
)
