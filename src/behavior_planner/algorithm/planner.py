"""The behaviour layer: finite state machine, cost ranking and safety veto.

One planning cycle proceeds in this order.

1. The finite state machine supplies the successors of the current state. No
   other state can be reached, whatever the cost says.
2. Successors that the road geometry forbids, a change into a lane that does not
   exist, are dropped as infeasible.
3. The safety gate rules on each remaining successor. A vetoed successor is
   removed from the candidate set before any cost is compared, so no cost can
   reinstate it.
4. The cost function ranks what is left, and the cheapest admissible candidate
   wins. Ties break towards lane keeping and then by the state's name, so the
   result never depends on the order the candidates were scored in.

Every candidate, admissible or not, is kept in the returned decision with its
terms and its verdict, which is what makes the trace readable after the fact.
The decision also carries the margin of the state it adopts, so a run can report
how much room the gate left and not only where it refused.
"""

from __future__ import annotations

from dataclasses import dataclass

from behavior_planner.algorithm.base import CostModel, SafetyGate
from behavior_planner.algorithm.cost import target_lane_for
from behavior_planner.algorithm.fsm import event_for, successors, transition
from behavior_planner.model.decision import (
    CandidateScore,
    CostTerms,
    Decision,
    DecisionContext,
    SafetyVerdict,
)
from behavior_planner.model.states import BehaviorEvent, BehaviorState
from behavior_planner.model.vehicle import LaneChange

__all__ = ["FiniteStateBehaviorPlanner", "KeepLaneBaseline"]


@dataclass(frozen=True, slots=True)
class FiniteStateBehaviorPlanner:
    """The default behaviour policy."""

    cost: CostModel
    gate: SafetyGate

    def decide(self, context: DecisionContext) -> Decision:
        """Choose the next behaviour state and the event that reaches it."""
        running = context.ego.lane_change
        if running is not None and context.state.is_changing:
            return self._resolve_running_change(context, running)
        if running is not None:
            # An aborted change leaves a manoeuvre returning the ego to the lane
            # it came from while the state is already lane keeping. No new
            # candidate may be started until that transient has retired.
            return self._stay(context, ())

        candidates = self._score_all(context)
        admissible = [candidate for candidate in candidates if candidate.admissible]
        if not admissible:
            # The only successor that is never gated is lane keeping, so this
            # branch is unreachable while the machine is in a legal state; it is
            # kept so the policy is total rather than conditionally total.
            return self._stay(context, candidates)
        best = min(admissible, key=lambda candidate: (candidate.total, candidate.state.value))
        return Decision(
            state=best.state,
            event=best.event,
            target_lane=best.target_lane,
            candidates=candidates,
            gate_margin=best.verdict.margin,
        )

    def _resolve_running_change(self, context: DecisionContext, maneuver: LaneChange) -> Decision:
        """Continue, complete or abort a lateral manoeuvre already in progress.

        A committed manoeuvre is not re-optimised. The only questions are
        whether it has arrived and whether it is still safe, in that order.
        """
        state = context.state
        if maneuver.is_complete:
            return Decision(
                state=transition(state, BehaviorEvent.COMPLETE),
                event=BehaviorEvent.COMPLETE,
                target_lane=maneuver.target_lane,
                candidates=(),
            )
        verdict = self.gate.review(context, state)
        if not verdict.allowed and maneuver.progress < context.config.abort_progress_limit:
            terms = self.cost.evaluate(context, state)
            return Decision(
                state=transition(state, BehaviorEvent.ABORT),
                event=BehaviorEvent.ABORT,
                target_lane=maneuver.source_lane,
                candidates=(
                    CandidateScore(
                        state=state,
                        event=BehaviorEvent.STAY,
                        target_lane=maneuver.target_lane,
                        terms=terms,
                        total=terms.weighted_total(context.config.cost.weights),
                        verdict=verdict,
                    ),
                ),
            )
        return Decision(
            state=transition(state, BehaviorEvent.STAY),
            event=BehaviorEvent.STAY,
            target_lane=maneuver.target_lane,
            candidates=(),
            gate_margin=verdict.margin,
        )

    def _score_all(self, context: DecisionContext) -> tuple[CandidateScore, ...]:
        """Score every feasible successor of the current state."""
        scored = []
        for successor in successors(context.state):
            if not self._is_feasible(context, successor):
                continue
            event = self._event_for(context.state, successor)
            verdict = self.gate.review(context, successor)
            terms = self.cost.evaluate(context, successor)
            scored.append(
                CandidateScore(
                    state=successor,
                    event=event,
                    target_lane=target_lane_for(context, successor),
                    terms=terms,
                    total=terms.weighted_total(context.config.cost.weights),
                    verdict=verdict,
                )
            )
        return tuple(scored)

    @staticmethod
    def _is_feasible(context: DecisionContext, successor: BehaviorState) -> bool:
        """True when the road geometry admits ``successor``.

        Feasibility is not safety. A change into a lane that does not exist is
        not a dangerous manoeuvre, it is a nonexistent one, and the distinction
        keeps the gate's veto reasons meaningful.
        """
        if successor.lane_offset == 0:
            return True
        return context.road.contains_lane(context.target_lane(successor))

    @staticmethod
    def _event_for(state: BehaviorState, successor: BehaviorState) -> BehaviorEvent:
        """Event carrying ``state`` to ``successor`` outside a running manoeuvre."""
        if state is successor:
            return BehaviorEvent.STAY
        return event_for(state, successor)

    def _stay(self, context: DecisionContext, candidates: tuple[CandidateScore, ...]) -> Decision:
        """Fall back to holding the current state."""
        return Decision(
            state=transition(context.state, BehaviorEvent.STAY),
            event=BehaviorEvent.STAY,
            target_lane=context.ego_lane,
            candidates=candidates,
        )


@dataclass(frozen=True, slots=True)
class KeepLaneBaseline:
    """A policy that never leaves its lane.

    It exists to give the suite a control. Reporting the ego's mean speed on its
    own says nothing; reporting it beside the speed the same traffic allows a
    vehicle that cannot overtake says what the behaviour layer is worth. It also
    demonstrates that the policy interface is the only coupling between the
    behaviour layer and the simulator.
    """

    def decide(self, context: DecisionContext) -> Decision:
        """Always hold :attr:`BehaviorState.KEEP_LANE`.

        The transition is taken through the state machine rather than asserted,
        so the control policy is held to the same rules as the planner.
        """
        return Decision(
            state=transition(BehaviorState.KEEP_LANE, BehaviorEvent.STAY),
            event=BehaviorEvent.STAY,
            target_lane=context.ego_lane,
            candidates=(
                CandidateScore(
                    state=BehaviorState.KEEP_LANE,
                    event=BehaviorEvent.STAY,
                    target_lane=context.ego_lane,
                    terms=_NO_COST,
                    total=0.0,
                    verdict=SafetyVerdict.allow(0.0),
                ),
            ),
        )


_NO_COST: CostTerms = CostTerms(progress=0.0, safety=0.0, comfort=0.0, lane_preference=0.0)
"""The control policy compares nothing, so its single candidate scores zero."""
