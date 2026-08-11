"""The cost-based decision layer: four weighted, normalised terms per successor.

The cost function ranks manoeuvres. It does not decide whether one is allowed.
Every number it reads lives in :class:`CostConfig`, and every term it produces
is normalised to ``[0, 1]`` before weighting, so the weights are comparable to
each other and a change to one of them has a bounded effect.
"""

from __future__ import annotations

from dataclasses import dataclass

from behavior_planner.model.config import CostConfig
from behavior_planner.model.decision import CostTerms, DecisionContext
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import LaneNeighbors
from behavior_planner.model.vehicle import Vehicle

__all__ = ["WeightedCostModel", "target_lane_for"]


def target_lane_for(context: DecisionContext, successor: BehaviorState) -> int:
    """Lane ``successor`` is working towards, clamped to the road.

    The caller is expected to have filtered infeasible successors already; the
    clamp is here so scoring a successor never raises.
    """
    return min(max(context.target_lane(successor), 0), context.road.lane_count - 1)


def _clip_unit(value: float) -> float:
    """Clamp ``value`` to ``[0, 1]``."""
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class WeightedCostModel:
    """The default cost function.

    The four terms are:

    *Progress.* The speed a lane can deliver, measured as the shortfall against
    the ego's free flow speed. A leader reduces it in proportion to how close it
    is, linearly over :attr:`CostConfig.progress_horizon`, so a slow vehicle two
    hundred metres away does not provoke a lane change while the same vehicle at
    thirty metres does. A prepare state is scored on the mean of the lane it is
    aiming at and the lane it is still in, because that is where the ego
    actually is: this is what stops the machine from parking in a prepare state,
    which would otherwise collect the benefit of the target lane without ever
    paying for the manoeuvre.

    *Safety.* The largest shortfall against the cost function's own desired gaps
    to the leader and the follower in the evaluated lane. This is a soft term.
    It shapes the ranking; it cannot forbid anything.

    *Comfort.* The lateral effort of the successor state, taken from the
    configuration. Lane keeping is free, a prepare state costs
    :attr:`CostConfig.prepare_effort` and a change costs
    :attr:`CostConfig.change_effort`.

    *Lane preference.* Normalised distance from
    :attr:`CostConfig.preferred_lane`, which encodes the keep-right rule.
    """

    config: CostConfig

    def evaluate(self, context: DecisionContext, successor: BehaviorState) -> CostTerms:
        """The normalised cost terms of moving to ``successor``."""
        current_lane = context.ego_lane
        target = target_lane_for(context, successor)
        return CostTerms(
            progress=self._progress(context, current_lane, target, successor),
            safety=self._safety(context, target),
            comfort=self._comfort(successor),
            lane_preference=self._lane_preference(context, target),
        )

    def _progress(
        self,
        context: DecisionContext,
        current_lane: int,
        target_lane: int,
        successor: BehaviorState,
    ) -> float:
        """Speed shortfall of the lane or lanes this successor commits the ego to."""
        target_cost = self._lane_inefficiency(context, target_lane)
        if not successor.is_preparing:
            return target_cost
        current_cost = self._lane_inefficiency(context, current_lane)
        return 0.5 * (target_cost + current_cost)

    def _lane_inefficiency(self, context: DecisionContext, lane: int) -> float:
        """Normalised shortfall of the speed ``lane`` can deliver to the ego."""
        ego = context.ego
        free_speed = ego.driver.idm.desired_speed
        leader = context.snapshot.neighbors(ego, lane).leader
        achievable = free_speed
        if leader is not None:
            proximity = _clip_unit(1.0 - max(leader.gap, 0.0) / self.config.progress_horizon)
            achievable = free_speed + proximity * (min(leader.speed, free_speed) - free_speed)
        return _clip_unit((free_speed - achievable) / free_speed)

    def _safety(self, context: DecisionContext, lane: int) -> float:
        """Largest normalised shortfall against the desired gaps in ``lane``."""
        neighbors = context.snapshot.neighbors(context.ego, lane)
        return max(
            self._leader_shortfall(context.ego, neighbors),
            self._follower_shortfall(neighbors),
        )

    def _leader_shortfall(self, ego: Vehicle, neighbors: LaneNeighbors) -> float:
        """Shortfall against the desired gap to the leader."""
        if neighbors.leader is None:
            return 0.0
        desired = (
            self.config.desired_standstill_gap + self.config.desired_leader_headway * ego.speed
        )
        return _clip_unit((desired - neighbors.leader.gap) / desired)

    def _follower_shortfall(self, neighbors: LaneNeighbors) -> float:
        """Shortfall against the desired gap left to the follower."""
        follower = neighbors.follower
        if follower is None:
            return 0.0
        desired = (
            self.config.desired_standstill_gap
            + self.config.desired_follower_headway * follower.speed
        )
        return _clip_unit((desired - follower.gap) / desired)

    def _comfort(self, successor: BehaviorState) -> float:
        """Lateral effort charged to ``successor``."""
        if successor.is_preparing:
            return self.config.prepare_effort
        if successor.is_changing:
            return self.config.change_effort
        return 0.0

    def _lane_preference(self, context: DecisionContext, lane: int) -> float:
        """Normalised distance from the preferred lane."""
        span = max(context.road.lane_count - 1, 1)
        return _clip_unit(abs(lane - self.config.preferred_lane) / span)
