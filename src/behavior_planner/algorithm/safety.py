"""The safety gate: a veto over manoeuvres, structurally separate from the cost.

The gate answers one question, may this manoeuvre be attempted at all, and it
answers it without seeing a single cost. It reads its own thresholds from
:class:`SafetyLimits`, not from :class:`CostConfig`, and it returns a verdict,
not a penalty.

This separation is the point of the layer. A cost function that also carried
safety would express safety as a large number, and a large number can always be
outvoted by a larger one: enough speed advantage, or a slightly mistuned weight,
and the planner trades a gap it should not have traded. Reviewing a weight
vector for that failure means reasoning about every situation the weights might
meet. Reviewing a veto means reading the conditions listed below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from behavior_planner.algorithm.base import CarFollowingModel
from behavior_planner.model.config import SafetyLimits
from behavior_planner.model.decision import DecisionContext, SafetyVerdict, VetoReason
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import Neighbor, time_to_collision
from behavior_planner.model.vehicle import Vehicle

__all__ = ["GapAndDecelerationGate"]


@dataclass(frozen=True, slots=True)
class GapAndDecelerationGate:
    """Vetoes a lane change that fails any one of five hard conditions.

    1. The target lane exists.
    2. No vehicle already overlaps the ego longitudinally in the target lane.
    3. The gap to the target lane leader is at least
       ``minimum_leader_gap + leader_headway * v_ego``, and the time to
       collision with it is at least ``minimum_time_to_collision``.
    4. The gap to the target lane follower is at least
       ``minimum_follower_gap + follower_headway * v_follower``, and the time to
       collision with it is at least ``minimum_time_to_collision``.
    5. The deceleration the manoeuvre would force on that follower, evaluated
       with the same car following model the follower actually uses, is at most
       ``maximum_follower_deceleration``.

    Conditions 3 and 4 are geometric and hold whatever the speeds are doing.
    Condition 5 is behavioural and catches the merge that is geometrically legal
    but would make somebody stand on the brakes. It is MOBIL's safety criterion
    applied to the ego's own manoeuvre, which keeps the standard the ego is held
    to the same as the standard the traffic is held to.

    Prepare states are not gated. They involve no lateral motion, so there is
    nothing to veto; the gate acts at the commit, and again at every cycle while
    the change runs.
    """

    limits: SafetyLimits
    car_following: CarFollowingModel

    def review(self, context: DecisionContext, successor: BehaviorState) -> SafetyVerdict:
        """Whether ``successor`` may be entered."""
        if not successor.is_changing:
            return SafetyVerdict.allow(math.inf)

        road = context.road
        target_lane = context.target_lane(successor)
        if not road.contains_lane(target_lane):
            return SafetyVerdict.veto(VetoReason.OFF_ROAD, -1.0)

        ego = context.ego
        neighbors = context.snapshot.neighbors(ego, target_lane)
        margins: list[float] = []

        leader = neighbors.leader
        if leader is not None:
            if leader.gap <= 0.0:
                return SafetyVerdict.veto(VetoReason.OCCUPIED, leader.gap)
            required = self.limits.minimum_leader_gap + self.limits.leader_headway * ego.speed
            margin = (leader.gap - required) / required
            if margin < 0.0:
                return SafetyVerdict.veto(VetoReason.LEADER_GAP, margin)
            margins.append(margin)

            ttc = time_to_collision(leader.gap, ego.speed - leader.speed)
            ttc_margin = self._time_margin(ttc)
            if ttc_margin < 0.0:
                return SafetyVerdict.veto(VetoReason.LEADER_TIME_TO_COLLISION, ttc_margin)
            margins.append(ttc_margin)

        follower = neighbors.follower
        if follower is not None:
            if follower.gap <= 0.0:
                return SafetyVerdict.veto(VetoReason.OCCUPIED, follower.gap)
            required = (
                self.limits.minimum_follower_gap + self.limits.follower_headway * follower.speed
            )
            margin = (follower.gap - required) / required
            if margin < 0.0:
                return SafetyVerdict.veto(VetoReason.FOLLOWER_GAP, margin)
            margins.append(margin)

            ttc = time_to_collision(follower.gap, follower.speed - ego.speed)
            ttc_margin = self._time_margin(ttc)
            if ttc_margin < 0.0:
                return SafetyVerdict.veto(VetoReason.FOLLOWER_TIME_TO_COLLISION, ttc_margin)
            margins.append(ttc_margin)

            imposed = self._imposed_deceleration(follower, ego)
            allowed = self.limits.maximum_follower_deceleration
            decel_margin = (allowed - imposed) / allowed
            if decel_margin < 0.0:
                return SafetyVerdict.veto(VetoReason.FOLLOWER_DECELERATION, decel_margin)
            margins.append(decel_margin)

        return SafetyVerdict.allow(min(margins) if margins else math.inf)

    def _time_margin(self, ttc: float) -> float:
        """Normalised margin of a time to collision against the limit."""
        limit = self.limits.minimum_time_to_collision
        if math.isinf(ttc):
            return math.inf
        return (ttc - limit) / limit

    def _imposed_deceleration(self, follower: Neighbor, ego: Vehicle) -> float:
        """Braking the follower would have to apply with the ego in front of it."""
        acceleration = self.car_following.acceleration(
            speed=follower.vehicle.speed,
            gap=follower.gap,
            leader_speed=ego.speed,
            driver=follower.vehicle.driver,
        )
        return max(0.0, -acceleration)
