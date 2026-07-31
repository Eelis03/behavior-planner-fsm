"""The MOBIL lane change model of Kesting, Treiber and Helbing (2007).

MOBIL drives the traffic vehicles, not the ego. Without it the surrounding
traffic would hold its lane whatever happened around it, and the ego would be
planning against a straw problem: every gap it saw would still be there when it
arrived. With it the traffic competes for the same gaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from behavior_planner.algorithm.base import CarFollowingModel
from behavior_planner.model.traffic import Neighbor, TrafficSnapshot
from behavior_planner.model.vehicle import Vehicle

__all__ = ["MobilAssessment", "MobilLaneChangeModel"]


@dataclass(frozen=True, slots=True)
class MobilAssessment:
    """The full result of assessing one candidate lane."""

    target_lane: int
    safe: bool
    incentive: float
    """Left hand side of the incentive criterion, in metres per second squared."""

    threshold: float
    """Right hand side of the incentive criterion, including any bias."""

    @property
    def accepted(self) -> bool:
        """True when the change is both safe and worth making."""
        return self.safe and self.incentive > self.threshold


@dataclass(frozen=True, slots=True)
class MobilLaneChangeModel:
    """Minimising Overall Braking Induced by Lane changes.

    A change from the current lane to a candidate lane is accepted when both

    * the safety criterion holds, the new follower's acceleration after the
      change is not below ``-b_safe``, and
    * the incentive criterion holds,

      ``a_c' - a_c + p * ((a_n' - a_n) + (a_o' - a_o)) > delta_a_th``

      where ``c`` is the vehicle considering the change, ``n`` its prospective
      new follower and ``o`` its current follower, and a prime marks the value
      after the change.

    The politeness factor ``p`` is what makes the model interesting: at ``p = 0``
    drivers ignore the cost they impose, at ``p = 1`` they weigh it equally with
    their own gain. The keep-right bias of the asymmetric variant is available
    through :attr:`MobilParams.right_bias` and is zero by default, which selects
    the symmetric rule.
    """

    car_following: CarFollowingModel

    def choose_lane(self, subject: Vehicle, snapshot: TrafficSnapshot) -> int | None:
        """Lane the subject should move into, or ``None`` to stay put.

        A vehicle already crossing a boundary is never given a second manoeuvre.
        When both neighbouring lanes are acceptable the larger incentive wins,
        and ties break towards the lower lane index so the result does not
        depend on iteration order.
        """
        if subject.is_changing_lane:
            return None
        road = snapshot.road
        current = subject.assigned_lane(road)
        best: MobilAssessment | None = None
        for target in (current - 1, current + 1):
            if not road.contains_lane(target):
                continue
            assessment = self.assess(subject, snapshot, target)
            if not assessment.accepted:
                continue
            gain = assessment.incentive - assessment.threshold
            if best is None or gain > best.incentive - best.threshold:
                best = assessment
        return None if best is None else best.target_lane

    def assess(
        self, subject: Vehicle, snapshot: TrafficSnapshot, target_lane: int
    ) -> MobilAssessment:
        """Evaluate the safety and incentive criteria for one candidate lane."""
        road = snapshot.road
        current = subject.assigned_lane(road)
        params = subject.driver.mobil
        here = snapshot.neighbors(subject, current)
        there = snapshot.neighbors(subject, target_lane)

        subject_before = self._follow(subject, here.leader)
        subject_after = self._follow(subject, there.leader)

        new_follower_before = self._follow_leader_of(there.follower, snapshot, target_lane)
        new_follower_after = self._follow_behind(there.follower, subject, snapshot)
        old_follower_before = self._follow_leader_of(here.follower, snapshot, current)
        old_follower_after = self._follow_skipping(here.follower, here.leader, snapshot)

        safe = new_follower_after >= -params.safe_deceleration

        incentive = (subject_after - subject_before) + params.politeness * (
            (new_follower_after - new_follower_before) + (old_follower_after - old_follower_before)
        )
        bias = params.right_bias if target_lane < current else -params.right_bias
        threshold = params.changing_threshold - bias
        return MobilAssessment(
            target_lane=target_lane, safe=safe, incentive=incentive, threshold=threshold
        )

    def _follow(self, subject: Vehicle, leader: Neighbor | None) -> float:
        """Acceleration of ``subject`` behind ``leader``, free flow if absent."""
        if leader is None:
            return self.car_following.acceleration(
                speed=subject.speed,
                gap=math.inf,
                leader_speed=0.0,
                driver=subject.driver,
            )
        return self.car_following.acceleration(
            speed=subject.speed,
            gap=leader.gap,
            leader_speed=leader.speed,
            driver=subject.driver,
        )

    def _follow_leader_of(
        self, follower: Neighbor | None, snapshot: TrafficSnapshot, lane: int
    ) -> float:
        """Acceleration of ``follower`` behind whatever currently leads it."""
        if follower is None:
            return 0.0
        ahead = snapshot.neighbors(follower.vehicle, lane).leader
        return self._follow(follower.vehicle, ahead)

    def _follow_behind(
        self, follower: Neighbor | None, subject: Vehicle, snapshot: TrafficSnapshot
    ) -> float:
        """Acceleration of ``follower`` once ``subject`` is in front of it."""
        if follower is None:
            return 0.0
        road = snapshot.road
        ahead = road.forward_distance(follower.vehicle.s, subject.s)
        gap = ahead - 0.5 * (follower.vehicle.shape.length + subject.shape.length)
        return self.car_following.acceleration(
            speed=follower.vehicle.speed,
            gap=gap,
            leader_speed=subject.speed,
            driver=follower.vehicle.driver,
        )

    def _follow_skipping(
        self,
        follower: Neighbor | None,
        new_leader: Neighbor | None,
        snapshot: TrafficSnapshot,
    ) -> float:
        """Acceleration of ``follower`` once the subject has left its lane."""
        if follower is None:
            return 0.0
        if new_leader is None:
            return self._follow(follower.vehicle, None)
        road = snapshot.road
        ahead = road.forward_distance(follower.vehicle.s, new_leader.vehicle.s)
        gap = ahead - 0.5 * (follower.vehicle.shape.length + new_leader.vehicle.shape.length)
        return self.car_following.acceleration(
            speed=follower.vehicle.speed,
            gap=gap,
            leader_speed=new_leader.speed,
            driver=follower.vehicle.driver,
        )
