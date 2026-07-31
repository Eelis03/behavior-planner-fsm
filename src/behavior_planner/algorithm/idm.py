"""The Intelligent Driver Model of Treiber, Hennecke and Helbing (2000)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from behavior_planner.model.config import DriverParams, IdmParams

__all__ = ["IntelligentDriverModel"]

# Smallest gap the model is evaluated at. The interaction term diverges as the
# gap goes to zero, so a floor is needed to keep the acceleration finite once
# two bodies have already overlapped. It is well below any gap the simulation
# reaches while it is behaving correctly.
_MINIMUM_GAP: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class IntelligentDriverModel:
    """Car following acceleration as a function of speed, gap and closing rate.

    The acceleration is

    ``a = a_max * (1 - (v / v0)^delta - (s_star / s)^2)``

    with the dynamic desired gap

    ``s_star = s0 + max(0, v * T + v * dv / (2 * sqrt(a_max * b)))``

    where ``dv = v - v_lead`` is the closing rate. The first two terms give free
    flow behaviour, which drives the speed to ``v0`` from either side, and the
    third is the interaction with the leader, which is the only term that can
    make the acceleration negative.

    The model is collision free for a leader that does not brake harder than
    ``b``, which is why the traffic in this simulation never rear-ends anything
    on its own. Cut-ins are a different matter and are handled by MOBIL's safety
    criterion and, for the ego, by the safety gate.
    """

    def acceleration(
        self,
        *,
        speed: float,
        gap: float,
        leader_speed: float,
        driver: DriverParams,
    ) -> float:
        """Acceleration in metres per second squared."""
        params = driver.idm
        free = self.free_acceleration(speed=speed, driver=driver)
        if math.isinf(gap):
            return free
        effective_gap = max(gap, _MINIMUM_GAP)
        desired = self.desired_gap(speed=speed, leader_speed=leader_speed, params=params)
        interaction = params.maximum_acceleration * (desired / effective_gap) ** 2
        return free - interaction

    def free_acceleration(self, *, speed: float, driver: DriverParams) -> float:
        """Acceleration with no leader in range."""
        params = driver.idm
        ratio = max(speed, 0.0) / params.desired_speed
        shortfall = 1.0 - float(ratio**params.acceleration_exponent)
        return params.maximum_acceleration * shortfall

    def desired_gap(self, *, speed: float, leader_speed: float, params: IdmParams) -> float:
        """The dynamic desired gap ``s_star``, in metres."""
        closing = speed - leader_speed
        braking_scale = 2.0 * math.sqrt(
            params.maximum_acceleration * params.comfortable_deceleration
        )
        dynamic = speed * params.time_headway + speed * closing / braking_scale
        return params.minimum_gap + max(0.0, dynamic)

    def equilibrium_gap(self, *, speed: float, params: IdmParams) -> float:
        """Steady state gap behind a leader travelling at ``speed``.

        Setting the acceleration to zero with zero closing rate gives

        ``s_eq = (s0 + v * T) / sqrt(1 - (v / v0)^delta)``

        which is a closed-form quantity and is used as such in the tests. It is
        finite only below the desired speed; at or above it no finite gap makes
        the acceleration vanish, because the driver is already being slowed by
        the free flow term alone.
        """
        ratio = speed / params.desired_speed
        slack = 1.0 - ratio**params.acceleration_exponent
        if slack <= 0.0:
            return math.inf
        return (params.minimum_gap + speed * params.time_headway) / math.sqrt(slack)
