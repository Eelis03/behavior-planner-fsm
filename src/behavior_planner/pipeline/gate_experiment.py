"""The experiment behind the central claim: a veto is not a large penalty.

The scene below is the conflict the safety gate exists to resolve. The ego is
held up by a slow leader, so the progress term prefers the lane beside it; the
lane beside it is occupied by a vehicle sitting just behind the ego, which
lowers no cost term and leaves nowhere to go; and the soft safety weight is zero,
which is the misconfiguration a separate gate has to survive.

Sweeping the progress weight across three orders of magnitude turns that scene
into a measurement. If safety were a cost term, some weight would buy the
manoeuvre. It is not, so none does.

The scene is defined here rather than only inside a test because the same three
consumers need it: the test that asserts the property, the figure that shows it,
and the example script that prints the numbers the README quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from behavior_planner.algorithm.cost import WeightedCostModel
from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.model.config import (
    CostConfig,
    CostWeights,
    DriverParams,
    IdmParams,
    PlannerConfig,
)
from behavior_planner.model.decision import DecisionContext, VetoReason
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorState
from behavior_planner.model.traffic import TrafficSnapshot
from behavior_planner.model.vehicle import EGO_ID, Vehicle, VehicleShape

__all__ = [
    "PROGRESS_WEIGHTS",
    "WeightSample",
    "blocked_merge_context",
    "gate_weight_sweep",
]

PROGRESS_WEIGHTS: Final[tuple[float, ...]] = (
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
)
"""Progress weights swept, from the default of 5 to a thousand times it.

Ten points rather than two, because the claim is that the veto is unaffected by
the weight rather than that it survives one large value.
"""

_CANDIDATE: Final[BehaviorState] = BehaviorState.LANE_CHANGE_LEFT
_ROAD: Final[Road] = Road(lane_count=3, length=1200.0)


@dataclass(frozen=True, slots=True)
class WeightSample:
    """What the cost function, the gate and the planner each said at one weight."""

    progress_weight: float
    keep_lane_cost: float
    lane_change_cost: float
    allowed: bool
    """Whether the safety gate permitted the lane change."""

    reason: VetoReason
    chosen: BehaviorState
    """The state the planner actually selected."""

    @property
    def cost_prefers_change(self) -> bool:
        """True when the cost function ranks the lane change ahead of keeping lane."""
        return self.lane_change_cost < self.keep_lane_cost

    @property
    def cost_margin(self) -> float:
        """How much cheaper the lane change is than keeping lane, in cost units.

        Positive means the cost function wants the manoeuvre. The size of this
        number is what the sweep drives across three orders of magnitude while
        the verdict does not move.
        """
        return self.keep_lane_cost - self.lane_change_cost


def _vehicle(vehicle_id: int, lane: int, s: float, speed: float, desired_speed: float) -> Vehicle:
    """A vehicle centred in ``lane`` at ``s`` on the experiment's road."""
    return Vehicle(
        vehicle_id=vehicle_id,
        s=_ROAD.wrap(s),
        d=_ROAD.lane_center(lane),
        speed=speed,
        shape=VehicleShape(),
        driver=DriverParams(idm=IdmParams(desired_speed=desired_speed)),
    )


def blocked_merge_context(config: PlannerConfig) -> DecisionContext:
    """The scene: a slow leader ahead, and the only escape already occupied.

    The blocker is 25 m ahead at 15 m/s, which is what makes the progress term
    prefer the next lane. The vehicle beside is 1.5 m behind the ego and one
    lane left, which is inside the ego's own body length, so the space the
    manoeuvre would move into is not merely tight but taken.
    """
    ego = _vehicle(EGO_ID, lane=0, s=0.0, speed=18.0, desired_speed=31.0)
    blocker = _vehicle(1, lane=0, s=25.0, speed=15.0, desired_speed=15.0)
    beside = _vehicle(2, lane=1, s=-1.5, speed=26.0, desired_speed=31.0)
    return DecisionContext(
        road=_ROAD,
        snapshot=TrafficSnapshot(road=_ROAD, vehicles=(ego, blocker, beside)),
        ego=ego,
        state=BehaviorState.PREPARE_LANE_CHANGE_LEFT,
        config=config,
    )


def gate_weight_sweep(
    weights: tuple[float, ...] = PROGRESS_WEIGHTS,
    *,
    safety_weight: float = 0.0,
) -> tuple[WeightSample, ...]:
    """Score the scene at each progress weight and record all three answers.

    ``safety_weight`` defaults to zero on purpose. The soft safety term is what
    would otherwise mask the result by making the lane change expensive for a
    reason that has nothing to do with the gate.
    """
    if not weights:
        raise ValueError("the sweep needs at least one progress weight")

    samples: list[WeightSample] = []
    for progress in weights:
        config = PlannerConfig(
            cost=CostConfig(
                weights=CostWeights(
                    progress=progress,
                    safety=safety_weight,
                    comfort=CostWeights().comfort,
                    lane_preference=CostWeights().lane_preference,
                )
            )
        )
        cost = WeightedCostModel(config.cost)
        gate = GapAndDecelerationGate(limits=config.safety, car_following=IntelligentDriverModel())
        planner = FiniteStateBehaviorPlanner(cost=cost, gate=gate)
        context = blocked_merge_context(config)

        weighted = config.cost.weights
        verdict = gate.review(context, _CANDIDATE)
        samples.append(
            WeightSample(
                progress_weight=progress,
                keep_lane_cost=cost.evaluate(context, BehaviorState.KEEP_LANE).weighted_total(
                    weighted
                ),
                lane_change_cost=cost.evaluate(context, _CANDIDATE).weighted_total(weighted),
                allowed=verdict.allowed,
                reason=verdict.reason,
                chosen=planner.decide(context).state,
            )
        )
    return tuple(samples)
