"""Decision algorithms: driver models, the state machine, cost, gate, trajectory.

Nothing in this layer plots, prints or reads a file. Every component is reached
through a Protocol declared in :mod:`behavior_planner.algorithm.base`.
"""

from __future__ import annotations

from behavior_planner.algorithm.base import (
    BehaviorPolicy,
    CarFollowingModel,
    CostModel,
    LaneChangeModel,
    SafetyGate,
)
from behavior_planner.algorithm.cost import WeightedCostModel, target_lane_for
from behavior_planner.algorithm.fsm import (
    REJECTED,
    TRANSITIONS,
    IllegalTransitionError,
    event_for,
    is_legal,
    successors,
    transition,
)
from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.mobil import MobilAssessment, MobilLaneChangeModel
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner, KeepLaneBaseline
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.algorithm.trajectory import (
    TrajectoryGenerator,
    TrajectoryPoint,
    binding_lanes,
    lane_change_for,
    longitudinal_acceleration,
)

__all__ = [
    "REJECTED",
    "TRANSITIONS",
    "BehaviorPolicy",
    "CarFollowingModel",
    "CostModel",
    "FiniteStateBehaviorPlanner",
    "GapAndDecelerationGate",
    "IllegalTransitionError",
    "IntelligentDriverModel",
    "KeepLaneBaseline",
    "LaneChangeModel",
    "MobilAssessment",
    "MobilLaneChangeModel",
    "SafetyGate",
    "TrajectoryGenerator",
    "TrajectoryPoint",
    "WeightedCostModel",
    "binding_lanes",
    "event_for",
    "is_legal",
    "lane_change_for",
    "longitudinal_acceleration",
    "successors",
    "target_lane_for",
    "transition",
]
