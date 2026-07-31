"""Finite state machine and cost-based lane change decisions on a highway traffic simulation.

The package is layered. :mod:`behavior_planner.model` holds pure data,
:mod:`behavior_planner.algorithm` the decision logic behind Protocols,
:mod:`behavior_planner.pipeline` the simulator and the scenario suite, and
:mod:`behavior_planner.analysis` the metrics and figures. Each layer imports
only from the ones above it.
"""

from __future__ import annotations

from behavior_planner.algorithm.cost import WeightedCostModel
from behavior_planner.algorithm.fsm import (
    REJECTED,
    TRANSITIONS,
    IllegalTransitionError,
    successors,
    transition,
)
from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.mobil import MobilLaneChangeModel
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner, KeepLaneBaseline
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.algorithm.trajectory import TrajectoryGenerator
from behavior_planner.analysis.metrics import scenario_metrics, suite_metrics
from behavior_planner.model.config import (
    CostConfig,
    CostWeights,
    DriverParams,
    IdmParams,
    MobilParams,
    PlannerConfig,
    SafetyLimits,
)
from behavior_planner.model.road import Road
from behavior_planner.model.states import BehaviorEvent, BehaviorState
from behavior_planner.model.vehicle import Vehicle
from behavior_planner.pipeline.scenarios import Scenario, VehicleSpec, standard_suite
from behavior_planner.pipeline.simulator import SimulationConfig, TrafficSimulator
from behavior_planner.pipeline.suite import run_baseline, run_scenario, run_suite

__all__ = [
    "REJECTED",
    "TRANSITIONS",
    "BehaviorEvent",
    "BehaviorState",
    "CostConfig",
    "CostWeights",
    "DriverParams",
    "FiniteStateBehaviorPlanner",
    "GapAndDecelerationGate",
    "IdmParams",
    "IllegalTransitionError",
    "IntelligentDriverModel",
    "KeepLaneBaseline",
    "MobilLaneChangeModel",
    "MobilParams",
    "PlannerConfig",
    "Road",
    "SafetyLimits",
    "Scenario",
    "SimulationConfig",
    "TrafficSimulator",
    "TrajectoryGenerator",
    "Vehicle",
    "VehicleSpec",
    "WeightedCostModel",
    "__version__",
    "run_baseline",
    "run_scenario",
    "run_suite",
    "scenario_metrics",
    "standard_suite",
    "successors",
    "suite_metrics",
    "transition",
]

__version__ = "0.1.0"
