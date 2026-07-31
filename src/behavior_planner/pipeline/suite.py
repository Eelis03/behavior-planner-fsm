"""Wiring that assembles the default planner and runs a scenario suite.

This is the one place where the concrete implementations are chosen. Every
layer below is written against the Protocols in
:mod:`behavior_planner.algorithm.base`, so substituting a policy, a car
following model or a gate is a change to this module and to nothing else.
"""

from __future__ import annotations

from behavior_planner.algorithm.base import BehaviorPolicy
from behavior_planner.algorithm.cost import WeightedCostModel
from behavior_planner.algorithm.idm import IntelligentDriverModel
from behavior_planner.algorithm.planner import FiniteStateBehaviorPlanner, KeepLaneBaseline
from behavior_planner.algorithm.safety import GapAndDecelerationGate
from behavior_planner.model.config import PlannerConfig
from behavior_planner.pipeline.scenarios import Scenario, standard_suite
from behavior_planner.pipeline.simulator import SimulationConfig, TrafficSimulator
from behavior_planner.pipeline.trace import RunTrace

__all__ = ["build_planner", "build_simulator", "run_baseline", "run_scenario", "run_suite"]


def build_planner(config: PlannerConfig) -> FiniteStateBehaviorPlanner:
    """The default behaviour policy at ``config``."""
    return FiniteStateBehaviorPlanner(
        cost=WeightedCostModel(config.cost),
        gate=GapAndDecelerationGate(
            limits=config.safety, car_following=IntelligentDriverModel()
        ),
    )


def build_simulator(
    *,
    policy: BehaviorPolicy,
    policy_name: str,
    config: PlannerConfig,
    simulation: SimulationConfig,
) -> TrafficSimulator:
    """A simulator wired to ``policy``."""
    return TrafficSimulator(
        policy=policy,
        planner_config=config,
        simulation=simulation,
        car_following=IntelligentDriverModel(),
        policy_name=policy_name,
    )


def run_scenario(
    scenario: Scenario,
    *,
    config: PlannerConfig | None = None,
    simulation: SimulationConfig | None = None,
    steps: int | None = None,
) -> RunTrace:
    """Run one scenario under the default planner."""
    planner_config = config or PlannerConfig()
    simulator = build_simulator(
        policy=build_planner(planner_config),
        policy_name="fsm",
        config=planner_config,
        simulation=simulation or SimulationConfig(),
    )
    return simulator.run(scenario, steps=steps)


def run_baseline(
    scenario: Scenario,
    *,
    config: PlannerConfig | None = None,
    simulation: SimulationConfig | None = None,
    steps: int | None = None,
) -> RunTrace:
    """Run one scenario under the lane keeping control policy."""
    planner_config = config or PlannerConfig()
    simulator = build_simulator(
        policy=KeepLaneBaseline(),
        policy_name="keep_lane",
        config=planner_config,
        simulation=simulation or SimulationConfig(),
    )
    return simulator.run(scenario, steps=steps)


def run_suite(
    scenarios: tuple[Scenario, ...] | None = None,
    *,
    config: PlannerConfig | None = None,
    simulation: SimulationConfig | None = None,
    steps: int | None = None,
    baseline: bool = False,
) -> tuple[RunTrace, ...]:
    """Run every scenario, under the planner or under the control policy."""
    chosen = standard_suite() if scenarios is None else scenarios
    runner = run_baseline if baseline else run_scenario
    return tuple(
        runner(scenario, config=config, simulation=simulation, steps=steps)
        for scenario in chosen
    )
