"""Metrics computed from run traces.

The collision count is the only metric with a required value. Everything else
describes how the planner spent the freedom the gate left it, and is only
meaningful once the collision count is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from behavior_planner.pipeline.trace import RunTrace

__all__ = ["ScenarioMetrics", "SuiteMetrics", "scenario_metrics", "suite_metrics"]


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    """What one run of one scenario produced."""

    scenario: str
    policy: str
    seed: int
    duration: float
    collisions: int
    mean_speed: float
    """Mean ego speed over the run, in metres per second."""

    distance: float
    """Arc length covered by the ego, in metres."""

    lane_changes: int
    aborted_changes: int
    minimum_time_headway: float
    """Smallest gap-over-speed observed, in seconds."""

    minimum_time_to_collision: float
    """Smallest time to collision with the ego's leader, in seconds."""

    ttc_p05: float
    """Fifth percentile of the finite times to collision, in seconds."""

    ttc_median: float
    """Median of the finite times to collision, in seconds."""

    ttc_samples: int
    """Number of steps at which the ego was closing on a leader."""

    def as_row(self) -> tuple[str, ...]:
        """The metric values as strings, in reporting order."""
        return (
            self.scenario,
            str(self.collisions),
            f"{self.mean_speed:.2f}",
            f"{self.distance:.0f}",
            str(self.lane_changes),
            _format_seconds(self.minimum_time_headway),
            _format_seconds(self.minimum_time_to_collision),
            _format_seconds(self.ttc_p05),
            _format_seconds(self.ttc_median),
        )


@dataclass(frozen=True, slots=True)
class SuiteMetrics:
    """Aggregate over every scenario in a suite."""

    policy: str
    scenarios: tuple[ScenarioMetrics, ...]

    def __post_init__(self) -> None:
        """Reject an empty suite, which has no aggregate."""
        if not self.scenarios:
            raise ValueError("a suite must contain at least one scenario")

    @property
    def total_collisions(self) -> int:
        """Collisions summed over every scenario. Required to be zero."""
        return sum(item.collisions for item in self.scenarios)

    @property
    def total_lane_changes(self) -> int:
        """Completed ego lane changes summed over every scenario."""
        return sum(item.lane_changes for item in self.scenarios)

    @property
    def mean_speed(self) -> float:
        """Mean ego speed across scenarios, weighted by run duration."""
        weights = [item.duration for item in self.scenarios]
        values = [item.mean_speed for item in self.scenarios]
        return float(np.average(values, weights=weights))

    @property
    def minimum_time_headway(self) -> float:
        """Smallest time headway seen anywhere in the suite."""
        return min(item.minimum_time_headway for item in self.scenarios)

    @property
    def minimum_time_to_collision(self) -> float:
        """Smallest time to collision seen anywhere in the suite."""
        return min(item.minimum_time_to_collision for item in self.scenarios)


def scenario_metrics(trace: RunTrace) -> ScenarioMetrics:
    """Reduce one run trace to its metrics."""
    speeds = np.array([record.speed for record in trace.records], dtype=np.float64)
    ttc = np.array(trace.finite_times_to_collision(), dtype=np.float64)
    return ScenarioMetrics(
        scenario=trace.scenario,
        policy=trace.policy,
        seed=trace.seed,
        duration=trace.duration,
        collisions=trace.collision_count,
        mean_speed=float(speeds.mean()),
        distance=trace.distance_travelled,
        lane_changes=trace.lane_changes,
        aborted_changes=trace.aborted_changes,
        minimum_time_headway=trace.minimum_time_headway,
        minimum_time_to_collision=float(ttc.min()) if ttc.size else math.inf,
        ttc_p05=float(np.percentile(ttc, 5.0)) if ttc.size else math.inf,
        ttc_median=float(np.median(ttc)) if ttc.size else math.inf,
        ttc_samples=int(ttc.size),
    )


def suite_metrics(traces: tuple[RunTrace, ...]) -> SuiteMetrics:
    """Reduce a suite of run traces to per-scenario metrics and the aggregate."""
    if not traces:
        raise ValueError("a suite must contain at least one trace")
    policies = {trace.policy for trace in traces}
    if len(policies) != 1:
        raise ValueError(f"a suite must use one policy, got {sorted(policies)}")
    return SuiteMetrics(
        policy=traces[0].policy,
        scenarios=tuple(scenario_metrics(trace) for trace in traces),
    )


def _format_seconds(value: float) -> str:
    """Render a duration, writing an unbounded one as ``inf``."""
    return "inf" if math.isinf(value) else f"{value:.2f}"
