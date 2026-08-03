"""Metrics computed from run traces.

The collision count is the only metric with a required value. Everything else
describes how the planner spent the freedom the gate left it, and is only
meaningful once the collision count is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from behavior_planner.pipeline.suite import DensityGroup
from behavior_planner.pipeline.trace import RunTrace

__all__ = [
    "DensityMetrics",
    "ScenarioMetrics",
    "SuiteMetrics",
    "SweepMetrics",
    "scenario_metrics",
    "suite_metrics",
    "sweep_metrics",
]


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    """What one run of one scenario produced."""

    scenario: str
    policy: str
    seed: int
    duration: float
    collisions: int
    """Distinct pairs of vehicles that overlapped, the ego included."""

    ego_collisions: int
    """Distinct vehicles the ego itself overlapped, a subset of the above."""

    mean_speed: float
    """Mean ego speed over the run, in metres per second."""

    minimum_speed: float
    """Slowest the ego travelled at any step, in metres per second.

    A mean hides a stop. On a ring dense enough for the car following model to
    produce stop and go waves, the difference between a run that averaged
    20 m/s smoothly and one that alternated between 30 and 0 is the whole story.
    """

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


@dataclass(frozen=True, slots=True)
class DensityMetrics:
    """The distribution of outcomes over every seed run at one density.

    A single run reports a number. A set of runs at one density reports a
    spread, and the spread is the part that says whether the number was typical
    or lucky.
    """

    density: int
    runs: tuple[ScenarioMetrics, ...]

    def __post_init__(self) -> None:
        """Reject a density with no runs behind it."""
        if not self.runs:
            raise ValueError(f"density {self.density} produced no runs")

    @property
    def count(self) -> int:
        """Number of seeds run at this density."""
        return len(self.runs)

    @property
    def collisions(self) -> int:
        """Overlapping vehicle pairs summed over every seed, the ego included."""
        return sum(item.collisions for item in self.runs)

    @property
    def ego_collisions(self) -> int:
        """Overlaps involving the ego, summed over every seed. Required to be zero."""
        return sum(item.ego_collisions for item in self.runs)

    @property
    def lane_changes(self) -> int:
        """Completed ego lane changes summed over every seed."""
        return sum(item.lane_changes for item in self.runs)

    @property
    def speed_p05(self) -> float:
        """Fifth percentile of the per-run mean ego speed, in metres per second."""
        return self._speed_percentile(5.0)

    @property
    def speed_median(self) -> float:
        """Median of the per-run mean ego speed, in metres per second."""
        return self._speed_percentile(50.0)

    @property
    def speed_p95(self) -> float:
        """Ninety-fifth percentile of the per-run mean ego speed."""
        return self._speed_percentile(95.0)

    @property
    def minimum_speed(self) -> float:
        """Slowest the ego travelled at any step of any seed, in metres per second."""
        return min(item.minimum_speed for item in self.runs)

    @property
    def minimum_time_headway(self) -> float:
        """Smallest time headway seen at this density, over every seed."""
        return min(item.minimum_time_headway for item in self.runs)

    @property
    def minimum_time_to_collision(self) -> float:
        """Smallest time to collision seen at this density, over every seed."""
        return min(item.minimum_time_to_collision for item in self.runs)

    def _speed_percentile(self, percentile: float) -> float:
        values = np.array([item.mean_speed for item in self.runs], dtype=np.float64)
        return float(np.percentile(values, percentile))


@dataclass(frozen=True, slots=True)
class SweepMetrics:
    """Every density of one sweep, under one policy."""

    policy: str
    densities: tuple[DensityMetrics, ...]

    def __post_init__(self) -> None:
        """Reject an empty sweep, which has no aggregate."""
        if not self.densities:
            raise ValueError("a sweep must contain at least one density")

    @property
    def total_runs(self) -> int:
        """Number of runs across every density."""
        return sum(item.count for item in self.densities)

    @property
    def total_collisions(self) -> int:
        """Overlapping vehicle pairs summed over the whole sweep."""
        return sum(item.collisions for item in self.densities)

    @property
    def total_ego_collisions(self) -> int:
        """Overlaps involving the ego, over the whole sweep. Required to be zero."""
        return sum(item.ego_collisions for item in self.densities)

    @property
    def minimum_speed(self) -> float:
        """Slowest the ego travelled anywhere in the sweep, in metres per second."""
        return min(item.minimum_speed for item in self.densities)

    @property
    def minimum_time_headway(self) -> float:
        """Smallest time headway anywhere in the sweep."""
        return min(item.minimum_time_headway for item in self.densities)

    @property
    def minimum_time_to_collision(self) -> float:
        """Smallest time to collision anywhere in the sweep."""
        return min(item.minimum_time_to_collision for item in self.densities)

    @property
    def slowest_run(self) -> ScenarioMetrics:
        """The run with the lowest mean ego speed, named so it can be reproduced."""
        return min(
            (item for density in self.densities for item in density.runs),
            key=lambda item: item.mean_speed,
        )


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
        ego_collisions=trace.ego_collision_count,
        mean_speed=float(speeds.mean()),
        minimum_speed=float(speeds.min()),
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


def sweep_metrics(groups: tuple[DensityGroup, ...]) -> SweepMetrics:
    """Reduce a density sweep to a distribution of outcomes per density."""
    if not groups:
        raise ValueError("a sweep must contain at least one density")
    policies = {trace.policy for group in groups for trace in group.traces}
    if len(policies) != 1:
        raise ValueError(f"a sweep must use one policy, got {sorted(policies)}")
    return SweepMetrics(
        policy=groups[0].traces[0].policy,
        densities=tuple(
            DensityMetrics(
                density=group.density,
                runs=tuple(scenario_metrics(trace) for trace in group.traces),
            )
            for group in groups
        ),
    )


def _format_seconds(value: float) -> str:
    """Render a duration, writing an unbounded one as ``inf``."""
    return "inf" if math.isinf(value) else f"{value:.2f}"
