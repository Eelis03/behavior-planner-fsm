"""Rendering of metrics as the Markdown tables the README reports."""

from __future__ import annotations

from typing import Final

import numpy as np

from behavior_planner.analysis.metrics import (
    DensityMetrics,
    ScenarioMetrics,
    SuiteMetrics,
    SweepMetrics,
)

__all__ = ["comparison_table", "scenario_table", "sweep_table", "worst_paired_gain"]

_HEADERS: Final[tuple[str, ...]] = (
    "Scenario",
    "Collisions",
    "Mean speed (m/s)",
    "Distance (m)",
    "Lane changes",
    "Min headway (s)",
    "Min TTC (s)",
    "TTC p05 (s)",
    "TTC median (s)",
)


def _render(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    """Render a Markdown table with a left-aligned first column."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join(lines)


def scenario_table(metrics: SuiteMetrics) -> str:
    """One row per scenario, in suite order."""
    rows = tuple(item.as_row() for item in metrics.scenarios)
    return _render(_HEADERS, rows)


def comparison_table(planned: SuiteMetrics, baseline: SuiteMetrics) -> str:
    """Ego speed and lane changes under the planner beside the lane keeping control."""
    headers = (
        "Scenario",
        "Planner speed (m/s)",
        "Baseline speed (m/s)",
        "Gain (percent)",
        "Lane changes",
    )
    index = {item.scenario: item for item in baseline.scenarios}
    rows = tuple(_comparison_row(item, index[item.scenario]) for item in planned.scenarios)
    return _render(headers, rows)


def sweep_table(planned: SweepMetrics, baseline: SweepMetrics) -> str:
    """One row per density, reporting a distribution rather than a single run.

    The gain columns are computed from the per-seed gains, not from the gain
    between the two medians. The runs are paired by seed, so the paired
    statistic is available and is the one that answers whether the planner
    helped on the run the reader would have made. Both the median and the mean
    are reported because they disagree, and the disagreement is the finding: the
    benefit is concentrated in the minority of runs where a gap existed.
    """
    headers = (
        "Vehicles per lane",
        "Runs",
        "Collisions",
        "Speed p05 (m/s)",
        "Speed median (m/s)",
        "Speed p95 (m/s)",
        "Median gain (percent)",
        "Mean gain (percent)",
        "Lane changes",
    )
    index = {item.density: item for item in baseline.densities}
    rows = tuple(_sweep_row(item, index[item.density]) for item in planned.densities)
    return _render(headers, rows)


def worst_paired_gain(planned: SweepMetrics, baseline: SweepMetrics) -> tuple[str, float]:
    """The seed on which the planner did worst against the control, and by how much.

    A distribution summarised only by its middle would let the worst case go
    unreported, and the worst case is the one a reader is entitled to see named
    so they can reproduce it.
    """
    control = {
        item.scenario: item.mean_speed
        for density in baseline.densities
        for item in density.runs
    }
    gains = [
        (100.0 * (item.mean_speed - control[item.scenario]) / control[item.scenario], item.scenario)
        for density in planned.densities
        for item in density.runs
    ]
    gain, scenario = min(gains)
    return scenario, gain


def _sweep_row(planned: DensityMetrics, baseline: DensityMetrics) -> tuple[str, ...]:
    """One row of the sweep table, with the gain paired by seed."""
    control = {item.scenario: item.mean_speed for item in baseline.runs}
    gains = [
        100.0 * (item.mean_speed - control[item.scenario]) / control[item.scenario]
        for item in planned.runs
    ]
    return (
        str(planned.density),
        str(planned.count),
        str(planned.collisions),
        f"{planned.speed_p05:.2f}",
        f"{planned.speed_median:.2f}",
        f"{planned.speed_p95:.2f}",
        f"{float(np.median(gains)):+.1f}",
        f"{float(np.mean(gains)):+.1f}",
        str(planned.lane_changes),
    )


def _comparison_row(planned: ScenarioMetrics, baseline: ScenarioMetrics) -> tuple[str, ...]:
    """One row of the comparison table."""
    gain = 100.0 * (planned.mean_speed - baseline.mean_speed) / baseline.mean_speed
    return (
        planned.scenario,
        f"{planned.mean_speed:.2f}",
        f"{baseline.mean_speed:.2f}",
        f"{gain:+.1f}",
        str(planned.lane_changes),
    )
