"""Rendering of metrics as the Markdown tables the README reports."""

from __future__ import annotations

from typing import Final

from behavior_planner.analysis.metrics import ScenarioMetrics, SuiteMetrics

__all__ = ["comparison_table", "scenario_table"]

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
