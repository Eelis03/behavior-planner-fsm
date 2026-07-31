"""Tier one. Metrics and the tables built from them."""

from __future__ import annotations

import math

import pytest

from behavior_planner.analysis.metrics import scenario_metrics, suite_metrics
from behavior_planner.analysis.report import comparison_table, scenario_table
from behavior_planner.model.traffic import time_headway, time_to_collision
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_suite

SHORT = tuple(scenario.with_duration(10.0) for scenario in standard_suite())


@pytest.fixture(scope="module")
def planned() -> tuple[object, ...]:
    """Short runs of every scenario under the planner."""
    return run_suite(SHORT)


@pytest.fixture(scope="module")
def baseline() -> tuple[object, ...]:
    """Short runs of every scenario under the control policy."""
    return run_suite(SHORT, baseline=True)


def test_time_to_collision_is_infinite_when_the_gap_is_opening() -> None:
    """Not closing means never meeting, which is infinity rather than a large number."""
    assert math.isinf(time_to_collision(50.0, -3.0))
    assert math.isinf(time_to_collision(50.0, 0.0))
    assert time_to_collision(50.0, 10.0) == pytest.approx(5.0)


def test_time_to_collision_is_zero_on_contact() -> None:
    """A non-positive gap has already run out of time; it does not go negative."""
    assert time_to_collision(0.0, 10.0) == pytest.approx(0.0)
    assert time_to_collision(-2.0, 10.0) == pytest.approx(0.0)


def test_time_headway_is_infinite_at_a_standstill() -> None:
    """A stopped vehicle is not about to close any gap."""
    assert math.isinf(time_headway(30.0, 0.0))
    assert time_headway(30.0, 15.0) == pytest.approx(2.0)
    assert time_headway(-1.0, 15.0) == pytest.approx(0.0)


def test_metrics_carry_the_scenario_identity(planned: tuple[object, ...]) -> None:
    """Every metric row names the scenario, the policy and the seed that produced it."""
    for trace, scenario in zip(planned, SHORT, strict=True):
        metrics = scenario_metrics(trace)  # type: ignore[arg-type]
        assert metrics.scenario == scenario.name
        assert metrics.policy == "fsm"
        assert metrics.seed == scenario.seed
        assert metrics.duration == pytest.approx(scenario.duration)


def test_the_suite_aggregate_is_a_duration_weighted_mean(
    planned: tuple[object, ...],
) -> None:
    """Every scenario in the suite runs for the same time, so the mean is the plain one."""
    metrics = suite_metrics(planned)  # type: ignore[arg-type]
    plain = sum(item.mean_speed for item in metrics.scenarios) / len(metrics.scenarios)
    assert metrics.mean_speed == pytest.approx(plain)


def test_the_suite_aggregate_reports_the_worst_case(planned: tuple[object, ...]) -> None:
    """The suite headway and time to collision are minima, not averages."""
    metrics = suite_metrics(planned)  # type: ignore[arg-type]
    assert metrics.minimum_time_headway == min(
        item.minimum_time_headway for item in metrics.scenarios
    )
    assert metrics.minimum_time_to_collision == min(
        item.minimum_time_to_collision for item in metrics.scenarios
    )


def test_the_scenario_table_has_one_row_per_scenario(planned: tuple[object, ...]) -> None:
    """The table is the Results section, so its shape is part of the contract."""
    table = scenario_table(suite_metrics(planned))  # type: ignore[arg-type]
    lines = table.splitlines()
    assert len(lines) == len(SHORT) + 2
    assert lines[0].startswith("| Scenario")
    assert set(lines[1]) <= {"|", "-", " "}
    for scenario, line in zip(SHORT, lines[2:], strict=True):
        assert scenario.name in line


def test_the_comparison_table_pairs_the_policies(
    planned: tuple[object, ...], baseline: tuple[object, ...]
) -> None:
    """Each row compares the same scenario under both policies."""
    table = comparison_table(
        suite_metrics(planned),  # type: ignore[arg-type]
        suite_metrics(baseline),  # type: ignore[arg-type]
    )
    lines = table.splitlines()
    assert len(lines) == len(SHORT) + 2
    assert "Gain (percent)" in lines[0]
    for scenario, line in zip(SHORT, lines[2:], strict=True):
        assert scenario.name in line


def test_an_unbounded_quantity_is_rendered_as_such(planned: tuple[object, ...]) -> None:
    """A scenario with no leader prints ``inf`` rather than a misleading large number."""
    table = scenario_table(suite_metrics(planned))  # type: ignore[arg-type]
    free_flow = next(line for line in table.splitlines() if "free_flow" in line)
    assert "inf" in free_flow


def test_an_empty_suite_is_rejected() -> None:
    """Aggregating nothing is an error, not an empty table."""
    with pytest.raises(ValueError, match="at least one"):
        suite_metrics(())
