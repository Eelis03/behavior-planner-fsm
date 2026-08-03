"""Tier one and two. The density and seed sweep that replaced the single runs.

The seven scenarios of the standard suite each run once, at one seed, which
demonstrates the behaviours and does not characterise the planner. This module
covers the grid that does: several densities, several seeds at each, and a
distribution rather than a number.

Runs here are short on purpose. What is under test is that the grid is built
correctly, reduced correctly and reproducible, not what the planner does over a
full minute, which the example script measures.
"""

from __future__ import annotations

import pytest

from behavior_planner.analysis.metrics import DensityMetrics, SweepMetrics, sweep_metrics
from behavior_planner.analysis.report import sweep_table, worst_paired_gain
from behavior_planner.pipeline.scenarios import SWEEP_DENSITIES, SWEEP_SEEDS, sweep_scenarios
from behavior_planner.pipeline.suite import DensityGroup, run_sweep

DENSITIES = (4, 16)
SEEDS = (11, 12, 13)
DURATION = 12.0


@pytest.fixture(scope="module")
def planned() -> tuple[DensityGroup, ...]:
    """A short sweep under the planner."""
    return run_sweep(densities=DENSITIES, seeds=SEEDS, duration=DURATION)


@pytest.fixture(scope="module")
def control() -> tuple[DensityGroup, ...]:
    """The same grid under the lane keeping control policy."""
    return run_sweep(densities=DENSITIES, seeds=SEEDS, duration=DURATION, baseline=True)


def test_the_grid_is_the_product_of_densities_and_seeds() -> None:
    """A sweep that quietly dropped a cell would report a smaller spread."""
    scenarios = sweep_scenarios(densities=DENSITIES, seeds=SEEDS, duration=DURATION)
    assert len(scenarios) == len(DENSITIES) * len(SEEDS)
    assert len({scenario.name for scenario in scenarios}) == len(scenarios)
    assert {scenario.seed for scenario in scenarios} == set(SEEDS)


def test_each_scenario_names_its_own_density_and_seed() -> None:
    """A run has to be identifiable from its trace, which carries only the name."""
    for scenario in sweep_scenarios(densities=DENSITIES, seeds=SEEDS):
        assert scenario.fill is not None
        assert scenario.name == f"density_{scenario.fill.per_lane:02d}_seed_{scenario.seed}"


def test_only_the_density_and_the_seed_vary_across_the_grid() -> None:
    """A sweep is a controlled experiment or it is nothing.

    If the road, the ego or the speed distribution moved with the density, a
    difference between two rows of the table would have several possible causes
    and would therefore have none.
    """
    scenarios = sweep_scenarios(densities=SWEEP_DENSITIES, seeds=SWEEP_SEEDS)
    fills = [scenario.fill for scenario in scenarios]
    assert all(fill is not None for fill in fills)
    assert len({scenario.road for scenario in scenarios}) == 1
    assert len({scenario.ego for scenario in scenarios}) == 1
    assert len({scenario.duration for scenario in scenarios}) == 1
    assert len({(fill.speed_mean, fill.speed_gradient, fill.speed_sigma) for fill in fills}) == 1  # type: ignore[union-attr]


def test_an_empty_grid_is_rejected() -> None:
    """Sweeping nothing is a mistake, not an empty result."""
    with pytest.raises(ValueError, match="at least one density"):
        sweep_scenarios(densities=(), seeds=SEEDS)
    with pytest.raises(ValueError, match="at least one seed"):
        sweep_scenarios(densities=DENSITIES, seeds=())
    with pytest.raises(ValueError, match="must be positive"):
        sweep_scenarios(densities=(0,), seeds=SEEDS)


def test_the_sweep_groups_its_runs_by_density(planned: tuple[DensityGroup, ...]) -> None:
    """Each group carries its density rather than leaving it to be parsed back."""
    assert tuple(group.density for group in planned) == DENSITIES
    for group in planned:
        assert len(group.traces) == len(SEEDS)
        assert all(f"density_{group.density:02d}" in trace.scenario for trace in group.traces)


def test_a_group_with_no_runs_is_rejected() -> None:
    """A density that produced nothing has no distribution to report."""
    with pytest.raises(ValueError, match="no runs"):
        DensityGroup(density=8, traces=())
    with pytest.raises(ValueError, match="no runs"):
        DensityMetrics(density=8, runs=())


def test_the_sweep_is_reproducible(planned: tuple[DensityGroup, ...]) -> None:
    """The same grid run twice gives the same numbers, or the spread means nothing."""
    again = run_sweep(densities=DENSITIES, seeds=SEEDS, duration=DURATION)
    first = sweep_metrics(planned)
    second = sweep_metrics(again)
    for left, right in zip(first.densities, second.densities, strict=True):
        assert left.speed_median == right.speed_median
        assert left.lane_changes == right.lane_changes
        assert left.collisions == right.collisions


def test_a_different_seed_gives_different_traffic() -> None:
    """A sweep over seeds that produced the same run every time would measure nothing."""
    groups = run_sweep(densities=(12,), seeds=(21, 22, 23), duration=DURATION)
    speeds = {round(trace.records[-1].speed, 6) for trace in groups[0].traces}
    assert len(speeds) == len(groups[0].traces)


def test_no_run_in_the_sweep_puts_the_ego_in_a_collision(
    planned: tuple[DensityGroup, ...],
) -> None:
    """The one required result, asserted over the grid rather than over one run."""
    metrics = sweep_metrics(planned)
    assert metrics.total_ego_collisions == 0
    assert metrics.total_runs == len(DENSITIES) * len(SEEDS)


def test_the_distribution_is_ordered(planned: tuple[DensityGroup, ...]) -> None:
    """The three percentiles are a summary of one sample, so they must be sorted."""
    for item in sweep_metrics(planned).densities:
        assert item.speed_p05 <= item.speed_median <= item.speed_p95
        assert item.count == len(SEEDS)


def test_density_costs_speed(planned: tuple[DensityGroup, ...]) -> None:
    """More traffic is slower traffic, which is the sweep's sanity check.

    If this failed, the density parameter would not be reaching the placement
    routine and every row of the table would be the same experiment.
    """
    metrics = sweep_metrics(planned)
    sparse, dense = metrics.densities[0], metrics.densities[-1]
    assert sparse.density < dense.density
    assert sparse.speed_median > dense.speed_median


def test_the_sweep_aggregate_reports_the_worst_case(planned: tuple[DensityGroup, ...]) -> None:
    """The headway and time to collision of a sweep are minima over every run."""
    metrics = sweep_metrics(planned)
    assert metrics.minimum_time_headway == min(
        item.minimum_time_headway for item in metrics.densities
    )
    assert metrics.minimum_time_to_collision == min(
        item.minimum_time_to_collision for item in metrics.densities
    )


def test_the_ego_collision_count_is_a_subset_of_all_collisions(
    planned: tuple[DensityGroup, ...],
) -> None:
    """A collision between two traffic vehicles is not a planner failure.

    Separating the two is what lets the sweep report an honest zero for the ego
    without claiming the traffic model is collision free at every density.
    """
    for group in planned:
        for trace in group.traces:
            assert trace.ego_collision_count <= trace.collision_count
            assert all(0 in pair for pair in trace.ego_collision_pairs)
            assert set(trace.ego_collision_pairs) <= set(trace.collision_pairs)


def test_the_sweep_table_has_one_row_per_density(
    planned: tuple[DensityGroup, ...], control: tuple[DensityGroup, ...]
) -> None:
    """The table is the characterisation, so its shape is part of the contract."""
    table = sweep_table(sweep_metrics(planned), sweep_metrics(control))
    lines = table.splitlines()
    assert len(lines) == len(DENSITIES) + 2
    assert lines[0].startswith("| Vehicles per lane")
    assert "Median gain (percent)" in lines[0]
    assert "Mean gain (percent)" in lines[0]
    for density, line in zip(DENSITIES, lines[2:], strict=True):
        assert line.startswith(f"| {density}")


def test_the_gain_is_paired_by_seed(
    planned: tuple[DensityGroup, ...], control: tuple[DensityGroup, ...]
) -> None:
    """A gain computed across seeds would compare two different traffic streams."""
    metrics = sweep_metrics(planned)
    reference = sweep_metrics(control)
    for left, right in zip(metrics.densities, reference.densities, strict=True):
        assert {item.scenario for item in left.runs} == {item.scenario for item in right.runs}


def test_a_sweep_mixing_policies_is_rejected(
    planned: tuple[DensityGroup, ...], control: tuple[DensityGroup, ...]
) -> None:
    """One table column is one policy, and a mixed sweep would silently blend them."""
    mixed = (planned[0], control[0])
    with pytest.raises(ValueError, match="one policy"):
        sweep_metrics(mixed)


def test_an_empty_sweep_is_rejected() -> None:
    """Reducing nothing is an error, not an empty table."""
    with pytest.raises(ValueError, match="at least one density"):
        sweep_metrics(())
    with pytest.raises(ValueError, match="at least one density"):
        SweepMetrics(policy="fsm", densities=())


def test_the_worst_case_is_reported_by_name(
    planned: tuple[DensityGroup, ...], control: tuple[DensityGroup, ...]
) -> None:
    """A summary that reports only the middle would hide the run that matters.

    The worst paired gain has to name a run that exists in the grid, so a reader
    can rerun exactly that seed at exactly that density.
    """
    metrics = sweep_metrics(planned)
    reference = sweep_metrics(control)
    scenario, gain = worst_paired_gain(metrics, reference)
    names = {item.scenario for density in metrics.densities for item in density.runs}
    assert scenario in names

    control_speed = {
        item.scenario: item.mean_speed for density in reference.densities for item in density.runs
    }
    every = [
        100.0 * (item.mean_speed - control_speed[item.scenario]) / control_speed[item.scenario]
        for density in metrics.densities
        for item in density.runs
    ]
    assert gain == pytest.approx(min(every))


def test_the_slowest_run_is_the_one_with_the_lowest_mean(
    planned: tuple[DensityGroup, ...],
) -> None:
    """Naming the slowest run is only useful if it is actually the slowest."""
    metrics = sweep_metrics(planned)
    every = [item for density in metrics.densities for item in density.runs]
    assert metrics.slowest_run.mean_speed == min(item.mean_speed for item in every)


def test_the_minimum_speed_is_not_the_mean_speed(planned: tuple[DensityGroup, ...]) -> None:
    """A mean hides a stop, which is why both are recorded."""
    metrics = sweep_metrics(planned)
    for density in metrics.densities:
        for item in density.runs:
            assert item.minimum_speed <= item.mean_speed
        assert density.minimum_speed == min(item.minimum_speed for item in density.runs)
    assert metrics.minimum_speed == min(item.minimum_speed for item in metrics.densities)
