"""Sweep traffic density and seed, and report distributions rather than single runs.

    uv run python examples/run_sweep.py

Every cell of the grid is a fresh random fill at one density and one seed, run
under the planner and again under the lane keeping control. What the table
reports is the spread over seeds, which is what says whether a single run was
typical.
"""

from __future__ import annotations

import argparse
import time

from behavior_planner.analysis.metrics import sweep_metrics
from behavior_planner.analysis.report import sweep_table, worst_paired_gain
from behavior_planner.pipeline.scenarios import SWEEP_DENSITIES, SWEEP_SEEDS
from behavior_planner.pipeline.suite import run_sweep


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="duration of every run in the grid, in seconds",
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="+",
        default=list(SWEEP_DENSITIES),
        help="vehicles per lane to sweep over",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SWEEP_SEEDS),
        help="seeds drawn at each density",
    )
    return parser.parse_args()


def main() -> None:
    """Run the grid under both policies and report the distribution per density."""
    arguments = parse_arguments()
    densities = tuple(arguments.densities)
    seeds = tuple(arguments.seeds)

    started = time.perf_counter()
    planned = sweep_metrics(
        run_sweep(densities=densities, seeds=seeds, duration=arguments.duration)
    )
    baseline = sweep_metrics(
        run_sweep(densities=densities, seeds=seeds, duration=arguments.duration, baseline=True)
    )
    elapsed = time.perf_counter() - started

    print(f"Density sweep: {len(densities)} densities by {len(seeds)} seeds, both policies")
    print()
    print(sweep_table(planned, baseline))
    print()
    scenario, gain = worst_paired_gain(planned, baseline)
    slowest = planned.slowest_run
    print(f"Runs under the planner:      {planned.total_runs}")
    print(f"Ego collisions:              {planned.total_ego_collisions}")
    print(f"Collisions anywhere:         {planned.total_collisions}")
    print(f"Smallest time headway:       {planned.minimum_time_headway:.2f} s")
    print(f"Smallest time to collision:  {planned.minimum_time_to_collision:.2f} s")
    print(f"Lowest ego speed:            {planned.minimum_speed:.2f} m/s")
    print(
        f"Slowest run:                 {slowest.scenario} at {slowest.mean_speed:.2f} m/s, "
        f"{slowest.lane_changes} changes, {slowest.aborted_changes} abandoned"
    )
    print(f"Worst paired gain:           {gain:+.1f} percent on {scenario}")
    print(f"Runs under the control:      {baseline.total_runs}")
    print(f"Control ego collisions:      {baseline.total_ego_collisions}")
    print(f"Control collisions anywhere: {baseline.total_collisions}")
    print(f"Control lowest ego speed:    {baseline.minimum_speed:.2f} m/s")
    print(f"Wall clock:                  {elapsed:.1f} s")


if __name__ == "__main__":
    main()
