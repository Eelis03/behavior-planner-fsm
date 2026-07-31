"""Write the figures for the scenario suite.

    uv run python examples/plot_results.py

Three files land in the output directory: the behaviour timeline of one
scenario, the cumulative distribution of the times to collision across the
suite, and the ego speed of the planner beside the lane keeping control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from behavior_planner.analysis.figures import plot_scenario, plot_suite, plot_time_to_collision
from behavior_planner.analysis.metrics import suite_metrics
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_suite


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    names = [scenario.name for scenario in standard_suite()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--scenario", choices=names, default="slow_leader")
    parser.add_argument("--duration", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the suite and write every figure."""
    arguments = parse_arguments()
    scenarios = standard_suite()
    if arguments.duration is not None:
        scenarios = tuple(item.with_duration(arguments.duration) for item in scenarios)

    planned = run_suite(scenarios)
    baseline = run_suite(scenarios, baseline=True)
    chosen = next(trace for trace in planned if trace.scenario == arguments.scenario)

    written = (
        plot_scenario(chosen, arguments.output / f"{chosen.scenario}_timeline.png"),
        plot_time_to_collision(planned, arguments.output / "time_to_collision.png"),
        plot_suite(
            suite_metrics(planned),
            suite_metrics(baseline),
            arguments.output / "suite_speed.png",
        ),
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
