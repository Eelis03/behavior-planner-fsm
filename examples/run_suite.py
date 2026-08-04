"""Run the scenario suite under the planner and under the lane keeping control.

    uv run python examples/run_suite.py

The two tables printed here are the ones reproduced in the Results section of
the README.
"""

from __future__ import annotations

import argparse

from behavior_planner.analysis.metrics import suite_metrics
from behavior_planner.analysis.report import comparison_table, scenario_table
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_suite


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="override the duration of every scenario, in seconds",
    )
    return parser.parse_args()


def main() -> None:
    """Run both policies over the standard suite and report the tables."""
    arguments = parse_arguments()
    scenarios = standard_suite()
    if arguments.duration is not None:
        scenarios = tuple(item.with_duration(arguments.duration) for item in scenarios)

    traces = run_suite(scenarios)
    planned = suite_metrics(traces)
    baseline = suite_metrics(run_suite(scenarios, baseline=True))

    print("Scenario suite under the finite state machine planner")
    print()
    print(scenario_table(planned))
    print()
    print("Ego speed against the lane keeping control policy")
    print()
    print(comparison_table(planned, baseline))
    print()
    print(f"Total collisions: {planned.total_collisions}")
    print(f"Total lane changes: {planned.total_lane_changes}")
    print(f"Mean ego speed: {planned.mean_speed:.2f} m/s")
    print(f"Smallest time headway: {planned.minimum_time_headway:.2f} s")
    print(f"Smallest time to collision: {planned.minimum_time_to_collision:.2f} s")
    print(f"Tightest safety gate margin: {planned.minimum_gate_margin:.3f} of the threshold")

    reasons = sorted({reason.value for trace in traces for reason in trace.veto_reasons})
    print(f"Safety gate veto reasons raised: {', '.join(reasons) if reasons else 'none'}")


if __name__ == "__main__":
    main()
