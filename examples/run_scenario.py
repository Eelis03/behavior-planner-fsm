"""Run one scenario and print the decisions the behaviour layer took.

    uv run python examples/run_scenario.py --scenario slow_leader

Only the planning cycles are printed, so the output is the decision sequence
rather than the integration trace.
"""

from __future__ import annotations

import argparse

from behavior_planner.analysis.metrics import scenario_metrics
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_scenario


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    names = [scenario.name for scenario in standard_suite()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=names, default="slow_leader")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--transitions-only",
        action="store_true",
        help="print a line only where the behaviour state changes",
    )
    return parser.parse_args()


def main() -> None:
    """Run the chosen scenario and print its decision timeline and metrics."""
    arguments = parse_arguments()
    scenario = next(item for item in standard_suite() if item.name == arguments.scenario)
    if arguments.duration is not None:
        scenario = scenario.with_duration(arguments.duration)

    trace = run_scenario(scenario)
    print(f"{scenario.name}: {scenario.description}")
    print()
    header = f"{'time':>6}  {'lane':>4}  {'speed':>6}  {'gap':>8}  {'state':<25} {'event':<9} veto"
    print(header)
    print("-" * len(header))
    previous = None
    for record in trace.records:
        if not record.planned:
            continue
        if arguments.transitions_only and record.state is previous:
            continue
        previous = record.state
        gap = "inf" if record.leader_gap == float("inf") else f"{record.leader_gap:8.2f}"
        print(
            f"{record.time:6.1f}  {record.lane:>4}  {record.speed:6.2f}  {gap:>8}  "
            f"{record.state.value:<25} {record.event.value:<9} {record.veto_reason.value}"
        )

    metrics = scenario_metrics(trace)
    print()
    print(f"collisions          {metrics.collisions}")
    print(f"mean speed          {metrics.mean_speed:.2f} m/s")
    print(f"minimum speed       {metrics.minimum_speed:.2f} m/s")
    print(f"distance            {metrics.distance:.0f} m")
    print(f"lane changes        {metrics.lane_changes}")
    print(f"minimum headway     {metrics.minimum_time_headway:.2f} s")
    print(f"minimum TTC         {metrics.minimum_time_to_collision:.2f} s")


if __name__ == "__main__":
    main()
