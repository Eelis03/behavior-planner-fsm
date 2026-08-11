"""Regenerate every figure the README embeds.

    uv run python examples/plot_results.py --output docs/figures

Three files land in the output directory: the gate weight sweep behind the
claim the README opens with, the timeline of an overtake that succeeds, and the
timeline of one that is refused four times before a gap arrives.

The tracked copies under ``docs/figures/`` are snapshots. Matplotlib output is
not byte reproducible across platforms or font sets, so CI never compares them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from behavior_planner.analysis.figures import plot_gate_weight_sweep, plot_scenario
from behavior_planner.pipeline.gate_experiment import gate_weight_sweep
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_suite

TIMELINES: tuple[str, ...] = ("slow_leader", "gap_wait")
"""The two scenarios plotted, chosen as a contrast rather than a sample.

One is an overtake the gate permits and one is an overtake it refuses four
times, and the same three panels tell both stories.
"""


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="override the duration of every scenario, in seconds",
    )
    return parser.parse_args()


def main() -> None:
    """Run what each figure needs and write the three published files."""
    arguments = parse_arguments()
    scenarios = tuple(scenario for scenario in standard_suite() if scenario.name in TIMELINES)
    if arguments.duration is not None:
        scenarios = tuple(item.with_duration(arguments.duration) for item in scenarios)

    traces = run_suite(scenarios)
    written = [
        plot_gate_weight_sweep(gate_weight_sweep(), arguments.output / "gate_weight_sweep.png")
    ]
    written.extend(
        plot_scenario(trace, arguments.output / f"{trace.scenario}_timeline.png")
        for trace in traces
    )
    for path in written:
        print(f"wrote {path} ({path.stat().st_size // 1024} kB)")
    print(f"total {sum(path.stat().st_size for path in written) // 1024} kB")


if __name__ == "__main__":
    main()
