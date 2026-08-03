"""Sweep the progress weight and print what the cost, the gate and the planner said.

    uv run python examples/run_gate_experiment.py

The scene is fixed: the ego is behind a slow leader, the lane it wants is
occupied, and the soft safety weight is zero. Only the progress weight moves,
from the default of 5 to a thousand times it. The numbers the README quotes for
its opening claim are the ones this script prints.
"""

from __future__ import annotations

import argparse

from behavior_planner.pipeline.gate_experiment import PROGRESS_WEIGHTS, gate_weight_sweep


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=0.0,
        help="soft safety weight in the cost function, zero by default",
    )
    return parser.parse_args()


def main() -> None:
    """Print the sweep as a table and then the property it demonstrates."""
    arguments = parse_arguments()
    samples = gate_weight_sweep(PROGRESS_WEIGHTS, safety_weight=arguments.safety_weight)

    header = (
        f"{'progress':>9}  {'keep lane':>10}  {'change left':>12}  "
        f"{'cheaper':>8}  {'gate':>9}  {'reason':<12} chosen"
    )
    print(header)
    print("-" * len(header))
    for sample in samples:
        print(
            f"{sample.progress_weight:9.0f}  {sample.keep_lane_cost:10.4f}  "
            f"{sample.lane_change_cost:12.4f}  "
            f"{'change' if sample.cost_prefers_change else 'keep':>8}  "
            f"{'permitted' if sample.allowed else 'refused':>9}  "
            f"{sample.reason.value:<12} {sample.chosen.value}"
        )

    preferred = sum(1 for sample in samples if sample.cost_prefers_change)
    refused = sum(1 for sample in samples if not sample.allowed)
    taken = sum(1 for sample in samples if sample.chosen.value == "lane_change_left")
    span = max(sample.cost_margin for sample in samples) / min(
        sample.cost_margin for sample in samples
    )
    print()
    print(f"weights swept                        {len(samples)}")
    print(f"weights at which the cost prefers    {preferred}")
    print(f"weights at which the gate refuses    {refused}")
    print(f"weights at which the change is taken {taken}")
    print(f"cost preference grows by a factor of {span:.0f}")


if __name__ == "__main__":
    main()
