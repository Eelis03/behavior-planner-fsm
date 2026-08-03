"""The three published figures.

Only three exist, and each is here because it shows something a table cannot.
A decision sequence unfolding in time is one such thing: the order in which the
ego approached, prepared, was refused, waited and eventually crossed is not
recoverable from any column of aggregates. A verdict that does not move while
the quantity it supposedly trades against grows by three orders of magnitude is
another.

A bar chart of the numbers already printed in a table was deleted rather than
kept, on the grounds that it showed the reader nothing the table had not.

Figure size and resolution are chosen so the three tracked files fit inside a
quarter of a megabyte without any compression step: 7.4 by 6.0 inches at 110
dots per inch is legible at full width on a laptop screen and costs about 60 kB
per figure as a paletted PNG.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from behavior_planner.model.decision import VetoReason
from behavior_planner.model.states import BehaviorState
from behavior_planner.pipeline.gate_experiment import WeightSample
from behavior_planner.pipeline.trace import RunTrace

__all__ = ["DPI", "plot_gate_weight_sweep", "plot_scenario"]

DPI: int = 110
"""Resolution of every published figure, in dots per inch."""

_STATE_ORDER: tuple[BehaviorState, ...] = (
    BehaviorState.LANE_CHANGE_RIGHT,
    BehaviorState.PREPARE_LANE_CHANGE_RIGHT,
    BehaviorState.KEEP_LANE,
    BehaviorState.PREPARE_LANE_CHANGE_LEFT,
    BehaviorState.LANE_CHANGE_LEFT,
)


def plot_scenario(trace: RunTrace, path: Path) -> Path:
    """Speed, lateral offset and behaviour state against time, with the vetoes marked.

    The three panels share a time axis on purpose. Reading down a vertical line
    gives the whole decision at that instant: how fast the ego was going, where
    it was between the lane centres, which behaviour state it was in, and
    whether the safety gate refused something on that cycle.
    """
    times = [record.time for record in trace.records]
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 6.0), sharex=True)

    axes[0].plot(times, [record.speed for record in trace.records], color="#1f77b4")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, [record.d for record in trace.records], color="#2ca02c")
    for lane in _lanes_visited(trace):
        centre = _lane_centre(trace, lane)
        axes[1].axhline(centre, color="#999999", linestyle=":", linewidth=0.8)
        axes[1].annotate(
            f"lane {lane}",
            xy=(times[-1], centre),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="#666666",
        )
    axes[1].set_ylabel("lateral offset (m)")
    axes[1].grid(alpha=0.3)

    positions = {state: index for index, state in enumerate(_STATE_ORDER)}
    axes[2].step(
        times,
        [positions[record.state] for record in trace.records],
        where="post",
        color="#d62728",
    )
    vetoed = [
        (record.time, positions[record.state])
        for record in trace.records
        if record.planned and record.veto_reason is not VetoReason.NONE
    ]
    if vetoed:
        axes[2].plot(
            [time for time, _ in vetoed],
            [level for _, level in vetoed],
            linestyle="none",
            marker="x",
            markersize=6,
            color="#333333",
            label=f"gate veto ({len(vetoed)} cycles)",
        )
        axes[2].legend(fontsize=7, loc="lower right")
    axes[2].set_yticks(range(len(_STATE_ORDER)))
    axes[2].set_yticklabels([state.value for state in _STATE_ORDER], fontsize=8)
    axes[2].set_ylabel("behaviour state")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)

    figure.suptitle(f"{trace.scenario} under the {trace.policy} policy")
    figure.tight_layout()
    return _write(figure, path)


def plot_gate_weight_sweep(samples: Sequence[WeightSample], path: Path) -> Path:
    """Cost preference against progress weight, beside the gate's verdict.

    The upper panel is what the cost function wants and the lower panel is what
    happens. The point of the figure is that the two are unrelated: the upper
    curve rises by three orders of magnitude and the lower one is a straight
    line of refusals.
    """
    if not samples:
        raise ValueError("the sweep figure needs at least one sample")

    weights = [sample.progress_weight for sample in samples]
    figure, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.2), sharex=True, height_ratios=(3.0, 1.0)
    )

    axes[0].plot(
        weights,
        [sample.cost_margin for sample in samples],
        marker="o",
        markersize=4,
        color="#1f77b4",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("cost preference for\nthe lane change")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title(
        "The cost function wants the manoeuvre more at every weight, "
        "and never gets it",
        fontsize=10,
    )

    allowed = [1 if sample.allowed else 0 for sample in samples]
    axes[1].plot(
        weights,
        allowed,
        marker="s",
        markersize=5,
        linestyle="-",
        color="#d62728",
    )
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_yticks((0, 1))
    axes[1].set_yticklabels(("refused", "permitted"), fontsize=8)
    axes[1].set_ylabel("safety gate")
    axes[1].set_xlabel("progress weight")
    axes[1].grid(alpha=0.3, which="both")
    reasons = sorted({sample.reason.value for sample in samples if not sample.allowed})
    if reasons:
        axes[1].annotate(
            f"reason: {', '.join(reasons)}",
            xy=(weights[0], 0.0),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    figure.tight_layout()
    return _write(figure, path)


def _write(figure: Figure, path: Path) -> Path:
    """Save ``figure`` to ``path`` and release it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path


def _lanes_visited(trace: RunTrace) -> tuple[int, ...]:
    """Every lane the ego was assigned to during the run, in index order."""
    return tuple(sorted({record.lane for record in trace.records}))


def _lane_centre(trace: RunTrace, lane: int) -> float:
    """Lateral offset of the centre of ``lane``, recovered from the records.

    The figure layer is handed a trace and nothing else, so the geometry is read
    back from the run rather than from a road object the analysis layer would
    otherwise have to import. The median rather than the mean, because the ego
    is assigned to its target lane from the moment it commits and the steps
    spent crossing would otherwise pull the line off the lane centre.
    """
    settled = [record.d for record in trace.records if record.lane == lane]
    return float(np.median(settled))
