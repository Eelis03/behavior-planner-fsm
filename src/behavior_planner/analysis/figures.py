"""Figures produced from run traces and suite metrics."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from behavior_planner.analysis.metrics import SuiteMetrics
from behavior_planner.model.states import BehaviorState
from behavior_planner.pipeline.trace import RunTrace

__all__ = ["plot_scenario", "plot_suite", "plot_time_to_collision"]

_STATE_ORDER: tuple[BehaviorState, ...] = (
    BehaviorState.LANE_CHANGE_RIGHT,
    BehaviorState.PREPARE_LANE_CHANGE_RIGHT,
    BehaviorState.KEEP_LANE,
    BehaviorState.PREPARE_LANE_CHANGE_LEFT,
    BehaviorState.LANE_CHANGE_LEFT,
)


def plot_scenario(trace: RunTrace, path: Path) -> Path:
    """Speed, lateral offset and behaviour state against time for one run."""
    times = [record.time for record in trace.records]
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), sharex=True)

    axes[0].plot(times, [record.speed for record in trace.records], color="#1f77b4")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, [record.d for record in trace.records], color="#2ca02c")
    axes[1].set_ylabel("lateral offset (m)")
    axes[1].grid(alpha=0.3)

    positions = {state: index for index, state in enumerate(_STATE_ORDER)}
    axes[2].step(
        times,
        [positions[record.state] for record in trace.records],
        where="post",
        color="#d62728",
    )
    axes[2].set_yticks(range(len(_STATE_ORDER)))
    axes[2].set_yticklabels([state.value for state in _STATE_ORDER], fontsize=8)
    axes[2].set_ylabel("behaviour state")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)

    figure.suptitle(f"{trace.scenario} under the {trace.policy} policy")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_time_to_collision(traces: tuple[RunTrace, ...], path: Path) -> Path:
    """Cumulative distribution of the finite times to collision per scenario."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for trace in traces:
        samples = np.array(
            [value for value in trace.finite_times_to_collision() if math.isfinite(value)],
            dtype=np.float64,
        )
        if samples.size == 0:
            continue
        ordered = np.sort(samples)
        fraction = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
        axis.step(ordered, fraction, where="post", label=trace.scenario)
    axis.set_xscale("log")
    axis.set_xlabel("time to collision (s)")
    axis.set_ylabel("fraction of closing steps at or below")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_suite(planned: SuiteMetrics, baseline: SuiteMetrics, path: Path) -> Path:
    """Mean ego speed per scenario under the planner and under the control."""
    names = [item.scenario for item in planned.scenarios]
    index = {item.scenario: item.mean_speed for item in baseline.scenarios}
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.38

    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.bar(
        positions - 0.5 * width,
        [item.mean_speed for item in planned.scenarios],
        width,
        label=planned.policy,
        color="#1f77b4",
    )
    axis.bar(
        positions + 0.5 * width,
        [index[name] for name in names],
        width,
        label=baseline.policy,
        color="#aec7e8",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axis.set_ylabel("mean ego speed (m/s)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
