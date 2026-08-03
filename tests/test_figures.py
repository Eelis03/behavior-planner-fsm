"""Tier one. The published figures, checked as artefacts rather than as pictures.

Nothing here compares pixels. Matplotlib output is not byte reproducible across
platforms, font sets or backend versions, so a byte comparison would fail for
reasons that have nothing to do with the figure being wrong. What is checked is
what the README depends on: the files appear where they are said to appear, they
carry the content they are said to carry, and they fit inside the size budget
the repository publishes them under.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_planner.analysis.figures import (
    DPI,
    plot_gate_weight_sweep,
    plot_scenario,
)
from behavior_planner.pipeline.gate_experiment import gate_weight_sweep
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.suite import run_scenario
from behavior_planner.pipeline.trace import RunTrace

REPOSITORY = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPOSITORY / "docs" / "figures"
FIGURE_BUDGET_BYTES = 250 * 1024
PUBLISHED = ("gate_weight_sweep.png", "slow_leader_timeline.png", "gap_wait_timeline.png")


@pytest.fixture(scope="module")
def slow_leader() -> RunTrace:
    """A short run of the scenario the lead timeline is drawn from."""
    scenario = next(item for item in standard_suite() if item.name == "slow_leader")
    return run_scenario(scenario.with_duration(20.0))


@pytest.fixture(scope="module")
def gap_wait() -> RunTrace:
    """A short run of the scenario in which the gate refuses repeatedly."""
    scenario = next(item for item in standard_suite() if item.name == "gap_wait")
    return run_scenario(scenario.with_duration(20.0))


def test_the_timeline_figure_is_written_where_it_is_asked_for(
    slow_leader: RunTrace, tmp_path: Path
) -> None:
    """The path is returned as well as written, so a caller can report it."""
    path = plot_scenario(slow_leader, tmp_path / "nested" / "timeline.png")
    assert path == tmp_path / "nested" / "timeline.png"
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_timeline_figure_marks_the_vetoes(gap_wait: RunTrace, tmp_path: Path) -> None:
    """The veto markers are the point of the second timeline, so the run must have some."""
    from behavior_planner.model.decision import VetoReason

    vetoed = [
        record
        for record in gap_wait.records
        if record.planned and record.veto_reason is not VetoReason.NONE
    ]
    assert vetoed, "the gap_wait figure is only worth publishing if the gate objects"
    assert plot_scenario(gap_wait, tmp_path / "gap_wait.png").is_file()


def test_the_gate_sweep_figure_is_written(tmp_path: Path) -> None:
    """The lead figure of the README comes from the same experiment as its numbers."""
    path = plot_gate_weight_sweep(gate_weight_sweep(), tmp_path / "sweep.png")
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_the_gate_sweep_figure_refuses_an_empty_sweep(tmp_path: Path) -> None:
    """A figure of nothing would publish a blank axis as though it were a result."""
    with pytest.raises(ValueError, match="at least one"):
        plot_gate_weight_sweep((), tmp_path / "empty.png")


def test_the_published_figures_are_tracked() -> None:
    """A README that embeds a figure needs the figure in the repository."""
    assert FIGURE_DIR.is_dir(), "docs/figures is missing"
    assert sorted(path.name for path in FIGURE_DIR.glob("*.png")) == sorted(PUBLISHED)


def test_every_published_figure_is_a_valid_png() -> None:
    """Checked as a format rather than byte for byte.

    Matplotlib output is not byte reproducible across platforms, font sets or
    backend versions, so a hash comparison would fail on a font substitution and
    say nothing about the planner. The header is the part that has to hold.
    """
    for path in FIGURE_DIR.glob("*.png"):
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", path.name


def test_the_published_figures_fit_the_size_budget() -> None:
    """Figure size and resolution are chosen so no compression step is needed.

    A quarter of a megabyte for the whole set is what keeps a clone cheap. If
    this fails, the fix is a smaller figure or a lower resolution rather than a
    new dependency.
    """
    total = sum(path.stat().st_size for path in FIGURE_DIR.glob("*.png"))
    assert total <= FIGURE_BUDGET_BYTES, f"tracked figures total {total} bytes"


def test_every_published_figure_is_embedded_in_the_readme() -> None:
    """A tracked figure nobody sees is a tracked file for no reason."""
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for name in PUBLISHED:
        assert f"docs/figures/{name}" in readme, name


def test_the_figure_resolution_is_declared_once() -> None:
    """One constant rather than a number repeated at each call site."""
    assert DPI == 110
