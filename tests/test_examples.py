"""Tier three. Every script in ``examples/`` runs to completion.

The scripts are run as subprocesses under a reduced duration, so this tier
checks the wiring, the command line and the output paths rather than the
numbers, which the other two tiers cover. Figures are written into a temporary
directory so the tests leave nothing behind.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY / "examples"
SCRIPTS = sorted(path for path in EXAMPLES.glob("*.py") if not path.name.startswith("_"))


def run_script(script: Path, *arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one example script and return its completed process."""
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_the_examples_directory_is_not_empty() -> None:
    """A parametrised test over an empty collection passes vacuously, so check it."""
    assert SCRIPTS, "examples/ contains no scripts to run"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_each_example_runs_to_completion(script: Path, tmp_path: Path) -> None:
    """Every script exits zero under a reduced duration and prints something."""
    arguments = ["--duration", "6.0"]
    if script.name == "plot_results.py":
        arguments += ["--output", str(tmp_path)]
    result = run_script(script, *arguments, cwd=REPOSITORY)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_the_suite_example_reports_zero_collisions(tmp_path: Path) -> None:
    """The headline number of the Results section is printed by the script itself."""
    result = run_script(EXAMPLES / "run_suite.py", "--duration", "10.0", cwd=REPOSITORY)
    assert result.returncode == 0, result.stderr
    assert "Total collisions: 0" in result.stdout


def test_the_scenario_example_accepts_every_scenario_name(tmp_path: Path) -> None:
    """Each scenario in the suite is reachable from the command line."""
    from behavior_planner.pipeline.scenarios import standard_suite

    for scenario in standard_suite():
        result = run_script(
            EXAMPLES / "run_scenario.py",
            "--scenario",
            scenario.name,
            "--duration",
            "4.0",
            "--transitions-only",
            cwd=REPOSITORY,
        )
        assert result.returncode == 0, result.stderr
        assert scenario.name in result.stdout


def test_the_scenario_example_rejects_an_unknown_scenario() -> None:
    """An unknown name is refused by the parser rather than silently ignored."""
    result = run_script(EXAMPLES / "run_scenario.py", "--scenario", "nope", cwd=REPOSITORY)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_the_plotting_example_writes_its_figures(tmp_path: Path) -> None:
    """Every figure the README refers to is produced by one command."""
    result = run_script(
        EXAMPLES / "plot_results.py",
        "--duration",
        "8.0",
        "--output",
        str(tmp_path),
        cwd=REPOSITORY,
    )
    assert result.returncode == 0, result.stderr
    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == [
        "slow_leader_timeline.png",
        "suite_speed.png",
        "time_to_collision.png",
    ]
    for path in tmp_path.glob("*.png"):
        assert path.stat().st_size > 1000
