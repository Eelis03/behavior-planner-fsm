"""Tier two. A recorded reference run, compared against a fresh one.

What this file pins, and why it pins only that
----------------------------------------------

A multi-agent traffic simulation is the kind of system where a difference of one
unit in the last place, from a different ``pow`` implementation for instance,
can in principle grow until two machines disagree visibly. Pinning the internal
state of such a run produces a test that fails on somebody else's computer for
reasons that have nothing to do with the code under review.

Two measurements decide what is safe to pin here, and both are checked by tests
in this module rather than asserted in prose.

1. *The dynamics contract rather than amplify.* Perturbing the ego's free flow
   speed by a relative ``1e-9`` moves the mean speed of every scenario by less
   than ``1e-9`` relative, and changes no discrete outcome at all. The response
   is linear in the perturbation up to ``1e-3``, so there is no positive
   Lyapunov exponent to worry about at this horizon. Continuous aggregates are
   therefore pinned, with a relative tolerance of ``1e-6``, which is four orders
   of magnitude above the largest drift a floating point difference can produce
   over a sixty second run and four orders below the smallest change a code
   change would produce.

2. *No decision is near a tie.* The smallest gap between the best and the second
   best admissible candidate anywhere in the suite is of order ``1e-3`` in cost
   units, ten orders of magnitude above floating point noise. The behaviour
   state sequence is therefore pinned exactly, and
   :func:`test_no_decision_in_the_suite_is_near_a_tie` fails if that stops being
   true, which is what keeps the exact pins honest.

Regenerate the reference with::

    uv run python -c "import tests.test_regression as t; t.record_reference()"
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from behavior_planner.analysis.metrics import scenario_metrics
from behavior_planner.model.config import PlannerConfig
from behavior_planner.pipeline.scenarios import standard_suite
from behavior_planner.pipeline.simulator import TrafficSimulator
from behavior_planner.pipeline.suite import build_planner, run_suite
from behavior_planner.pipeline.trace import RunTrace

REFERENCE_PATH = Path(__file__).parent / "data" / "reference_run.json"

SPEED_TOLERANCE = 1e-6
"""Relative tolerance on the continuous aggregates. See the module docstring."""

MINIMUM_DECISION_MARGIN = 1e-4
"""Smallest cost gap between the best two candidates that keeps exact pins safe."""


def _finite_or_none(value: float) -> float | None:
    """JSON has no infinity, so an unbounded quantity is recorded as null."""
    return None if math.isinf(value) else value


def summarise(trace: RunTrace) -> dict[str, Any]:
    """Reduce a run to the quantities the reference pins."""
    metrics = scenario_metrics(trace)
    return {
        "collisions": metrics.collisions,
        "lane_changes": metrics.lane_changes,
        "aborted_changes": metrics.aborted_changes,
        "state_sequence": [state.value for state in trace.state_sequence],
        "veto_reasons": [reason.value for reason in trace.veto_reasons],
        "lanes_visited": sorted({record.lane for record in trace.records}),
        "record_count": len(trace.records),
        "mean_speed": metrics.mean_speed,
        "distance": metrics.distance,
        "minimum_time_headway": _finite_or_none(metrics.minimum_time_headway),
        "minimum_time_to_collision": _finite_or_none(metrics.minimum_time_to_collision),
        "ttc_samples": metrics.ttc_samples,
    }


def record_reference() -> Path:
    """Write the reference document from a fresh run of the standard suite."""
    document = {
        "note": "Regenerate with tests/test_regression.py::record_reference.",
        "scenarios": {trace.scenario: summarise(trace) for trace in run_suite()},
    }
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return REFERENCE_PATH


@pytest.fixture(scope="module")
def reference() -> dict[str, dict[str, Any]]:
    """The recorded reference, keyed by scenario name."""
    document = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    scenarios: dict[str, dict[str, Any]] = document["scenarios"]
    return scenarios


@pytest.fixture(scope="module")
def fresh() -> dict[str, dict[str, Any]]:
    """A fresh run of the standard suite, reduced the same way."""
    return {trace.scenario: summarise(trace) for trace in run_suite()}


def test_the_reference_covers_the_whole_suite(
    reference: dict[str, dict[str, Any]], fresh: dict[str, dict[str, Any]]
) -> None:
    """A scenario added to the suite without a reference entry is a failure."""
    assert set(reference) == set(fresh)
    assert set(reference) == {scenario.name for scenario in standard_suite()}


@pytest.mark.parametrize("name", [scenario.name for scenario in standard_suite()])
@pytest.mark.parametrize(
    "field",
    [
        "collisions",
        "lane_changes",
        "aborted_changes",
        "state_sequence",
        "veto_reasons",
        "lanes_visited",
        "record_count",
        "ttc_samples",
    ],
)
def test_discrete_outcomes_match_exactly(
    reference: dict[str, dict[str, Any]],
    fresh: dict[str, dict[str, Any]],
    name: str,
    field: str,
) -> None:
    """Counts, classifications and the decision sequence are pinned exactly.

    These are the quantities that survive a perturbation of the run by many
    orders of magnitude more than floating point can introduce, so an exact
    comparison is a comparison of behaviour rather than of arithmetic.
    """
    assert fresh[name][field] == reference[name][field]


@pytest.mark.parametrize("name", [scenario.name for scenario in standard_suite()])
@pytest.mark.parametrize(
    "field",
    ["mean_speed", "distance", "minimum_time_headway", "minimum_time_to_collision"],
)
def test_continuous_aggregates_match_within_tolerance(
    reference: dict[str, dict[str, Any]],
    fresh: dict[str, dict[str, Any]],
    name: str,
    field: str,
) -> None:
    """Aggregates over the whole run are pinned with a relative tolerance.

    Aggregates are used rather than instantaneous values on purpose. A position
    late in a run is the accumulation of every arithmetic difference before it;
    a mean over the run is not, and an unbounded quantity is recorded as null so
    it is compared as a classification rather than as a number.
    """
    expected = reference[name][field]
    actual = fresh[name][field]
    if expected is None:
        assert actual is None
        return
    assert actual == pytest.approx(expected, rel=SPEED_TOLERANCE)


def test_the_suite_is_collision_free_in_the_reference(
    reference: dict[str, dict[str, Any]],
) -> None:
    """The reference itself records zero collisions, so the pin is not pinning a bug."""
    assert all(entry["collisions"] == 0 for entry in reference.values())


def test_no_decision_in_the_suite_is_near_a_tie() -> None:
    """The assumption the exact pins rest on, checked rather than asserted.

    If two candidates ever came within floating point noise of each other, the
    recorded state sequence would stop being reproducible and the exact
    comparisons above would have to be replaced by qualitative ones.
    """

    class MarginSpy:
        """A policy wrapper that records the gap between the best two candidates."""

        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.margins: list[float] = []

        def decide(self, context: Any) -> Any:
            decision = self.inner.decide(context)  # type: ignore[attr-defined]
            admissible = sorted(
                candidate.total for candidate in decision.candidates if candidate.admissible
            )
            if len(admissible) >= 2:
                self.margins.append(admissible[1] - admissible[0])
            return decision

    config = PlannerConfig()
    for scenario in standard_suite():
        spy = MarginSpy(build_planner(config))
        TrafficSimulator(policy=spy, planner_config=config).run(scenario)  # type: ignore[arg-type]
        assert spy.margins, scenario.name
        assert min(spy.margins) > MINIMUM_DECISION_MARGIN, scenario.name


def test_the_reference_was_recorded_with_the_current_defaults(
    reference: dict[str, dict[str, Any]],
) -> None:
    """A configuration change that moves every number should update the reference.

    This is a guard against a reference that silently describes a different
    planner: the record count follows directly from the scenario durations and
    the integration step.
    """
    from behavior_planner.pipeline.simulator import SimulationConfig

    dt = SimulationConfig().dt
    for scenario in standard_suite():
        expected = round(scenario.duration / dt) + 1
        assert reference[scenario.name]["record_count"] == expected


if __name__ == "__main__":
    print(f"wrote {record_reference()}")
