"""Tier one. What the distribution promises to anything that installs it.

A package can pass ``mypy --strict`` on every line it contains and still deliver
no type information at all, because a type checker reading an installed
dependency ignores its annotations unless PEP 561 marker file is present. That
makes the marker part of the public interface rather than packaging trivia, and
therefore something to assert rather than to remember.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import behavior_planner

PACKAGE = Path(behavior_planner.__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[1]


def test_the_typing_marker_sits_inside_the_package_directory() -> None:
    """PEP 561 requires ``py.typed`` beside ``__init__.py``, not beside it."""
    marker = PACKAGE / "py.typed"
    assert marker.is_file(), f"no py.typed in {PACKAGE}"
    assert (PACKAGE / "__init__.py").is_file()
    assert marker.parent == PACKAGE


def test_the_typing_marker_is_empty() -> None:
    """PEP 561 gives the file no contents. Anything in it is a mistake."""
    assert (PACKAGE / "py.typed").read_bytes() == b""


def test_the_wheel_is_built_from_the_directory_holding_the_marker() -> None:
    """A marker the build backend does not ship is a marker that does nothing."""
    config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert packages == ["src/behavior_planner"]
    assert (REPOSITORY / packages[0] / "py.typed").is_file()
