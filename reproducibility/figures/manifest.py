"""Validated manifest for every publication figure.

The manifest deliberately describes presentation inputs, not simulations. An
experiment owns the expensive run; this layer owns the deterministic conversion
of its small derived output into a paper-ready TikZ fragment.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from mairl_exp.config import DEFAULT_MANIFEST, GENERATED_ROOT, REPOSITORY_ROOT

_FIGURE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SECTIONS = frozenset({"main", "appendix"})
_DATA_RECIPES = frozenset(
    {
        "directional_transfer",
        "lobeke_behavioral_transfer",
        "lobeke_diagnostics",
        "lobeke_environment",
        "lobeke_tuning",
        "lobeke_source_convergence",
        "lobeke_transfer",
        "lobeke_transfer_compact",
        "lobeke_downstream",
        "lower_bound",
        "sparse_information",
        "zero_sum",
    }
)
_RECIPES = frozenset({"copy", *_DATA_RECIPES})


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """One registered figure and its deterministic rendering recipe."""

    id: str
    section: str
    title: str
    recipe: str
    output: Path
    source: Path | None = None
    data: Path | None = None
    x_label: str | None = None
    variant: str | None = None

    @property
    def output_path(self) -> Path:
        return GENERATED_ROOT / self.output


@dataclass(frozen=True, slots=True)
class FigureManifest:
    """Validated collection of figure specifications."""

    figures: tuple[FigureSpec, ...]

    def select(self, identifiers: list[str] | tuple[str, ...]) -> tuple[FigureSpec, ...]:
        if not identifiers:
            return self.figures
        by_id = {figure.id: figure for figure in self.figures}
        unknown = sorted(set(identifiers) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown figure ids: {', '.join(unknown)}")
        return tuple(by_id[identifier] for identifier in identifiers)


def _repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"manifest paths must be repository-relative: {value!r}")
    return REPOSITORY_ROOT / path


def _parse_figure(raw: dict[str, object]) -> FigureSpec:
    required = {"id", "section", "title", "recipe", "output"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"figure entry is missing fields: {', '.join(missing)}")

    identifier = str(raw["id"])
    section = str(raw["section"])
    recipe = str(raw["recipe"])
    output = Path(str(raw["output"]))
    source_value = raw.get("source")
    data_value = raw.get("data")

    if not _FIGURE_ID.fullmatch(identifier):
        raise ValueError(f"invalid figure id: {identifier!r}")
    if section not in _SECTIONS:
        raise ValueError(f"invalid section for {identifier}: {section!r}")
    if recipe not in _RECIPES:
        raise ValueError(f"invalid recipe for {identifier}: {recipe!r}")
    if output.is_absolute() or ".." in output.parts or output.suffix != ".tex":
        raise ValueError(f"invalid generated output for {identifier}: {output}")
    if output.parts[0] != section:
        raise ValueError(f"output for {identifier} must begin with {section}/")

    source = None if source_value is None else _repository_path(str(source_value))
    data = None if data_value is None else _repository_path(str(data_value))
    if recipe == "copy" and source is None:
        raise ValueError(f"copy recipe for {identifier} requires a source")
    if recipe != "copy" and source is not None:
        raise ValueError(f"only copy recipes may define a source ({identifier})")
    if recipe in _DATA_RECIPES and data is None:
        raise ValueError(f"data recipe for {identifier} requires a data input")
    if recipe not in _DATA_RECIPES and data is not None:
        raise ValueError(f"only data recipes may define a data input ({identifier})")
    variant = None if raw.get("variant") is None else str(raw["variant"])
    if recipe == "lobeke_behavioral_transfer" and variant not in {"tv", "gap"}:
        raise ValueError(f"behavioral-transfer figure {identifier} requires variant='tv' or 'gap'")
    if recipe != "lobeke_behavioral_transfer" and variant is not None:
        raise ValueError(f"recipe {recipe} may not define a variant ({identifier})")

    return FigureSpec(
        id=identifier,
        section=section,
        title=str(raw["title"]),
        recipe=recipe,
        output=output,
        source=source,
        data=data,
        x_label=None if raw.get("x_label") is None else str(raw["x_label"]),
        variant=variant,
    )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> FigureManifest:
    """Load and validate a figure manifest."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    raw_figures = raw.get("figures")
    if not isinstance(raw_figures, list) or not raw_figures:
        raise ValueError("figure manifest must contain at least one [[figures]] entry")
    figures = tuple(_parse_figure(entry) for entry in raw_figures)
    identifiers = [figure.id for figure in figures]
    outputs = [figure.output for figure in figures]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("figure ids must be unique")
    if len(outputs) != len(set(outputs)):
        raise ValueError("figure outputs must be unique")
    return FigureManifest(figures=figures)
