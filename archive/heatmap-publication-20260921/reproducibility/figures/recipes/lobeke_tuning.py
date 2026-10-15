"""Pure TikZ renderer for the protected Lobeke tuning study."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import numpy as np

_HEAT_COLORS = (
    "mairlHeatOne",
    "mairlHeatTwo",
    "mairlHeatThree",
    "mairlHeatFour",
    "mairlHeatFive",
    "mairlHeatSix",
    "mairlHeatSeven",
)


def _finite(value: object, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _mean_suboptimality(candidate: Mapping[str, object]) -> float:
    statistics = candidate.get("downstream_suboptimality", candidate.get("stackelberg_suboptimality"))
    if not isinstance(statistics, Mapping):
        raise ValueError("candidate statistics must be a mapping")
    mean = _finite(statistics["mean"], "mean Stackelberg suboptimality")
    if mean < 0:
        raise ValueError("mean Stackelberg suboptimality must be nonnegative")
    return mean


def _candidate_key(candidate: Mapping[str, object]) -> tuple[int, float]:
    return (int(candidate["warmup_budget"]), _finite(candidate["lambda_information_multiplier"], "lambda multiplier"))


def _cell_path(left: float, right: float, row: int) -> str:
    return f"(axis cs:{left:.8g},{row - 0.43:.8g}) rectangle (axis cs:{right:.8g},{row + 0.43:.8g})"


def _heatmap_cells(candidates: Mapping[tuple[int, float], Mapping[str, object]],
        *, budgets: tuple[int, ...], multipliers: tuple[float, ...], x_boundaries: tuple[float, ...],
        frozen: frozenset[tuple[int, float]], color_min: float, color_max: float) -> str:  # fmt: skip
    fills: list[str] = []
    outlines: list[str] = []
    for row, budget in enumerate(budgets, start=1):
        for column, multiplier in enumerate(multipliers):
            key = (budget, multiplier)
            candidate = candidates.get(key)
            if candidate is None:
                continue
            cell = _cell_path(x_boundaries[column], x_boundaries[column + 1], row)
            mean = _mean_suboptimality(candidate)
            normalized = (mean - color_min) / (color_max - color_min)
            color_index = min(6, max(0, int(np.floor(7 * normalized))))
            fills.append(
                rf"\path[draw=white, fill={_HEAT_COLORS[color_index]}, "
                rf"line width=0.45pt] {cell};"
            )
            if key in frozen:
                outlines.append(rf"\draw[draw=mairlInk, line width=1.1pt] {cell};")
            difference = candidate.get("paired_difference_from_frozen")
            if isinstance(difference, Mapping):
                lower = _finite(difference["ci95_lower"], "paired CI lower endpoint")
                upper = _finite(difference["ci95_upper"], "paired CI upper endpoint")
                contains_zero = bool(difference["contains_zero"])
                if contains_zero != (lower <= 0.0 <= upper):
                    raise ValueError("paired CI zero-containment flag is inconsistent")
    return "\n".join((*fills, *outlines))


def render_lobeke_tuning(payload: Mapping[str, Any]) -> str:
    """Render the protected joint tuning grid as a single heatmap."""

    schema_version = int(payload.get("schema_version", 0))
    if schema_version not in {1, 2, 3}:
        raise ValueError("unsupported Lobeke tuning-data schema")
    if schema_version == 1:
        warm = payload["selection"]["warmup"]
        no_warm = payload["selection"]["no_warmup"]
        all_candidates = tuple(no_warm["candidates"]) + tuple(warm["candidates"])
        selected_items = (no_warm["selected"], warm["selected"])
    elif schema_version == 2:
        all_candidates = tuple(payload["candidates"])
        selected_items = tuple(payload.get("selected", ()))
    else:
        all_candidates = tuple(payload["candidates"])
        selected_items = tuple(payload["frozen_protocols"])
        if float(payload["mle_ridge"]) != 0.0:
            raise ValueError("schema-3 tuning data must use the unpenalized estimator")
    if not all_candidates:
        raise ValueError("Lobeke tuning requires at least one candidate")

    candidates: dict[tuple[int, float], Mapping[str, object]] = {}
    for candidate in all_candidates:
        key = _candidate_key(candidate)
        if key in candidates:
            raise ValueError(f"duplicate tuning candidate: {key}")
        candidates[key] = candidate

    budgets = tuple(int(value) for value in payload.get("warmup_budgets", sorted({key[0] for key in candidates})))
    multipliers = tuple(
        float(value) for value in payload.get("lambda_information_multipliers", sorted({key[1] for key in candidates}))
    )
    if len(budgets) != len(set(budgets)):
        raise ValueError("warm-up budgets must be unique")
    if (
        len(multipliers) != len(set(multipliers))
        or any(value <= 0.0 for value in multipliers)
        or any(right <= left for left, right in pairwise(multipliers))
    ):
        raise ValueError("lambda multipliers must be unique, positive, and increasing")
    frozen = frozenset(_candidate_key(item) for item in selected_items)
    missing_selected = frozen.difference(candidates)
    if missing_selected:
        raise ValueError(f"frozen tuning candidates are absent: {sorted(missing_selected)}")
    declared_candidates = {(budget, multiplier) for budget in budgets for multiplier in multipliers}
    unexpected = candidates.keys() - declared_candidates
    if unexpected:
        raise ValueError(f"tuning candidates fall outside the declared grid: {sorted(unexpected)}")

    if schema_version == 3:
        if len(frozen) != 2:
            raise ValueError("schema-3 tuning data require exactly two frozen protocols")
        if int(payload["planned_candidates"]) != len(declared_candidates):
            raise ValueError("schema-3 planned-candidate count is inconsistent")
        if int(payload["available_candidates"]) != len(candidates):
            raise ValueError("schema-3 available-candidate count is inconsistent")
        for key, candidate in candidates.items():
            difference = candidate.get("paired_difference_from_frozen")
            if not isinstance(difference, Mapping):
                raise ValueError(f"schema-3 candidate has no paired interval: {key}")
            reference = difference.get("reference")
            if not isinstance(reference, Mapping):
                raise ValueError(f"schema-3 candidate has no frozen reference: {key}")
            expected_reference = (0, 0.03) if key[0] == 0 else (200, 0.003)
            if _candidate_key(reference) != expected_reference:
                raise ValueError(f"schema-3 candidate has the wrong frozen reference: {key}")
        color_scale = payload["color_scale"]
        color_min = _finite(color_scale["min"], "color-scale minimum")
        color_max = _finite(color_scale["max"], "color-scale maximum")
        if int(color_scale["bins"]) != len(_HEAT_COLORS):
            raise ValueError("color-scale bin count does not match the renderer")
        x_positions = tuple(float(index) for index in range(1, len(multipliers) + 1))
        x_boundaries = tuple(float(index) + 0.5 for index in range(len(multipliers) + 1))
        xmode = ""
        colorbar_ticks = r"""    ytick={0,0.0001,0.0002,0.0003,0.0004},
    yticklabels={$0$,$1$,$2$,$3$,$4$},
"""
    else:
        means = np.asarray([_mean_suboptimality(candidate) for candidate in all_candidates])
        color_min = float(np.floor(np.min(means) / 1.0e-5) * 1.0e-5)
        color_max = float(np.ceil(np.max(means) / 1.0e-5) * 1.0e-5)
        x_positions = tuple(float(index) for index in range(1, len(multipliers) + 1))
        x_boundaries = (0.53, *(float(index) + 1.47 for index in range(len(multipliers))))
        xmode = ""
        colorbar_ticks = "    yticklabel={\\pgfmathprintnumber[sci,precision=1]{\\tick}},\n"
    if color_min == color_max:
        color_max = color_min + 1.0e-5
    if color_min < 0.0 or color_max <= color_min:
        raise ValueError("color scale must have nonnegative increasing bounds")
    if schema_version == 3 and any(_mean_suboptimality(candidate) > color_max for candidate in all_candidates):
        raise ValueError("schema-3 candidate exceeds the declared color scale")

    cells = _heatmap_cells(
        candidates,
        budgets=budgets,
        multipliers=multipliers,
        x_boundaries=x_boundaries,
        frozen=frozen,
        color_min=color_min,
        color_max=color_max,
    )
    x_ticks = ",".join(f"{value:g}" for value in x_positions)
    x_labels = ",".join(f"{multiplier:g}" for multiplier in multipliers)
    y_ticks = ",".join(str(index) for index in range(1, len(budgets) + 1))
    y_labels = ",".join(str(budget) for budget in budgets)
    selection_note = (
        ""
        if schema_version == 3
        else r"""\node[anchor=south east, font=\scriptsize, text=mairlMuted]
  at (axis description cs:1,1.025)
  {outlined: selected};"""
        if frozen
        else ""
    )
    return rf"""% Generated by reproducibility.figures.recipes.lobeke_tuning; do not edit.
\begin{{tikzpicture}}
\begin{{axis}}[
  mairl axis,
  width=0.84\textwidth,
  height=5.35cm,
{xmode}  xmin={x_boundaries[0]:.8g}, xmax={x_boundaries[-1]:.8g},
  ymin=0.5, ymax={len(budgets) + 0.5},
  xtick={{{x_ticks}}},
  xticklabels={{{x_labels}}},
  ytick={{{y_ticks}}},
  yticklabels={{{y_labels}}},
  xlabel={{Information multiplier $\lambda/\eta^2$}},
  ylabel={{Warm-up samples}},
  axis line style={{draw=none}},
  tick style={{draw=none}},
  grid=none,
  colormap name=mairl heat,
  colormap access=map,
  point meta min={color_min:.8g},
  point meta max={color_max:.8g},
  colorbar,
  colorbar style={{
    width=0.20cm,
{colorbar_ticks}    scaled y ticks=false,
    yticklabel style={{font=\tiny}},
    ylabel={{Mean transfer $\operatorname{{SubOpt}}$}},
    ylabel style={{font=\scriptsize}},
    title={{\scriptsize $\times 10^{{-4}}$}},
    title style={{at={{(0.5,1.02)}}, anchor=south}}
  }}
]
\addplot[draw=none, only marks, mark=none, point meta=explicit] coordinates
  {{({x_positions[0]:.8g},1) [{color_min:.8g}] ({x_positions[0]:.8g},1) [{color_max:.8g}]}};
{cells}
{selection_note}
\end{{axis}}
\end{{tikzpicture}}
"""
