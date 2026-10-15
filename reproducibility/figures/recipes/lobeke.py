"""TikZ recipes for the two main Lobeke figures.

This module intentionally knows nothing about GUARD or run directories. The
experiment adapter loads and validates traces, summaries, policy grids, and the
ranger-base position before calling these pure presentation functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reproducibility.figures.policy_grid import (
    PolicyGridCell,
    PolicyGridRow,
    render_paired_policy_grid,
    render_policy_grid,
)
from reproducibility.figures.statistics import mean_interval

MethodKey = tuple[str, bool]

_LEGEND_SOLID = r"\rule[0.5ex]{0.28cm}{0.55pt}"
_LEGEND_DASHED = (
    r"\raisebox{0.5ex}{\hbox{\rule{0.11cm}{0.55pt}\hspace{0.04cm}%"
    "\n      "
    r"\rule{0.11cm}{0.55pt}}}"
)
_LEGEND_ALGORITHMS = (("Active IRL", "mairlActive"), ("Fixed random", "mairlFixed"), ("Online random", "mairlOnline"))


def _swatch(color: str) -> str:
    return rf"\textcolor{{{color}}}{{{_LEGEND_SOLID}}}"


# Stacked form, used by the main-text convergence figure.
ACTIVE_DESIGN_LEGEND_TABLE = (
    r"  \begin{tabular}{@{}l@{\hspace{5pt}}l@{}}"
    "\n    "
    r"\textbf{Algorithms} & \textbf{Sampling design} \\[-0.3ex]"
    "\n    "
    rf"{_swatch(_LEGEND_ALGORITHMS[0][1])}\;{_LEGEND_ALGORITHMS[0][0]} &"
    "\n    "
    rf"{_LEGEND_SOLID}\;w/ warm-up \\"
    "\n    "
    rf"{_swatch(_LEGEND_ALGORITHMS[1][1])}\;{_LEGEND_ALGORITHMS[1][0]} &"
    "\n    "
    rf"{_LEGEND_DASHED}\;w/o warm-up \\"
    "\n    "
    rf"{_swatch(_LEGEND_ALGORITHMS[2][1])}\;{_LEGEND_ALGORITHMS[2][0]} & \\"
    "\n  "
    r"\end{tabular}"
)

# Inline form: one row of algorithms over one row of sampling designs.
ACTIVE_DESIGN_LEGEND_INLINE = (
    r"\begin{tabular}{@{}c@{}}"
    + "\n"
    + "\\quad ".join(rf"{_swatch(color)}\;{label}" for label, color in _LEGEND_ALGORITHMS)
    + r" \\[-0.3ex]"
    + "\n"
    + rf"{_LEGEND_SOLID}\;w/ warm-up\quad {_LEGEND_DASHED}\;w/o warm-up \\"
    + "\n"
    + r"\end{tabular}"
)


CURVES: dict[MethodKey, tuple[str, str, str]] = {
    ("active", True): ("mairlActive", "mairl warm", "Active IRL"),
    ("active", False): ("mairlActive", "mairl no warm", "Active IRL"),
    ("fixed_random", True): ("mairlFixed", "mairl warm", "Fixed random"),
    ("fixed_random", False): ("mairlFixed", "mairl no warm", "Fixed random"),
    ("random_each_round", True): ("mairlOnline", "mairl warm", "Online random"),
    ("random_each_round", False): ("mairlOnline", "mairl no warm", "Online random"),
    ("fixed_expert_mle", False): ("mairlExpert", "mairl warm", "Fixed-expert MLE"),
    ("zero_sum_ols", False): ("mairlZeroSum", "mairl warm", "Zero-sum OLS"),
    ("follower_ols", False): ("mairlFollower", "mairl warm", "Follower-only OLS"),
}

TRANSFER_BASELINES: dict[MethodKey, tuple[str, str, str]] = {
    ("fixed_expert_mle", False): ("expert", "Fixed-expert MLE", "mairlExpert"),
    ("zero_sum_ols", False): ("liao", "Zero-sum OLS", "mairlZeroSum"),
    ("follower_ols", False): ("follower", "Follower-only OLS", "mairlFollower"),
    ("fixed_random", True): ("fixed", "Fixed random", "mairlFixed"),
    ("fixed_random", False): ("fixed_nowarm", "Fixed random", "mairlFixed"),
    ("random_each_round", True): ("random", "Online random", "mairlOnline"),
    ("random_each_round", False): ("random_nowarm", "Online random", "mairlOnline"),
}

MAIN_TRANSFER_BASELINE: MethodKey = ("fixed_expert_mle", False)


def coordinates(x: np.ndarray, y: np.ndarray) -> str:
    """Serialize paired arrays as compact PGFPlots coordinates."""

    return " ".join(f"({float(x_value):.8g},{float(y_value):.8g})" for x_value, y_value in zip(x, y, strict=True))


def _curve_tex(key: MethodKey, seed_traces: list[Any], *, confidence_level: float,
        y_min: float, add_legend: bool = True) -> str:  # fmt: skip
    color, line_style, label = CURVES[key]
    checkpoints = np.asarray(seed_traces[0].checkpoints, dtype=np.float64)
    values = np.asarray([trace.metrics["stackelberg_suboptimality"] for trace in seed_traces], dtype=np.float64)
    mean, lower, upper = mean_interval(values, confidence_level)
    lower = np.maximum(lower, y_min)
    upper = np.maximum(upper, y_min)
    slug = f"{key[0]}_{'warm' if key[1] else 'plain'}"
    forget = "" if add_legend else ", forget plot"
    commands = [
        rf"\addplot[draw=none, name path={slug}Upper, forget plot] coordinates "
        rf"{{{coordinates(checkpoints, upper)}}};",
        rf"\addplot[draw=none, name path={slug}Lower, forget plot] coordinates "
        rf"{{{coordinates(checkpoints, lower)}}};",
        rf"\addplot[{color}, mairl ci, forget plot] "
        rf"fill between[of={slug}Upper and {slug}Lower];",
        rf"\addplot[draw={color}, {line_style}, line width=0.65pt{forget}] coordinates "
        rf"{{{coordinates(checkpoints, mean)}}};",
    ]
    if add_legend:
        commands.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(commands)


def render_convergence_figure(traces: dict[MethodKey, list[Any]],
        *, confidence_level: float, warmup_budget: int) -> str:  # fmt: skip
    """Render the main source-task convergence fragment."""

    y_min = 1e-4
    y_max = 1.5e-1
    mle_keys = (
        ("active", True),
        ("active", False),
        ("fixed_random", True),
        ("fixed_random", False),
        ("random_each_round", True),
        ("random_each_round", False),
    )
    fixed_expert_keys = (("fixed_expert_mle", False), ("zero_sum_ols", False), ("follower_ols", False))
    missing = [key for key in (*mle_keys, *fixed_expert_keys) if key not in traces]
    if missing:
        raise ValueError(f"missing source-convergence protocols: {missing}")
    total_samples = max(int(items[0].checkpoints[-1]) for items in traces.values())
    if not 0 <= warmup_budget <= total_samples:
        raise ValueError("warmup_budget must lie within the plotted sample horizon")
    common_axis = rf"""
mairl axis,
width=0.505\textwidth,
height=3.85cm,
xmin=0, xmax={total_samples},
ymin={y_min:.8g}, ymax={y_max:.8g},
ymode=log,
y tick label style={{font=\tiny, text=mairlMuted}},
xtick={{0,400,800,1200,1600}},
xlabel={{Number of observed poacher actions}},
ytick style={{draw=none}},
minor y tick style={{draw=none}},
legend style={{draw=none, font=\tiny, cells={{anchor=west}}, row sep=-1pt}}
""".strip()
    mle_curves = "\n".join(
        _curve_tex(key, traces[key], confidence_level=confidence_level, y_min=y_min, add_legend=False)
        for key in mle_keys
    )
    fixed_expert_curves = "\n".join(
        _curve_tex(key, traces[key], confidence_level=confidence_level, y_min=y_min) for key in fixed_expert_keys
    )
    return rf"""% Generated by reproducibility.figures.recipes.lobeke; do not edit.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=0.85cm, ylabels at=edge left}},
  {common_axis}
]
\nextgroupplot[
  title={{(a) Active methods}},
  ylabel={{${{\operatorname{{SubOpt}}(\hat\mu)}}$}}
]
\path[fill=mairlActive!5] (axis cs:0,{y_min:.8g}) rectangle
  (axis cs:{warmup_budget},{y_max:.8g});
\draw[mairlGrid, densely dotted, line width=0.45pt]
  (axis cs:{warmup_budget},{y_min:.8g}) -- (axis cs:{warmup_budget},{y_max:.8g});
\node[rotate=90, font=\tiny, text=mairlActive!75!black]
  at (axis description cs:{warmup_budget / (2 * total_samples):.5f},0.20) {{Warm-up}};
{mle_curves}
\node[anchor=north east, fill=white, fill opacity=0.92, text opacity=1,
      inner sep=1.2pt, font=\tiny]
  at (axis description cs:0.985,0.985) {{
{ACTIVE_DESIGN_LEGEND_TABLE}
}};
\nextgroupplot[
  title={{(b) Fixed-expert methods}},
  yticklabels={{}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east, legend columns=1,
                 fill=white, fill opacity=0.90, text opacity=1, draw=none,
                 inner sep=1.2pt, font=\tiny, cells={{anchor=west}},
                 column sep=3pt, row sep=-1pt}}
]
{fixed_expert_curves}
\end{{groupplot}}
\end{{tikzpicture}}
"""


def _score_tex(result: dict[str, float]) -> str:
    mean = float(result["mean"])
    if not np.isfinite(mean):
        raise ValueError("downstream mean SubOpt must be finite")
    if mean == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(mean))))
    return rf"{mean / 10.0**exponent:.1f}\!\times\!10^{{{exponent}}}"


def render_transfer_figure(summary: dict[str, Any], grids: dict[str, np.ndarray],
        *, base_position: tuple[float, float], baseline_key: MethodKey) -> str:  # fmt: skip
    """Render the main downstream-transfer fragment from validated derived data."""

    active = summary["methods"]["active"]
    try:
        baseline_slug, baseline_label, baseline_color = TRANSFER_BASELINES[baseline_key]
    except KeyError as error:
        raise ValueError(f"unsupported transfer baseline {baseline_key!r}") from error
    baseline = summary["methods"][baseline_slug]
    active_results = [active["reference"], *active["tasks"]]
    baseline_results = [baseline["reference"], *baseline["tasks"]]
    rows = (
        PolicyGridRow("Ground truth", "oracle", "mairlMuted"),
        PolicyGridRow("Active IRL", "active", "mairlActive", tuple(_score_tex(result) for result in active_results)),
        PolicyGridRow(
            baseline_label, baseline_slug, baseline_color, tuple(_score_tex(result) for result in baseline_results)
        ),
    )
    return render_policy_grid(
        rows, grids, base_position=base_position, generated_by="reproducibility.figures.recipes.lobeke"
    )


REFERENCE_ROW = ("Ground truth", "oracle", "mairlMuted")
ACTIVE_DESIGN_ROWS = (
    ("Active IRL", "mairlActive", "active", "active_nowarm"),
    ("Fixed random", "mairlFixed", "fixed", "fixed_nowarm"),
    ("Online random", "mairlOnline", "random", "random_nowarm"),
)
FIXED_EXPERT_ROWS = (
    ("Fixed-expert MLE", "expert", "mairlExpert"),
    ("Zero-sum OLS", "liao", "mairlZeroSum"),
    ("Follower-only OLS", "follower", "mairlFollower"),
)


def _method_scores(summary: dict[str, Any], slug: str) -> tuple[str, ...]:
    method = summary["methods"][slug]
    return tuple(_score_tex(result) for result in (method["reference"], *method["tasks"]))


def _legend_line(groups: tuple[tuple[tuple[tuple[str, str], ...], str | None], ...],
        *, separator: str = r"\enspace") -> str:  # fmt: skip
    """Name coloured rows at document size, below a self-scaling grid.

    The appendix key carries six names and only just fits on one line, so it
    keeps the tighter separator; shorter keys can afford more air.
    """

    tag = r"\fontsize{7.2}{8.6}\selectfont"
    parts = []
    for entries, label in groups:
        names = ", ".join(rf"\mbox{{\textcolor{{{color}}}{{{name}}}}}" for name, color in entries)
        parts.append(names if label is None else rf"{names}\,{{{tag} ({label})}}")
    body = f"{separator} ".join(parts)
    return "{\\fontsize{7.5}{9}\\selectfont\\centering\n" + body + "\\par}\n"


def _reference_entry() -> tuple[str, str]:
    label, _slug, color = REFERENCE_ROW
    return label, color


def _downstream_transfer_key() -> str:
    return _legend_line(
        (
            ((_reference_entry(),), None),
            (tuple((label, color) for label, color, _warm, _plain in ACTIVE_DESIGN_ROWS), "Active"),
            (tuple((label, color) for label, _slug, color in FIXED_EXPERT_ROWS), "Fixed"),
        )
    )


def _grid_with_key(grid: str, key: str) -> str:
    """Place a row key under a self-scaling grid.

    Only the gap above the key is set here. The space below it is the
    caption skip, which ICLR requires to be one line.
    """

    return f"{grid}\\par\\vspace{{0.3em}}\n{key}"


def render_compact_transfer_figure(summary: dict[str, Any], grids: dict[str, np.ndarray],
        *, base_position: tuple[float, float], baseline_key: MethodKey) -> str:  # fmt: skip
    """Render the main transfer grid as one paired row instead of three stacked rows.

    Active IRL and its baseline sit side by side inside each task column, so the
    figure keeps every map at the shared size while costing two rows rather than
    three.
    """

    try:
        baseline_slug, baseline_label, baseline_color = TRANSFER_BASELINES[baseline_key]
    except KeyError as error:
        raise ValueError(f"unsupported transfer baseline {baseline_key!r}") from error
    rows = (
        (
            PolicyGridCell("active", "mairlActive", scores=_method_scores(summary, "active")),
            PolicyGridCell(baseline_slug, baseline_color, scores=_method_scores(summary, baseline_slug)),
        ),
    )
    grid = render_paired_policy_grid(
        PolicyGridCell(REFERENCE_ROW[1], REFERENCE_ROW[2]),
        rows,
        grids,
        base_position=base_position,
        generated_by="reproducibility.figures.recipes.lobeke",
    )
    key = _legend_line(
        (
            ((_reference_entry(),), None),
            ((("Active IRL", "mairlActive"),), "Active"),
            (((baseline_label, baseline_color),), "Fixed"),
        ),
        separator=r"\hspace{2em}",
    )
    return _grid_with_key(grid, key)


def write_compact_transfer_figure(data_dir: Path, output_path: Path,
        *, base_position: tuple[float, float], baseline_key: MethodKey) -> None:  # fmt: skip
    """Write the compact variant of the main downstream-transfer grid."""

    summary = json.loads((data_dir / "summary.json").read_text())
    with np.load(data_dir / "policy_grids.npz") as arrays:
        grids = {name: arrays[name].copy() for name in arrays.files}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_compact_transfer_figure(summary, grids, base_position=base_position, baseline_key=baseline_key)
    )


def render_downstream_transfer_figure(summary: dict[str, Any], grids: dict[str, np.ndarray],
        *, base_position: tuple[float, float]) -> str:  # fmt: skip
    """Render the single appendix grid covering both data-access assumptions.

    Every method is shown in its no-warm-up configuration, the only one the
    fixed-expert estimators admit, so all six are directly comparable on the
    reported value. The warm-up variant of each active-design method survives as
    a parenthesised SubOpt value rather than a second, visually indistinguishable
    row of maps.
    """

    reference = PolicyGridCell(REFERENCE_ROW[1], REFERENCE_ROW[2])
    rows = tuple(
        (
            PolicyGridCell(
                plain_slug, color, scores=_method_scores(summary, plain_slug), notes=_method_scores(summary, warm_slug)
            ),
            PolicyGridCell(fixed_slug, fixed_color, scores=_method_scores(summary, fixed_slug)),
        )
        for (_label, color, warm_slug, plain_slug), (_fixed_label, fixed_slug, fixed_color) in zip(
            ACTIVE_DESIGN_ROWS, FIXED_EXPERT_ROWS, strict=True
        )
    )
    grid = render_paired_policy_grid(
        reference, rows, grids, base_position=base_position, generated_by="reproducibility.figures.recipes.lobeke"
    )
    return _grid_with_key(grid, _downstream_transfer_key())


def write_downstream_transfer_figure(data_dir: Path, output_path: Path, *, base_position: tuple[float, float]) -> None:
    """Write the merged appendix policy grid."""

    summary = json.loads((data_dir / "summary.json").read_text())
    with np.load(data_dir / "policy_grids.npz") as arrays:
        grids = {name: arrays[name].copy() for name in arrays.files}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_downstream_transfer_figure(summary, grids, base_position=base_position))


def write_convergence_figure(traces: dict[MethodKey, list[Any]], output_path: Path,
        *, confidence_level: float, warmup_budget: int) -> None:  # fmt: skip
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_convergence_figure(traces, confidence_level=confidence_level, warmup_budget=warmup_budget)
    )


def write_transfer_figure(data_dir: Path, output_path: Path,
        *, base_position: tuple[float, float], baseline_key: MethodKey) -> None:  # fmt: skip
    summary = json.loads((data_dir / "summary.json").read_text())
    with np.load(data_dir / "policy_grids.npz") as arrays:
        grids = {name: arrays[name].copy() for name in arrays.files}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_transfer_figure(summary, grids, base_position=base_position, baseline_key=baseline_key)
    )
