"""Pure TikZ renderer for the final zero-sum estimator comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import t as student_t

_METHODS = (
    ("active", "Active IRL", "mairlActive"),
    ("zero_sum_ols", "Zero-sum OLS", "mairlZeroSum"),
    ("follower_ols", "Follower-only OLS", "mairlFollower"),
)


def _series(statistics: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(statistics["mean"], dtype=np.float64)
    standard_error = np.asarray(statistics["standard_error"], dtype=np.float64)
    seeds = int(statistics["seeds"])
    if (
        mean.ndim != 1
        or standard_error.shape != mean.shape
        or seeds < 2
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(standard_error))
        or np.any(standard_error < 0)
    ):
        raise ValueError("invalid zero-sum summary series")
    radius = float(student_t.ppf(0.975, df=seeds - 1)) * standard_error
    return mean, mean - radius, mean + radius


def _coordinates(x: np.ndarray, y: np.ndarray, *, floor: float) -> str:
    if x.shape != y.shape:
        raise ValueError("coordinate arrays must have identical shapes")
    return " ".join(f"({x_value:.8g},{max(floor, y_value):.8g})" for x_value, y_value in zip(x, y, strict=True))


def _source_plots(metric: Mapping[str, object], prefix: str, *, legend: bool) -> str:
    commands: list[str] = []
    for method, label, color in _METHODS:
        statistics = metric[method]
        checkpoints = np.asarray(statistics["checkpoints"], dtype=np.float64)
        mean, lower, upper = _series(statistics)
        if checkpoints.shape != mean.shape:
            raise ValueError("source checkpoints and summaries have different lengths")
        upper_name = f"{prefix}-{method}-upper"
        lower_name = f"{prefix}-{method}-lower"
        commands.extend(
            (
                rf"\addplot[name path={upper_name}, draw=none, forget plot] coordinates "
                rf"{{{_coordinates(checkpoints, upper, floor=1.0e-8)}}};",
                rf"\addplot[name path={lower_name}, draw=none, forget plot] coordinates "
                rf"{{{_coordinates(checkpoints, lower, floor=1.0e-8)}}};",
                rf"\addplot[fill={color}, fill opacity=0.09, draw=none, forget plot] "
                rf"fill between[of={upper_name} and {lower_name}];",
                rf"\addplot[draw={color}, line width=0.65pt] coordinates "
                rf"{{{_coordinates(checkpoints, mean, floor=1.0e-8)}}};",
            )
        )
        if legend:
            commands.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(commands)


def _downstream_plots(metric: Mapping[str, object], *, floor: float, legend: bool) -> str:
    commands: list[str] = []
    offsets = (-0.16, 0.0, 0.16)
    task_count: int | None = None
    for (method, label, color), offset in zip(_METHODS, offsets, strict=True):
        mean, lower, upper = _series(metric[method])
        if task_count is None:
            task_count = int(mean.size)
        if mean.size != task_count or task_count < 1:
            raise ValueError("zero-sum downstream methods must share a nonempty task grid")
        x = np.arange(1, task_count + 1, dtype=np.float64) + offset
        for x_value, low, high in zip(x, lower, upper, strict=True):
            commands.append(
                rf"\addplot[draw={color}, line width=0.45pt, forget plot] coordinates "
                rf"{{({x_value:.8g},{max(floor, low):.8g}) "
                rf"({x_value:.8g},{max(floor, high):.8g})}};"
            )
        commands.append(
            rf"\addplot[only marks, draw={color}, fill={color}, mark=*, "
            rf"mark size=1.45pt] coordinates "
            rf"{{{_coordinates(x, mean, floor=floor)}}};"
        )
        if legend:
            commands.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(commands)


def render_zero_sum(payload: Mapping[str, Any]) -> str:
    """Render source estimation and downstream deployment from protected data."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported zero-sum publication-data schema")
    if float(payload.get("confidence_level", 0.0)) != 0.95:
        raise ValueError("zero-sum figure expects 95% confidence intervals")
    source = payload["source"]
    downstream = payload["downstream"]["metrics"]
    fisher_source = _source_plots(source["fixed_leader_fisher_error"], "source-fisher", legend=True)
    policy_source = _source_plots(source["regularized_qre_leader_l1_error"], "source-policy", legend=False)
    downstream_subopt = _downstream_plots(downstream["stackelberg_suboptimality"], floor=1.0e-6, legend=False)
    task_count = len(downstream["stackelberg_suboptimality"]["active"]["mean"])
    task_ticks = ",".join(str(index) for index in range(1, task_count + 1))
    task_labels = ",".join(f"Task {index}" for index in range(1, task_count + 1))
    return rf"""% Generated by reproducibility.figures.recipes.zero_sum; do not edit.
\begin{{minipage}}{{\textwidth}}
\centering
{{\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, group name=zerosum, horizontal sep=1.45cm}},
  mairl log axis,
  y tick label style={{font=\tiny, text=mairlMuted}},
  width=0.445\textwidth,
  height=4.35cm
]
\nextgroupplot[
  title={{(a) Fixed-expert Fisher error}},
  xmin=0, xmax=200,
  ylabel={{Estimation error}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east, font=\tiny, row sep=-1pt}}
]
{fisher_source}
\nextgroupplot[
  title={{(b) Source QRE policy error}},
  xmin=0, xmax=200,
  ylabel={{Leader-policy $\ell_1$ error}}
]
{policy_source}
\end{{groupplot}}
\node[font=\small, text=mairlInk, anchor=north]
  at ($(zerosum c1r1.south)!0.5!(zerosum c2r1.south)+(0,-0.38cm)$)
  {{Number of observed follower actions}};
\end{{tikzpicture}}}}

\vspace{{0.78cm}}

{{\begin{{tikzpicture}}
\begin{{axis}}[
  mairl axis,
  /pgfplots/ymode=log,
  y tick label style={{font=\tiny, text=mairlMuted}},
  width=0.92\textwidth,
  height=4.15cm,
  title={{(c) Transfer to downstream leader objectives}},
  xmin=0.55, xmax={task_count + 0.45:.8g},
  xtick={{{task_ticks}}},
  xticklabels={{{task_labels}}},
  ylabel={{Final $\operatorname{{SubOpt}}(\hat\mu)$}},
  ymin=1.0e-6
]
{downstream_subopt}
\end{{axis}}
\end{{tikzpicture}}}}
\end{{minipage}}
"""
