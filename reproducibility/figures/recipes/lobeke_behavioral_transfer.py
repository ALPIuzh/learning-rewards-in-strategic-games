"""Dense source-relative transfer heatmaps for the Lobeke experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.data import METHOD_SLUGS

PANEL_METHODS = (
    ("active_warm", "Active IRL (warm-up)"),
    ("active_no_warm", "Active IRL (no warm-up)"),
    ("fixed_expert_mle", "Fixed-expert MLE"),
    ("fixed_random_warm", "Fixed random (warm-up)"),
    ("fixed_random_no_warm", "Fixed random (no warm-up)"),
    ("zero_sum_ols", "Zero-sum OLS"),
    ("online_random_warm", "Online random (warm-up)"),
    ("online_random_no_warm", "Online random (no warm-up)"),
    ("follower_ols", "Follower-only OLS"),
)
BUCKET_NAMES = ("low", "medium", "high")


def _heatmap_values(arrays: np.lib.npyio.NpzFile, *, slug: str, metric: str) -> np.ndarray:
    assignments = arrays[f"{metric}_bucket_assignments"]
    values = arrays[f"{slug}__taskwise_transfer_suboptimality"]
    if values.ndim != 3 or values.shape[:2] != (assignments.size, 50):
        raise ValueError(f"invalid dense transfer values for {slug}")
    rows = [values[assignments == bucket].mean(axis=(0, 1)) for bucket in range(len(BUCKET_NAMES))]
    result = np.asarray(rows)
    if not np.all(np.isfinite(result)) or np.min(result) <= 0.0:
        raise ValueError(f"dense transfer heatmap for {slug}/{metric} must be positive")
    return result


def _matrix_table(values: np.ndarray) -> str:
    lines = [r"x y meta\\"]
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            lines.append(rf"{column} {row} {np.log10(values[row, column]):.9g}\\")
    return "\n".join(lines)


def _panel(*, values: np.ndarray, title: str, panel_letter: str, checkpoints: np.ndarray, row: int, column: int,
        meta_min: float, meta_max: float, colorbar_ticks: str, colorbar_labels: str) -> str:  # fmt: skip
    x_tick_labels = ",".join(str(int(value)) for value in checkpoints) if row == 2 else "{}"
    y_tick_labels = ",".join(BUCKET_NAMES) if column == 0 else "{}"
    options = [
        rf"title={{{panel_letter} {title}}}",
        f"xticklabels={{{x_tick_labels}}}",
        f"yticklabels={{{y_tick_labels}}}",
    ]
    options.extend([f"point meta min={meta_min:.9g}", f"point meta max={meta_max:.9g}"])
    if row == 2 and column == 1:
        options.extend(
            [
                "colorbar horizontal",
                rf"""colorbar style={{
      at={{(0.5,-1.02)}}, anchor=north,
      width=0.62\textwidth, height=0.16cm,
      xtick={{{colorbar_ticks}}}, xticklabels={{{colorbar_labels}}},
      xlabel={{Mean transfer $\operatorname{{SubOpt}}(\hat\mu)$}},
      tick label style={{font=\tiny}}, label style={{font=\scriptsize}}
    }}""",
            ]
        )
    option_text = ",\n  ".join(options)
    return rf"""
\nextgroupplot[
  {option_text}
]
\addplot[
  matrix plot*, mesh/cols={values.shape[1]},
  point meta=explicit, colormap name=mairl heat,
  mesh/check=false
] table[meta=meta, row sep=\\] {{
{_matrix_table(values)}
}};
"""


def render_lobeke_behavioral_transfer(provenance: dict[str, object], provenance_path: Path, *, metric: str) -> str:
    """Render all nine methods under TV or source-policy-gap stratification."""

    if metric not in {"tv", "gap"}:
        raise ValueError(f"unsupported behavioral-transfer metric: {metric}")
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported behavioral-transfer export schema")
    task_policy = provenance.get("task_policy", {})
    if (
        not isinstance(task_policy.get("name"), str)
        or task_policy.get("count") != 180
        or not isinstance(task_policy.get("seed"), int)
        or task_policy.get("candidate_pool") != 180
    ):
        raise ValueError("behavioral-transfer figure requires the frozen dense task suite")
    evaluation = provenance.get("evaluation", {})
    if (
        evaluation.get("seed_count") != 50
        or evaluation.get("checkpoints") != [200, 400, 800, 1200, 1600]
        or set(evaluation.get("methods", {})) != set(METHOD_SLUGS.values())
    ):
        raise ValueError("behavioral-transfer figure requires all current protocols")
    distance_metrics = provenance.get("distance_metrics", {})
    counts = distance_metrics.get("bucket_counts", {}).get(metric)
    if not isinstance(counts, list) or len(counts) != 3 or sum(counts) != 180:
        raise ValueError("behavioral-transfer bucket counts are invalid")
    data_path = provenance_path.with_name("data.npz")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != provenance.get("data_sha256"):
        raise ValueError("behavioral-transfer data failed its checksum")

    with np.load(data_path, allow_pickle=False) as arrays:
        checkpoints = arrays["checkpoints"]
        values = {slug: _heatmap_values(arrays, slug=slug, metric=metric) for slug, _ in PANEL_METHODS}
        # The two figures share one scale, so their colors remain directly comparable.
        all_cells = []
        for distance_metric in ("tv", "gap"):
            all_cells.extend(
                _heatmap_values(arrays, slug=slug, metric=distance_metric).ravel() for slug, _ in PANEL_METHODS
            )
        log_cells = np.log10(np.concatenate(all_cells))
    meta_min = float(np.floor(np.min(log_cells) * 2.0) / 2.0)
    meta_max = float(np.ceil(np.max(log_cells) * 2.0) / 2.0)
    integer_ticks = np.arange(np.ceil(meta_min), np.floor(meta_max) + 1, dtype=int)
    if integer_ticks.size == 0:
        integer_ticks = np.asarray([round((meta_min + meta_max) / 2.0)])
    colorbar_ticks = ",".join(str(int(value)) for value in integer_ticks)
    colorbar_labels = ",".join(rf"$10^{{{int(value)}}}$" for value in integer_ticks)
    panels = []
    for index, (slug, title) in enumerate(PANEL_METHODS):
        panels.append(
            _panel(
                values=values[slug],
                title=title,
                panel_letter=f"({chr(ord('a') + index)})",
                checkpoints=checkpoints,
                row=index // 3,
                column=index % 3,
                meta_min=meta_min,
                meta_max=meta_max,
                colorbar_ticks=colorbar_ticks,
                colorbar_labels=colorbar_labels,
            )
        )
    group_name = f"lobekebehavior{metric}"
    return rf"""% Generated by lobeke_behavioral_transfer; do not edit.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=3 by 3, group name={group_name},
                horizontal sep=0.52cm, vertical sep=0.65cm}},
  mairl axis,
  width=0.31\textwidth,
  height=2.4cm,
  xmin=-0.5, xmax=4.5,
  ymin=-0.5, ymax=2.5,
  xtick={{0,1,2,3,4}},
  ytick={{0,1,2}},
  tick label style={{font=\tiny, text=mairlMuted}},
  title style={{font=\scriptsize, text=mairlInk, yshift=-4pt}},
  axis on top,
  grid=none,
  enlargelimits=false
]
{"".join(panels)}
\end{{groupplot}}
\node[font=\footnotesize, text=mairlInk, anchor=south]
  at ($ ({group_name} c1r1.north)!0.5!({group_name} c2r1.north) + (0,0.62cm) $)
  {{General data-collection assumptions}};
\node[font=\footnotesize, text=mairlInk, anchor=south]
  at ($ ({group_name} c3r1.north) + (0,0.62cm) $)
  {{Fixed-expert assumptions}};
\node[font=\scriptsize, text=mairlInk, rotate=90]
  at ($ ({group_name} c1r2.west) + (-1.0cm,0) $)
  {{Transfer-distance bucket}};
\node[font=\scriptsize, text=mairlInk, anchor=north]
  at ($ ({group_name} c1r3.south)!0.5!({group_name} c3r3.south) + (0,-0.38cm) $)
  {{Number of observed poacher actions}};
\end{{tikzpicture}}
"""
