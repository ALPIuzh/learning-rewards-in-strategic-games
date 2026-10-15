"""Full source-task diagnostics from the compact native Lobeke export."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.guard.paper.rewards import validate_evaluation_reward_normalization
from reproducibility.figures.recipes.lobeke import ACTIVE_DESIGN_LEGEND_INLINE, CURVES, coordinates
from reproducibility.figures.statistics import mean_interval

ACTIVE_METHODS = (
    ("active_warm", ("active", True), "Active IRL, w/ warm-up"),
    ("active_no_warm", ("active", False), "Active IRL, w/o warm-up"),
    ("fixed_random_warm", ("fixed_random", True), "Fixed random, w/ warm-up"),
    ("fixed_random_no_warm", ("fixed_random", False), "Fixed random, w/o warm-up"),
    ("online_random_warm", ("random_each_round", True), "Online random, w/ warm-up"),
    ("online_random_no_warm", ("random_each_round", False), "Online random, w/o warm-up"),
)
FIXED_EXPERT_METHODS = (
    ("fixed_expert_mle", ("fixed_expert_mle", False), "Fixed-expert MLE"),
    ("zero_sum_ols", ("zero_sum_ols", False), "Zero-sum OLS"),
    ("follower_ols", ("follower_ols", False), "Follower-only OLS"),
)
METHODS = (*ACTIVE_METHODS, *FIXED_EXPERT_METHODS)


@dataclass(frozen=True)
class LogAxisLimits:
    """Fixed publication bounds for one log-scale diagnostic panel."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 < self.lower < self.upper:
            raise ValueError("log-axis limits must satisfy 0 < lower < upper")


AXIS_LIMITS = {
    "target_suboptimality": LogAxisLimits(5e-5, 5e-2),
    "target_fisher_error": LogAxisLimits(2e-2, 1.3),
    "averaged_transfer_suboptimality": LogAxisLimits(1e-4, 2e-2),
    "averaged_transfer_fisher_error": LogAxisLimits(2e-2, 1.3),
}


def _curve(arrays: np.lib.npyio.NpzFile, *, slug: str, key: tuple[str, bool], label: str, metric: str,
        add_legend: bool, axis_limits: LogAxisLimits | None = None, floor: float | None = None) -> str:  # fmt: skip
    checkpoints = arrays[f"{slug}__checkpoints"]
    values = arrays[f"{slug}__{metric}"]
    if values.ndim != 2 or values.shape[0] != 50 or values.shape[1] != checkpoints.size:
        raise ValueError(f"invalid native Lobeke curve for {slug}/{metric}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(checkpoints)):
        raise ValueError(f"non-finite native Lobeke curve for {slug}/{metric}")
    mean, lower, upper = mean_interval(values, confidence_level=0.95)
    if (axis_limits is None) == (floor is None):
        raise ValueError("specify exactly one of axis_limits or floor")
    if axis_limits is not None:
        interval_upper = float(np.max(upper))
        if interval_upper > axis_limits.upper:
            raise ValueError(
                f"axis clips the upper 95% interval for {slug}/{metric}: "
                f"{interval_upper:.8g} exceeds {axis_limits.upper:.8g}"
            )
        # These metrics are nonnegative. Student-t lower endpoints can be
        # slightly negative at noisy early checkpoints; clip only the displayed
        # band to the explicit log-axis floor.
        mean = np.maximum(mean, axis_limits.lower)
        lower = np.maximum(lower, axis_limits.lower)
        upper = np.maximum(upper, axis_limits.lower)
    else:
        if floor is None or floor <= 0.0:
            raise ValueError("floor must be strictly positive")
        lower = np.maximum(lower, floor)
        upper = np.maximum(upper, floor)
    color, line_style, _ = CURVES[key]
    path = f"{slug}_{metric}"
    forget = "" if add_legend else ", forget plot"
    commands = [
        rf"\addplot[draw=none, name path={path}Upper, forget plot] coordinates "
        rf"{{{coordinates(checkpoints, upper)}}};",
        rf"\addplot[draw=none, name path={path}Lower, forget plot] coordinates "
        rf"{{{coordinates(checkpoints, lower)}}};",
        rf"\addplot[{color}, mairl ci, forget plot] "
        rf"fill between[of={path}Upper and {path}Lower];",
        rf"\addplot[draw={color}, {line_style}, line width=0.55pt{forget}] coordinates "
        rf"{{{coordinates(checkpoints, mean)}}};",
    ]
    if add_legend:
        commands.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(commands)


def render_lobeke_diagnostics(provenance: dict[str, object], provenance_path: Path) -> str:
    """Render target and averaged-transfer performance trajectories."""

    source = provenance.get("source", {})
    source_runs = source.get("runs", [])
    if (
        source.get("result_count") != 450
        or provenance["seed_policy"]["count"] != 50
        or not source_runs
        or sum(int(run.get("result_count", 0)) for run in source_runs) != 450
        or any(run.get("git_dirty") for run in source_runs)
        or {run.get("git_commit") for run in source_runs} != {source.get("git_commit")}
    ):
        raise ValueError("current-protocol Lobeke export is incomplete or unclean")
    if provenance.get("schema_version") != 5:
        raise ValueError("Lobeke diagnostics require the unit-reward export schema")
    try:
        validate_evaluation_reward_normalization(provenance.get("leader_reward_normalization", {}))
    except ValueError as error:
        raise ValueError("Lobeke diagnostics require unit-interval evaluation rewards") from error

    protocol = provenance.get("protocol", {})
    if protocol.get("fixed_expert_mle", {}).get("expert_definition") != ("source_stackelberg_optimum"):
        raise ValueError("native Lobeke export has the wrong MLE expert")
    ridge = protocol.get("fixed_expert_mle", {}).get("ridge")
    if not isinstance(ridge, (int, float)) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Lobeke MLE ridge must be finite and nonnegative")
    active = protocol.get("active", {})
    warmup = active.get("warmup_budget")
    multipliers = [
        active.get(name) for name in ("warm_lambda_information_multiplier", "no_warm_lambda_information_multiplier")
    ]
    if (
        not isinstance(warmup, int)
        or warmup <= 0
        or any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in multipliers)
    ):
        raise ValueError("Lobeke export has invalid Active protocols")
    ols = protocol.get("ols", {})
    if (
        ols.get("follower_reference_action") != 22
        or ols.get("leader_reference_action") != 22
        or ols.get("zero_sum_sampling") != "paired empirical leader and follower actions"
    ):
        raise ValueError("native Lobeke export has an invalid paired OLS protocol")
    task_policy = provenance.get("transfer_task_policy", {})
    if task_policy.get("count") != 10 or not task_policy.get("disjoint_from_tuning"):
        raise ValueError("fixed transfer tasks are missing or overlap tuning")
    data_path = provenance_path.with_name("data.npz")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != provenance["data_sha256"]:
        raise ValueError("native Lobeke derived data failed its checksum")

    with np.load(data_path, allow_pickle=False) as arrays:
        seeds = arrays["active_warm__seeds"]
        if len(np.unique(seeds)) != 50 or not np.array_equal(seeds, provenance["seed_policy"]["seeds"]):
            raise ValueError("native Lobeke seed set is invalid")
        if int(arrays["active_warm__checkpoints"][0]) != warmup:
            raise ValueError("native Lobeke warm-up boundary is invalid")
        metric_curves: dict[tuple[str, str], str] = {}
        for group_name, methods in (("active", ACTIVE_METHODS), ("expert", FIXED_EXPERT_METHODS)):
            for metric in AXIS_LIMITS:
                metric_curves[group_name, metric] = "\n".join(
                    _curve(
                        arrays,
                        slug=slug,
                        key=key,
                        label=label,
                        metric=metric,
                        axis_limits=AXIS_LIMITS[metric],
                        add_legend=(metric == "target_suboptimality" and group_name == "expert"),
                    )
                    for slug, key, label in methods
                )

    target_suboptimality_axis = AXIS_LIMITS["target_suboptimality"]
    target_fisher_axis = AXIS_LIMITS["target_fisher_error"]
    transfer_suboptimality_axis = AXIS_LIMITS["averaged_transfer_suboptimality"]
    transfer_fisher_axis = AXIS_LIMITS["averaged_transfer_fisher_error"]

    return rf"""% Generated by reproducibility.figures.recipes.lobeke_diagnostics; do not edit.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 4, group name=lobekediagnostics,
                horizontal sep=1.70cm, vertical sep=0.80cm}},
  mairl axis,
  width=0.455\textwidth,
  height=3.15cm,
  xmin=0, xmax=1600,
  xtick={{0,400,800,1200,1600}},
  xticklabels={{}},
  ymode=log,
  y tick label style={{font=\tiny, text=mairlMuted}},
  ytick style={{draw=none}},
  minor y tick style={{draw=none}}
]
\nextgroupplot[
  title={{(a) Target SubOpt}},
  ylabel={{${{\operatorname{{SubOpt}}(\hat\mu)}}$}},
  ymin={target_suboptimality_axis.lower:g}, ymax={target_suboptimality_axis.upper:g},
]
\path[fill=mairlActive!5] (axis cs:0,{target_suboptimality_axis.lower:g}) rectangle
  (axis cs:{warmup},{target_suboptimality_axis.upper:g});
\draw[mairlGrid, densely dotted, line width=0.45pt]
  (axis cs:{warmup},{target_suboptimality_axis.lower:g}) --
  (axis cs:{warmup},{target_suboptimality_axis.upper:g});
\node[rotate=90, font=\tiny, text=mairlActive!75!black]
  at (axis description cs:{warmup / 3200:.5f},0.25) {{Warm-up}};
{metric_curves["active", "target_suboptimality"]}
\nextgroupplot[
  title={{(b) Target SubOpt}},
  ymin={target_suboptimality_axis.lower:g}, ymax={target_suboptimality_axis.upper:g},
  legend to name=lobekeFixedExpertDiagnosticsLegend,
  legend style={{legend columns=3, font=\tiny, draw=none, fill=none,
                 cells={{anchor=west}}, row sep=-1pt}},
  legend image code/.code={{\draw[#1] (0cm,0.1ex) -- (0.28cm,0.1ex);}}
]
{metric_curves["expert", "target_suboptimality"]}

\nextgroupplot[
  title={{(c) Target Fisher error}}, ylabel={{Fisher error}},
  ymin={target_fisher_axis.lower:g}, ymax={target_fisher_axis.upper:g}
]
\path[fill=mairlActive!5] (axis cs:0,{target_fisher_axis.lower:g}) rectangle
  (axis cs:{warmup},{target_fisher_axis.upper:g});
\draw[mairlGrid, densely dotted, line width=0.45pt]
  (axis cs:{warmup},{target_fisher_axis.lower:g}) --
  (axis cs:{warmup},{target_fisher_axis.upper:g});
{metric_curves["active", "target_fisher_error"]}
\nextgroupplot[
  title={{(d) Target Fisher error}},
  ymin={target_fisher_axis.lower:g}, ymax={target_fisher_axis.upper:g}
]
{metric_curves["expert", "target_fisher_error"]}

\nextgroupplot[
  title={{(e) SubOpt}},
  yshift=-0.35cm,
  ylabel={{Transfer SubOpt}},
  ymin={transfer_suboptimality_axis.lower:g}, ymax={transfer_suboptimality_axis.upper:g}
]
\path[fill=mairlActive!5] (axis cs:0,{transfer_suboptimality_axis.lower:g}) rectangle
  (axis cs:{warmup},{transfer_suboptimality_axis.upper:g});
\draw[mairlGrid, densely dotted, line width=0.45pt]
  (axis cs:{warmup},{transfer_suboptimality_axis.lower:g}) --
  (axis cs:{warmup},{transfer_suboptimality_axis.upper:g});
{metric_curves["active", "averaged_transfer_suboptimality"]}
\nextgroupplot[
  title={{(f) SubOpt}},
  yshift=-0.35cm,
  ymin={transfer_suboptimality_axis.lower:g}, ymax={transfer_suboptimality_axis.upper:g}
]
{metric_curves["expert", "averaged_transfer_suboptimality"]}

\nextgroupplot[
  title={{(g) Fisher error}},
  yshift=-0.35cm,
  ylabel={{Transfer Fisher}}, xlabel={{Number of observed poacher actions}},
  xticklabels={{0,400,800,1200,1600}},
  ymin={transfer_fisher_axis.lower:g}, ymax={transfer_fisher_axis.upper:g}
]
\path[fill=mairlActive!5] (axis cs:0,{transfer_fisher_axis.lower:g}) rectangle
  (axis cs:{warmup},{transfer_fisher_axis.upper:g});
\draw[mairlGrid, densely dotted, line width=0.45pt]
  (axis cs:{warmup},{transfer_fisher_axis.lower:g}) --
  (axis cs:{warmup},{transfer_fisher_axis.upper:g});
{metric_curves["active", "averaged_transfer_fisher_error"]}
\nextgroupplot[
  title={{(h) Fisher error}},
  yshift=-0.35cm,
  xlabel={{Number of observed poacher actions}},
  xticklabels={{0,400,800,1200,1600}},
  ymin={transfer_fisher_axis.lower:g}, ymax={transfer_fisher_axis.upper:g}
]
{metric_curves["expert", "averaged_transfer_fisher_error"]}
\end{{groupplot}}
\draw[mairlGrid, line width=0.45pt]
  ($ (lobekediagnostics c1r1.north west) + (0,0.91cm) $) --
  ($ (lobekediagnostics c2r1.north east) + (0,0.91cm) $);
\node[font=\scriptsize\bfseries, text=mairlMuted, fill=white, inner xsep=6pt]
  at ($ (lobekediagnostics c1r1.north)!0.5!(lobekediagnostics c2r1.north)
        + (0,0.91cm) $)
  {{Target and estimation diagnostics}};
\node[font=\small\bfseries, text=mairlInk]
  at ($ (lobekediagnostics c1r1.north) + (0,1.55cm) $)
  {{Active methods}};
\node[font=\small\bfseries, text=mairlInk]
  at ($ (lobekediagnostics c2r1.north) + (0,1.55cm) $)
  {{Fixed-expert methods}};
\draw[mairlGrid, line width=0.45pt]
  ($ (lobekediagnostics c1r3.north west) + (0,0.44cm) $) --
  ($ (lobekediagnostics c2r3.north east) + (0,0.44cm) $);
\node[font=\scriptsize\bfseries, text=mairlMuted, fill=white, inner xsep=6pt]
  at ($ (lobekediagnostics c1r3.north)!0.5!(lobekediagnostics c2r3.north)
        + (0,0.44cm) $)
  {{Averaged transfer diagnostics}};
\node[anchor=north] at ($(lobekediagnostics c1r4.south)+(0,-1.30cm)$)
  {{\tiny{ACTIVE_DESIGN_LEGEND_INLINE}}};
\node[anchor=north] at ($(lobekediagnostics c2r4.south)+(0,-1.30cm)$)
  {{\pgfplotslegendfromname{{lobekeFixedExpertDiagnosticsLegend}}}};
\end{{tikzpicture}}
"""
