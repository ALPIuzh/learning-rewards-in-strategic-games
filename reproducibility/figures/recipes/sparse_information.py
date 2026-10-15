"""TikZ renderer for the protected sparse-information mechanism study."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from reproducibility.figures.statistics import mean_interval

METHODS = (
    ("active", "Active IRL", "mairlActive"),
    ("fixed_random", "Fixed random", "mairlFixed"),
    ("random_each_round", "Online random", "mairlOnline"),
)
ACTION_COUNTS = np.asarray([2, 4, 8, 16, 32], dtype=np.int64)
INFORMATION_RATIOS = np.asarray([3, 10, 36, 136, 528], dtype=np.int64)


def _coordinates(x: np.ndarray, y: np.ndarray, *, floor: float = 0.0) -> str:
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("invalid sparse-information coordinates")
    return " ".join(f"({x_value:.8g},{max(floor, y_value):.8g})" for x_value, y_value in zip(x, y, strict=True))


def _curves(x: np.ndarray, values: np.ndarray, *, prefix: str, floor: float, legend: bool) -> str:
    if values.shape != (len(METHODS), 100, x.size) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid sparse-information curve array {prefix}")
    commands: list[str] = []
    for index, (_method, label, color) in enumerate(METHODS):
        mean, lower, upper = mean_interval(values[index], confidence_level=0.95)
        lower = np.maximum(lower, floor)
        upper = np.maximum(upper, floor)
        path = f"{prefix}-{index}"
        commands.extend(
            (
                rf"\addplot[draw=none, name path={path}Upper, forget plot] coordinates "
                rf"{{{_coordinates(x, upper, floor=floor)}}};",
                rf"\addplot[draw=none, name path={path}Lower, forget plot] coordinates "
                rf"{{{_coordinates(x, lower, floor=floor)}}};",
                rf"\addplot[fill={color}, mairl ci, forget plot] "
                rf"fill between[of={path}Upper and {path}Lower];",
                rf"\addplot[draw={color}, line width=0.60pt] coordinates "
                rf"{{{_coordinates(x, mean, floor=floor)}}};",
            )
        )
        if legend:
            commands.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(commands)


def _final_curves(values: np.ndarray, *, prefix: str, floor: float, legend: bool) -> str:
    if values.shape != (ACTION_COUNTS.size, len(METHODS), 100):
        raise ValueError(f"invalid sparse-information final array {prefix}")
    reordered = np.transpose(values, (1, 2, 0))
    return _curves(ACTION_COUNTS.astype(np.float64), reordered, prefix=prefix, floor=floor, legend=legend)


def render_sparse_information(provenance: dict[str, object], provenance_path: Path) -> str:
    """Validate the compact protected export and render its four scientific panels."""

    if int(provenance.get("schema_version", 0)) != 1:
        raise ValueError("unsupported sparse-information data schema")
    if provenance.get("experiment") != "sparse-information protected action-count sweep":
        raise ValueError("unexpected sparse-information experiment")
    if provenance.get("action_counts") != ACTION_COUNTS.tolist():
        raise ValueError("unexpected sparse-information action-count grid")
    if provenance.get("methods") != [method for method, _label, _color in METHODS]:
        raise ValueError("unexpected sparse-information methods")
    if provenance.get("seed_policy") != {"paired": True, "seeds": list(range(72_000, 72_100)), "count": 100}:
        raise ValueError("unexpected sparse-information protected seeds")
    if provenance.get("analytical_ratio_validated") is not True:
        raise ValueError("analytical information ratios were not validated")

    data_path = provenance_path.with_suffix(".npz")
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != provenance["derived_npz_sha256"]:
        raise ValueError("sparse-information derived data failed its checksum")
    with np.load(data_path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["action_counts"], ACTION_COUNTS):
            raise ValueError("sparse-information action counts disagree with provenance")
        if not np.array_equal(arrays["analytical_initial_information_ratio"], INFORMATION_RATIOS):
            raise ValueError("sparse-information analytical information ratios are invalid")
        if not np.array_equal(arrays["seeds"], np.arange(72_000, 72_100)):
            raise ValueError("sparse-information seed array is invalid")
        checkpoints = np.asarray(arrays["evaluation_sample_counts"], dtype=np.float64)
        if not np.array_equal(checkpoints, np.arange(0, 201, 10)):
            raise ValueError("sparse-information checkpoints are invalid")
        subopt_curves = _curves(
            checkpoints, arrays["m16__stackelberg_suboptimality"], prefix="sparse-subopt", floor=0.0, legend=True
        )
        fisher_curves = _curves(
            checkpoints, arrays["m16__fisher_transfer_error"], prefix="sparse-fisher", floor=1.0e-3, legend=False
        )
        final_subopt = _final_curves(
            arrays["final__stackelberg_suboptimality"], prefix="sparse-final-subopt", floor=0.0, legend=False
        )
        final_fisher = _final_curves(
            arrays["final__fisher_transfer_error"], prefix="sparse-final-fisher", floor=1.0e-3, legend=False
        )

    return rf"""% Generated by reproducibility.figures.recipes.sparse_information; do not edit.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 2, horizontal sep=1.70cm, vertical sep=1.55cm}},
  mairl axis,
  width=0.455\textwidth,
  height=4.20cm,
  ytick style={{draw=none}},
  minor y tick style={{draw=none}}
]
\nextgroupplot[
  title={{(a) Target SubOpt at $m=16$}},
  xmin=0, xmax=200,
  ymin=0, ymax=0.14,
  xlabel={{Number of observed follower actions}},
  ylabel={{${{\operatorname{{SubOpt}}(\hat\mu)}}$}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east, legend columns=1,
                 font=\tiny, row sep=-1pt}}
]
{subopt_curves}
\nextgroupplot[
  title={{(b) Target Fisher error at $m=16$}},
  y tick label style={{font=\tiny, text=mairlMuted}},
  xmin=0, xmax=200,
  ymin=3e-2, ymax=1,
  ymode=log,
  xlabel={{Number of observed follower actions}},
  ylabel={{Fisher-weighted error}}
]
{fisher_curves}
\nextgroupplot[
  title={{(c) Target SubOpt after 200 observations}},
  xmin=1, xmax=33,
  xtick={{2,4,8,16,32}},
  ymin=0, ymax=0.14,
  xlabel={{Number of leader actions $m$}},
  ylabel={{${{\operatorname{{SubOpt}}(\hat\mu)}}$}}
]
{final_subopt}
\nextgroupplot[
  title={{(d) Target Fisher error after 200 observations}},
  y tick label style={{font=\tiny, text=mairlMuted}},
  xmin=1, xmax=33,
  xtick={{2,4,8,16,32}},
  ymin=3e-2, ymax=1,
  ymode=log,
  xlabel={{Number of leader actions $m$}},
  ylabel={{Fisher-weighted error}}
]
{final_fisher}
\end{{groupplot}}
\end{{tikzpicture}}
"""
