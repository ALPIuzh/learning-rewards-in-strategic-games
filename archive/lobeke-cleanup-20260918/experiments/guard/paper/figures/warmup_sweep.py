"""Render the archived, non-comparable Lobeke warm-up sweep.

This script is retained only to preserve its artifacts.  It is not part of the
paper reproduction target because budgets 0/200 reuse final seeds and tuned
lambda values, whereas budgets 50/100 use different pilot seeds and the default
lambda.  Running it requires an explicit acknowledgement flag.

Two of the four thresholds (0 and the flagship's own 200) are already in the
main run's data and are reused directly; only the other two are freshly
simulated here, checkpointed per-budget so a rerun skips finished budgets.

Usage::

    python -m experiments.guard.paper.figures.warmup_sweep --main-run-dir <path> \
        --acknowledge-noncomparable-protocols
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.application import LobekeZeroSumApplication
from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_corrected_run_traces
from experiments.guard.paper.figures.data import METHOD_STYLE, mean_interval
from experiments.guard.paper.figures.style import ACTIVE, GRID, PAPER_WIDTH, clean_axis, paper_style, save_paper_figure
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.config import RESULTS_DIR as PAPER_RESULTS_DIR
from mairl_exp.evaluation.mpl import plt
from mairl_exp.execution.jobs import StudyJob
from mairl_exp.studies.comparison import BaselineComparisonStudy

WARMUP_BUDGETS = (0, 50, 100, 200)
SEED_START = 100
SEED_COUNT = 50
TOTAL_SAMPLES = 1600
EVALUATION_INTERVAL = 40
RESULTS_DIR = PAPER_RESULTS_DIR / "warmup_sweep"
NUMERICS_CONFIG = REPOSITORY_ROOT / "conf" / "numerics" / "guard_lobeke_paper_numerics.yaml"

_BASELINE_KEYS = (
    ("fixed_random", False),
    ("random_each_round", False),
    ("zero_sum_ols", False),
    ("follower_ols", False),
)


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [warmup-sweep] {message}", flush=True)


def _unit_path(budget: int) -> Path:
    return RESULTS_DIR / f"active_warmup{budget}.npz"


def run_budget(runtime, numerics, budget: int) -> None:
    path = _unit_path(budget)
    if path.exists():
        _log(f"skip (already complete): {path.name}")
        return

    study = BaselineComparisonStudy(
        name=f"warmup-sweep-{budget}",
        seed_start=SEED_START,
        seed_count=SEED_COUNT,
        total_samples=TOTAL_SAMPLES,
        warmup_budget=budget,
        evaluation_interval=EVALUATION_INTERVAL,
        methods=("active",),
        warmup_modes=(True,),
    )
    executor = study.executor(runtime, numerics)
    jobs = tuple(StudyJob("active", True, seed) for seed in range(SEED_START, SEED_START + SEED_COUNT))

    started = time.perf_counter()
    results = executor.run_many(jobs)
    elapsed = time.perf_counter() - started
    _log(f"budget={budget}: {SEED_COUNT} seeds in {elapsed:.1f}s")

    checkpoints = results[0].arrays["evaluation_sample_counts"]
    if any(not np.array_equal(result.arrays["evaluation_sample_counts"], checkpoints) for result in results):
        raise ValueError(f"inconsistent checkpoints across seeds for budget={budget}")
    metric_names = [name for name in results[0].arrays if name.startswith("metric__")]
    arrays = {"checkpoints": checkpoints}
    arrays.update({name: np.stack([result.arrays[name] for result in results]) for name in metric_names})
    np.savez_compressed(path, **arrays)
    _log(f"wrote {path}")


def _load_budget_curve(budget: int, main_traces) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (checkpoints, {metric: (n_seeds, n_checkpoints)}) for one warm-up budget."""

    if budget == 0:
        traces = main_traces[("active", False)]
        checkpoints = traces[0].checkpoints
        metrics = {name: np.stack([trace.metrics[name] for trace in traces]) for name in traces[0].metrics}
        return checkpoints, metrics
    if budget == 200:
        traces = main_traces[("active", True)]
        checkpoints = traces[0].checkpoints
        metrics = {name: np.stack([trace.metrics[name] for trace in traces]) for name in traces[0].metrics}
        return checkpoints, metrics

    with np.load(_unit_path(budget)) as arrays:
        checkpoints = arrays["checkpoints"].copy()
        metrics = {
            name.removeprefix("metric__"): arrays[name].copy() for name in arrays.files if name.startswith("metric__")
        }
    return checkpoints, metrics


def render(main_run_dir: Path, *, confidence_level: float, band_alpha: float,
        dpi: int, correction_dir: Path = DEFAULT_CORRECTION_DIR) -> Path:  # fmt: skip
    main_traces = load_corrected_run_traces(main_run_dir, correction_dir)
    metric = "stackelberg_suboptimality"
    budget_colors = ("#9FC5DF", "#67A4CC", ACTIVE, "#0D4C78")
    final_statistics: list[tuple[int, float, float, float]] = []

    with paper_style():
        figure, axes = plt.subplots(
            1, 2, figsize=(PAPER_WIDTH, 2.55), gridspec_kw={"width_ratios": (1.45, 1.0), "wspace": 0.38}
        )
        trajectory_axis, endpoint_axis = axes
        for budget, color in zip(WARMUP_BUDGETS, budget_colors, strict=True):
            checkpoints, metrics = _load_budget_curve(budget, main_traces)
            mean, lower, upper = mean_interval(metrics[metric], confidence_level)
            trajectory_axis.plot(
                checkpoints,
                np.maximum(mean, 1e-9),
                label=f"{budget} samples",
                color=color,
                linewidth=1.5 if budget == 200 else 1.05,
                zorder=4 if budget == 200 else 3,
            )
            trajectory_axis.fill_between(
                checkpoints,
                np.maximum(lower, 1e-9),
                np.maximum(upper, 1e-9),
                color=color,
                alpha=band_alpha * 0.7,
                linewidth=0,
            )
            final_statistics.append((budget, float(mean[-1]), float(lower[-1]), float(upper[-1])))

        trajectory_axis.set(
            xlabel="Observed poacher actions", ylabel="Deployment regret (lower is better)", yscale="log"
        )
        clean_axis(trajectory_axis)
        trajectory_axis.legend(
            title="Warm-up budget", ncol=2, loc="upper right", title_fontsize=6.8, columnspacing=0.9, handlelength=2.0
        )

        for budget, mean, lower, upper in final_statistics:
            color = budget_colors[WARMUP_BUDGETS.index(budget)]
            endpoint_axis.errorbar(
                budget,
                mean,
                yerr=np.asarray([[mean - lower], [upper - mean]]),
                fmt="o",
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=6.0 if budget == 200 else 5.0,
                capsize=2.0,
                linewidth=1.0,
                zorder=5,
            )

        reference_keys = (("zero_sum_ols", False), ("random_each_round", False), ("follower_ols", False))
        for key in reference_keys:
            style = METHOD_STYLE[key]
            values = np.asarray([trace.metrics[metric][-1] for trace in main_traces[key]])
            reference = float(np.mean(values))
            endpoint_axis.axhline(
                reference, color=style["color"], linestyle=style["linestyle"], linewidth=1.0, alpha=0.72, zorder=1
            )
            endpoint_axis.text(
                205,
                reference,
                str(style["label"]).replace(" · no warm-up", ""),
                ha="left",
                va="center",
                fontsize=6.0,
                color=style["color"],
            )

        endpoint_axis.set(
            xlim=(-10, 285),
            xticks=WARMUP_BUDGETS,
            xlabel="Warm-up samples",
            ylabel="Final regret at 1,600 samples",
            yscale="log",
        )
        endpoint_axis.axvline(200, color=GRID, linewidth=0.8, zorder=0)
        clean_axis(endpoint_axis)

        figure.subplots_adjust(left=0.10, right=0.94, top=0.97, bottom=0.22)
        output_path = RESULTS_DIR.parent / "lobeke_paper_warmup_threshold_sweep.png"
        save_paper_figure(figure, output_path, dpi=dpi)
        plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run-dir", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--band-alpha", type=float, default=0.12)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--acknowledge-noncomparable-protocols", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_noncomparable_protocols:
        raise SystemExit(
            "refusing to render a non-comparable warm-up sweep; pass "
            "--acknowledge-noncomparable-protocols only for artifact inspection"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    numerics = instantiate(OmegaConf.load(NUMERICS_CONFIG))
    runtime = LobekeZeroSumApplication().build(numerics)

    for budget in WARMUP_BUDGETS:
        if budget in (0, 200):
            continue  # already in the main flagship run's data
        run_budget(runtime, numerics, budget)

    output_path = render(
        args.main_run_dir,
        confidence_level=args.confidence_level,
        band_alpha=args.band_alpha,
        dpi=args.dpi,
        correction_dir=args.correction_dir,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
