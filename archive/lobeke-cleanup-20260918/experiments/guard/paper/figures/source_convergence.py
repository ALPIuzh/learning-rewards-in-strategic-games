"""Selective main-text performance figure for the Lobéké experiment.

The appendix comparison contains every metric and warm-up variant.  This figure
keeps only the four scientifically distinct protocols needed in the main text:
Active IRL, specialized zero-sum OLS, follower-only OLS, and uninformed random
exploration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_corrected_run_traces
from experiments.guard.paper.figures.data import (
    infer_warmup_budget,
    mean_interval,
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_in_unit_reward,
)
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.style import (
    ACTIVE,
    FOLLOWER_OLS,
    GRID,
    LIAO,
    PAPER_WIDTH,
    RANDOM_ROUND,
    clean_axis,
    paper_style,
    save_paper_figure,
)
from mairl_exp.evaluation.mpl import plt

_ARMS = (
    (("active", True), "Active IRL", ACTIVE, "-", 1.7, 6),
    (("zero_sum_ols", False), "Zero-sum OLS", LIAO, "-", 1.55, 5),
    (("random_each_round", False), "Random patrols", RANDOM_ROUND, (0, (4, 2)), 1.15, 3),
    (("follower_ols", False), "Follower-only OLS", FOLLOWER_OLS, "-", 1.3, 4),
)


def render(run_dir: Path, *, output_path: Path, correction_dir: Path = DEFAULT_CORRECTION_DIR,
        confidence_level: float = 0.95, warmup_budget: int | None = None, dpi: int = 300) -> None:  # fmt: skip
    traces = load_corrected_run_traces(run_dir, correction_dir)
    runtime = load_runtime(run_dir)
    traces = source_suboptimality_in_unit_reward(
        traces,
        raw_reward_range=float(runtime.metadata["leader_reward_normalization"]["raw_range"]),
        already_unit_reward=run_source_suboptimality_is_unit_reward(run_dir),
    )
    if warmup_budget is None:
        warmup_budget = infer_warmup_budget(traces)
    metric = "stackelberg_suboptimality"
    with paper_style():
        figure, axis = plt.subplots(figsize=(PAPER_WIDTH, 2.45))
        total_samples = 0
        endpoint_data = []
        for key, label, color, linestyle, width, zorder in _ARMS:
            seed_traces = traces[key]
            checkpoints = seed_traces[0].checkpoints
            values = np.stack([trace.metrics[metric] for trace in seed_traces])
            mean, lower, upper = mean_interval(values, confidence_level)
            total_samples = max(total_samples, int(checkpoints[-1]))
            axis.plot(
                checkpoints, np.maximum(mean, 1e-9), color=color, linestyle=linestyle, linewidth=width, zorder=zorder
            )
            axis.fill_between(
                checkpoints,
                np.maximum(lower, 1e-9),
                np.maximum(upper, 1e-9),
                color=color,
                alpha=0.09,
                linewidth=0,
                zorder=zorder - 1,
            )
            endpoint_data.append((float(mean[-1]), label, color))

        axis.axvspan(0, warmup_budget, color="#DDEAF4", alpha=0.55, zorder=0)
        axis.axvline(warmup_budget, color=GRID, linewidth=0.8, zorder=1)
        axis.set(
            xlim=(0, total_samples * 1.26),
            yscale="log",
            xlabel="Observed poacher actions",
            ylabel="Deployment regret\n(lower is better)",
        )
        clean_axis(axis)

        # Direct labels make the comparison readable without legend lookup.
        for value, label, color in endpoint_data:
            axis.annotate(
                label,
                xy=(total_samples, value),
                xytext=(total_samples * 1.035, value),
                ha="left",
                va="center",
                fontsize=7.1,
                fontweight="semibold" if label in {"Active IRL", "Zero-sum OLS"} else "normal",
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
                annotation_clip=False,
            )

        figure.subplots_adjust(left=0.10, right=0.82, top=0.98, bottom=0.20)
        save_paper_figure(figure, output_path, dpi=dpi)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--warmup-budget", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output_path = args.output or (args.run_dir / "lobeke_paper_main_performance.png")
    render(
        args.run_dir,
        output_path=output_path,
        correction_dir=args.correction_dir,
        confidence_level=args.confidence_level,
        warmup_budget=args.warmup_budget,
        dpi=args.dpi,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
