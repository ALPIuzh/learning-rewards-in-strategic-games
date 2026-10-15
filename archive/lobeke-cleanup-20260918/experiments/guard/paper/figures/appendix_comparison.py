"""Render the main Lobeke comparison figure: curves, not bars, across all methods.

Usage::

    python -m experiments.guard.paper.figures.appendix_comparison --run-dir <path> [--output ...]

``--run-dir`` is a completed run directory under ``run.root`` (e.g.
``artifacts/runs/guard-lobeke-paper/<run_id>``), the same one printed by
``experiments/guard/paper/run.py`` as ``run_dir=...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_corrected_run_traces
from experiments.guard.paper.figures.data import (
    METHOD_STYLE,
    METRIC_LABELS,
    SeedTrace,
    infer_warmup_budget,
    mean_interval,
)
from experiments.guard.paper.figures.style import GRID, PAPER_WIDTH, clean_axis, paper_style, save_paper_figure
from mairl_exp.evaluation.mpl import Line2D, plt

_METRICS = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")


def render_comparison(traces: dict[tuple[str, bool], list[SeedTrace]], *, warmup_budget: int,
        confidence_level: float, band_alpha: float, dpi: int, output_path: Path) -> None:  # fmt: skip
    with paper_style():
        figure, axes = plt.subplots(
            1,
            len(_METRICS),
            figsize=(PAPER_WIDTH, 2.65),
            gridspec_kw={"width_ratios": (1.18, 1.0, 1.0), "wspace": 0.34},
        )
        total_samples = max(int(item.checkpoints[-1]) for method_traces in traces.values() for item in method_traces)

        for axis, metric in zip(axes, _METRICS, strict=True):
            for key, style in METHOD_STYLE.items():
                method_traces = traces.get(key)
                if not method_traces:
                    continue
                checkpoints = method_traces[0].checkpoints
                if any(not np.array_equal(item.checkpoints, checkpoints) for item in method_traces):
                    raise ValueError(f"inconsistent checkpoints across seeds for {key}")
                values = np.stack([item.metrics[metric] for item in method_traces])
                mean, lower, upper = mean_interval(values, confidence_level)
                axis.plot(
                    checkpoints,
                    np.maximum(mean, 1e-9),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                )
                axis.fill_between(
                    checkpoints,
                    np.maximum(lower, 1e-9),
                    np.maximum(upper, 1e-9),
                    color=style["color"],
                    alpha=band_alpha * float(style["alpha"]),
                    linewidth=0,
                    zorder=int(style["zorder"]) - 1,
                )
            axis.set(
                xlim=(0, total_samples), xlabel="Observed poacher actions", ylabel=METRIC_LABELS[metric], yscale="log"
            )
            axis.axvspan(0, warmup_budget, color="#DDEAF4", alpha=0.48, zorder=0)
            axis.axvline(warmup_budget, color=GRID, linestyle="-", linewidth=0.8, zorder=1)
            clean_axis(axis)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=style["label"],
            )
            for key, style in METHOD_STYLE.items()
            if key in traces
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.02),
            columnspacing=1.25,
            handlelength=2.5,
        )
        figure.subplots_adjust(left=0.09, right=0.995, top=0.98, bottom=0.28)
        save_paper_figure(figure, output_path, dpi=dpi)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmup-budget", type=int, default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--band-alpha", type=float, default=0.12)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    traces = load_corrected_run_traces(args.run_dir, args.correction_dir)
    warmup_budget = infer_warmup_budget(traces) if args.warmup_budget is None else args.warmup_budget

    output_path = args.output or (args.run_dir / "lobeke_paper_comparison.png")
    render_comparison(
        traces,
        warmup_budget=warmup_budget,
        confidence_level=args.confidence_level,
        band_alpha=args.band_alpha,
        dpi=args.dpi,
        output_path=output_path,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
