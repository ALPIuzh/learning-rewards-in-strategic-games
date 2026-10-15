"""Plot the coarse warm-up/lambda interaction discovered on paired seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.style import INK, PAPER_WIDTH, paper_style, save_paper_figure
from mairl_exp.evaluation.mpl import Line2D, LogNorm, Rectangle, plt

_PANELS = (
    ("stackelberg_suboptimality", r"Mean final $\operatorname{SubOpt}(\hat\mu)$"),
    ("fisher_transfer_error", "Mean final Fisher-weighted error"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _key(budget: int, multiplier: float) -> str:
    return f"warmup={budget},lambda={multiplier:g}"


def render(run_dir: Path, output: Path) -> None:
    summary_path = run_dir / "joint_discovery.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"completed joint discovery not found: {summary_path}")
    payload = json.loads(summary_path.read_text())
    budgets = np.asarray(payload["warmup_budgets"], dtype=np.int64)
    multipliers = np.asarray(payload["lambda_information_multipliers"], dtype=np.float64)
    statistics = payload["statistics"]
    shortlist = {_key(item["warmup_budget"], item["lambda_information_multiplier"]) for item in payload["shortlist"]}
    raw = payload["raw_eligible_winner"]
    reference = payload["reference_candidate"]

    with paper_style():
        figure, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 3.25))
        for axis, (metric, title) in zip(axes, _PANELS, strict=True):
            values = np.asarray(
                [
                    [statistics[_key(budget, multiplier)][metric]["mean"] for multiplier in multipliers]
                    for budget in budgets
                ],
                dtype=np.float64,
            )
            positive = values[values > 0]
            if not positive.size:
                raise ValueError(f"metric {metric!r} contains no positive mean")
            lower = max(float(positive.min()), np.finfo(np.float64).tiny)
            image = axis.imshow(
                np.maximum(values, lower),
                origin="lower",
                aspect="auto",
                cmap="viridis_r",
                norm=LogNorm(vmin=lower, vmax=float(values.max())),
            )
            for row, budget in enumerate(budgets):
                for column, multiplier in enumerate(multipliers):
                    if _key(int(budget), float(multiplier)) in shortlist:
                        axis.add_patch(
                            Rectangle(
                                (column - 0.46, row - 0.46), 0.92, 0.92, fill=False, edgecolor="white", linewidth=1.4
                            )
                        )
            axis.scatter(
                np.flatnonzero(multipliers == raw["lambda_information_multiplier"]),
                np.flatnonzero(budgets == raw["warmup_budget"]),
                marker="*",
                s=45,
                color=INK,
                zorder=4,
            )
            axis.scatter(
                np.flatnonzero(multipliers == reference["lambda_information_multiplier"]),
                np.flatnonzero(budgets == reference["warmup_budget"]),
                marker="o",
                s=35,
                facecolor="none",
                edgecolor=INK,
                linewidth=1.0,
                zorder=4,
            )
            axis.set_title(title)
            axis.set_xlabel(r"Information multiplier $\lambda/\eta^2$")
            axis.set_ylabel("Warm-up samples")
            axis.set_xticks(range(len(multipliers)), labels=[f"{value:g}" for value in multipliers])
            axis.set_yticks(range(len(budgets)), labels=[f"{value:g}" for value in budgets])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        figure.legend(
            handles=(
                Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, label="Shortlist"),
                Line2D([], [], color=INK, marker="*", linestyle="none", label="Raw winner"),
                Line2D(
                    [],
                    [],
                    color=INK,
                    marker="o",
                    markerfacecolor="none",
                    linestyle="none",
                    label="Current configuration",
                ),
            ),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
        )
        figure.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
        save_paper_figure(figure, output)
        plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    render(args.run_dir, args.output)


if __name__ == "__main__":
    main()
