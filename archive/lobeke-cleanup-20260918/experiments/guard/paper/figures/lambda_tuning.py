"""Plot the protected lambda study's selection and confirmation statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from experiments.guard.paper.figures.style import (
    ACTIVE,
    ACTIVE_LIGHT,
    PAPER_WIDTH,
    clean_axis,
    paper_style,
    save_paper_figure,
)
from mairl_exp.evaluation.mpl import plt

_METRICS = (
    ("stackelberg_suboptimality", r"Final $\operatorname{SubOpt}(\hat\mu)$"),
    ("fisher_transfer_error", "Final Fisher-weighted error"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _interval(statistics: dict[str, float | int]) -> tuple[float, float]:
    mean = float(statistics["mean"])
    seeds = int(statistics["seeds"])
    critical = float(student_t.ppf(0.975, df=seeds - 1))
    radius = critical * float(statistics["standard_error"])
    return mean, radius


def render(run_dir: Path, output: Path) -> None:
    selection_path = run_dir / "lambda_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"completed lambda selection not found: {selection_path}")
    payload = json.loads(selection_path.read_text())
    statistics = payload["statistics"]
    multipliers = np.asarray(sorted(float(value) for value in payload["actual_lambda_by_multiplier"]), dtype=np.float64)

    with paper_style():
        figure, axes = plt.subplots(2, 2, figsize=(PAPER_WIDTH, 4.25), sharex=True, constrained_layout=True)
        for column, mode in enumerate(("warmup", "no_warmup")):
            decision = payload["decisions"][mode]
            for row, (metric, label) in enumerate(_METRICS):
                axis = axes[row, column]
                for split, color, marker in (("selection", ACTIVE, "o"), ("confirmation", ACTIVE_LIGHT, "s")):
                    intervals = [_interval(statistics[mode][split][f"{value:g}"][metric]) for value in multipliers]
                    means = np.asarray([item[0] for item in intervals])
                    radii = np.asarray([item[1] for item in intervals])
                    axis.errorbar(
                        multipliers,
                        means,
                        yerr=radii,
                        color=color,
                        marker=marker,
                        markersize=3.2,
                        linewidth=1.0,
                        capsize=1.8,
                        label=split.capitalize(),
                    )
                selected = decision["provisional_multiplier"]
                axis.axvline(selected, color="#17212B", linewidth=0.8, linestyle="--")
                axis.set_xscale("log", base=2)
                axis.set_yscale("log")
                axis.set_ylabel(label)
                clean_axis(axis)
            axes[0, column].set_title("Warm-up" if mode == "warmup" else "No warm-up")
            axes[-1, column].set_xlabel(r"Information multiplier $\lambda/\eta^2$")
            axes[-1, column].set_xticks(multipliers, labels=[f"{value:g}" for value in multipliers])
        axes[0, 0].legend(loc="best")
        save_paper_figure(figure, output)
        plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    render(args.run_dir, args.output)


if __name__ == "__main__":
    main()
