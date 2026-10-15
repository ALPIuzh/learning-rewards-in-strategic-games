"""Render paper-quality source curves for the corrected OLS protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINAL_DATA = Path("reproducibility/data/lobeke/final_evaluation/data.npz")
CORRECTED_DATA = Path("investigations/warmup_baselines/corrected_ols_curves_v1/data.npz")


def _band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    radius = (
        student_t.ppf(0.975, values.shape[0] - 1)
        * np.std(values, axis=0, ddof=1)
        / np.sqrt(values.shape[0])
    )
    return mean, mean - radius, mean + radius


def render(output_dir: Path) -> dict[str, object]:
    with (
        np.load(FINAL_DATA, allow_pickle=False) as final,
        np.load(CORRECTED_DATA, allow_pickle=False) as corrected,
    ):
        series = {
            "Active IRL, no warm-up": (
                final["active_no_warm__checkpoints"],
                final["active_no_warm__target_suboptimality"],
                final["active_no_warm__target_fisher_error"],
                "#2776B9",
                "--",
                1.3,
            ),
            "Fixed-expert MLE": (
                final["fixed_expert_mle__checkpoints"],
                final["fixed_expert_mle__target_suboptimality"],
                final["fixed_expert_mle__target_fisher_error"],
                "#E07A13",
                "-",
                1.35,
            ),
            "Zero-sum OLS, exact leader π (legacy)": (
                final["zero_sum_ols__checkpoints"],
                final["zero_sum_ols__target_suboptimality"],
                final["zero_sum_ols__target_fisher_error"],
                "#14866D",
                ":",
                1.25,
            ),
            "Zero-sum OLS, paired samples (corrected)": (
                corrected["checkpoints"],
                corrected["paired_actions_ref22__stackelberg_suboptimality"],
                corrected["paired_actions_ref22__fisher_transfer_error"],
                "#735A9E",
                "-",
                1.6,
            ),
            "Follower-only OLS, stable reference": (
                corrected["checkpoints"],
                corrected["follower_only_ref22__stackelberg_suboptimality"],
                corrected["follower_only_ref22__fisher_transfer_error"],
                "#CF4436",
                "-",
                1.25,
            ),
        }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), constrained_layout=True)
    for axis, metric_index, title, ylabel in (
        (axes[0], 1, "(a) Target SubOpt", r"SubOpt$(\hat\mu)$"),
        (axes[1], 2, "(b) Target Fisher error", "Fisher-weighted error"),
    ):
        for label, values in series.items():
            checkpoints, *metrics, color, linestyle, linewidth = values
            mean, lower, upper = _band(metrics[metric_index - 1])
            axis.plot(
                checkpoints,
                mean,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            axis.fill_between(checkpoints, lower, upper, color=color, alpha=0.09, linewidth=0)
        axis.set_yscale("log")
        axis.set(
            title=title,
            xlabel="Number of observed poacher actions",
            ylabel=ylabel,
        )
        axis.grid(True, color="#D8DEE8", linewidth=0.65, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#354154")
    axes[0].text(
        0.03,
        0.05,
        "Corrected zero-sum OLS also observes one sampled leader action\n"
        "per poacher action; the legacy curve inserts the exact leader π.",
        transform=axes[0].transAxes,
        fontsize=7.7,
        color="#4B586A",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "corrected_ols_source_curves.png"
    pdf = output_dir / "corrected_ols_source_curves.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    summary = {
        label: {
            "final_target_suboptimality_mean": float(np.mean(values[1][:, -1])),
            "final_target_fisher_error_mean": float(np.mean(values[2][:, -1])),
        }
        for label, values in series.items()
    }
    (output_dir / "corrected_ols_source_curves.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("investigations/warmup_baselines/figures"),
    )
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
