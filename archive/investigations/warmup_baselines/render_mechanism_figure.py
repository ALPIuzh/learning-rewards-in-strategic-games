"""Render an evidence-first diagnostic figure for the three observed anomalies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.guard.paper.figures.downstream import load_runtime
from mairl.core.math import cumulative_fisher_information, fisher_information

METHOD_COLORS = {
    "active_no_warm": "#2776B9",
    "active_warm": "#72A5D1",
    "zero_sum_ols": "#14866D",
    "zero_sum_paired": "#735A9E",
    "fixed_expert_mle": "#E07A13",
    "follower_ols": "#CF4436",
}
CORRECTED_OLS = Path("investigations/warmup_baselines/corrected_ols_reanalysis/data.npz")
RAW_RUN = Path(
    "artifacts/runs/guard-lobeke-paper/a4b1b2a1cdf886dd"
)
LEGACY_TRANSFER_CACHE = Path(
    "artifacts/derived/guard-lobeke-transfer-diagnostics"
)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    radius = float(student_t.ppf(0.975, values.size - 1) * se)
    return mean, mean - radius, mean + radius


def raw_job_paths(method: str, warm: bool) -> list[Path]:
    manifest = json.loads((RAW_RUN / "manifest.json").read_text())
    entries = [
        entry
        for entry in manifest["expected_jobs"]
        if entry["spec"]["method"] == method and bool(entry["spec"]["warmup_enabled"]) == warm
    ]
    return [RAW_RUN / "jobs" / entry["job_id"] / "result.npz" for entry in entries]


def mean_prefix_information(runtime: object, method: str, warm: bool) -> np.ndarray:
    spectra = []
    for path in raw_job_paths(method, warm):
        with np.load(path, allow_pickle=False) as arrays:
            policies = arrays["policies"][:200]
        information = cumulative_fisher_information(
            runtime.problem,
            policies,
            runtime.true_omega,
        )
        spectra.append(np.linalg.eigvalsh(information))
    return np.mean(spectra, axis=0)


def taskwise_final(slug: str, final_average: np.ndarray) -> np.ndarray:
    tasks = []
    for task_index in range(10):
        path = LEGACY_TRANSFER_CACHE / slug / f"task-{task_index:02d}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            tasks.append(arrays["suboptimality"][:, -1].copy())
    values = np.asarray(tasks)
    np.testing.assert_allclose(np.mean(values, axis=0), final_average, rtol=1e-11, atol=1e-14)
    return values


def style_axes(axis: plt.Axes) -> None:
    axis.grid(True, color="#D8DEE8", linewidth=0.65, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#354154", labelsize=8)


def render(output_dir: Path) -> dict[str, object]:
    with (
        np.load(
            "reproducibility/data/lobeke/final_evaluation/data.npz", allow_pickle=False
        ) as final,
        np.load("reproducibility/data/lobeke/tuning_grid.npz", allow_pickle=False) as grid,
        np.load(CORRECTED_OLS, allow_pickle=False) as corrected,
    ):
        multipliers = grid["lambda_information_multipliers"]
        budgets = grid["warmup_budgets"]
        grid_seeds = grid["seeds"].copy()
        scores = grid["task_scores"].mean(axis=-1)
        budget0 = int(np.flatnonzero(budgets == 0)[0])
        budget200 = int(np.flatnonzero(budgets == 200)[0])
        warm_effects = scores[budget200] - scores[budget0]
        warm_intervals = np.asarray([interval(row) for row in warm_effects])

        transfer_final = {
            method: final[f"{method}__averaged_transfer_suboptimality"][:, -1].copy()
            for method in (
                "active_no_warm",
                "active_warm",
                "zero_sum_ols",
                "fixed_expert_mle",
                "follower_ols",
            )
        }
        transfer_final["zero_sum_paired"] = corrected[
            "paired_actions_ref22__averaged_transfer_suboptimality"
        ].copy()
        active_tasks = taskwise_final("active_no_warm", transfer_final["active_no_warm"])
        zero_tasks = taskwise_final("zero_sum_ols", transfer_final["zero_sum_ols"])
        paired_tasks = corrected["paired_actions_ref22__taskwise_transfer_suboptimality"].copy()
        task_differences = {
            "exact": zero_tasks - active_tasks,
            "paired": paired_tasks - active_tasks,
        }
        task_intervals = {
            name: np.asarray([interval(row) for row in differences])
            for name, differences in task_differences.items()
        }

    runtime = load_runtime(RAW_RUN)
    spectra = {
        "Fixed expert": 200
        * np.linalg.eigvalsh(
            fisher_information(runtime.problem, runtime.expert.max_policy, runtime.true_omega)
        ),
        "Active, no warm-up": mean_prefix_information(runtime, "active", False),
        "Active, warm-up": mean_prefix_information(runtime, "active", True),
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)

    axis = axes[0, 0]
    x = np.arange(multipliers.size)
    means = warm_intervals[:, 0] * 1e4
    errors = np.vstack(
        (
            (warm_intervals[:, 0] - warm_intervals[:, 1]) * 1e4,
            (warm_intervals[:, 2] - warm_intervals[:, 0]) * 1e4,
        )
    )
    axis.errorbar(
        x,
        means,
        yerr=errors,
        fmt="o-",
        color=METHOD_COLORS["active_no_warm"],
        ecolor="#7792AC",
        capsize=2.5,
        linewidth=1.25,
        markersize=4,
    )
    axis.axhline(0.0, color="#2E3440", linewidth=0.8)
    axis.set_xticks(x, [f"{value:g}" for value in multipliers])
    axis.set(
        title="(a) Matched-λ warm-up effect",
        xlabel=r"Information multiplier $\lambda/\eta^2$",
        ylabel=r"Paired $\Delta$ transfer SubOpt ($10^{-4}$)",
    )
    axis.text(
        0.02,
        0.95,
        r"$\Delta=K_0{=}200-K_0{=}0$; negative favors warm-up",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#4B586A",
        va="top",
    )
    style_axes(axis)

    axis = axes[0, 1]
    categories = list(spectra)
    width = 0.24
    for eigen_index in range(3):
        axis.bar(
            np.arange(len(categories)) + (eigen_index - 1) * width,
            [spectra[name][eigen_index] for name in categories],
            width=width,
            label=rf"eigenvalue {eigen_index + 1}",
            color=("#B8C5D6", "#678DB7", "#235A91")[eigen_index],
        )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(categories)), categories)
    axis.set(
        title="(b) Information after 200 actions",
        ylabel="Cumulative Fisher eigenvalue",
    )
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.text(
        0.02,
        0.80,
        r"Fixed-expert condition number: $3.6\times10^8$",
        transform=axis.transAxes,
        fontsize=7.7,
        color="#4B586A",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    style_axes(axis)

    axis = axes[1, 0]
    ecdf_methods = (
        ("Active, no warm-up", "active_no_warm"),
        ("Zero-sum OLS, exact leader π", "zero_sum_ols"),
        ("Zero-sum OLS, paired samples", "zero_sum_paired"),
        ("Fixed-expert MLE", "fixed_expert_mle"),
        ("Follower-only OLS", "follower_ols"),
    )
    for label, method in ecdf_methods:
        values = np.sort(transfer_final[method])
        probability = np.arange(1, values.size + 1) / values.size
        axis.step(
            values,
            probability,
            where="post",
            label=label,
            color=METHOD_COLORS[method],
            linewidth=1.55,
        )
    axis.set_xscale("log")
    axis.set(
        title="(c) Transfer SubOpt distribution",
        xlabel="Averaged transfer SubOpt at 1,600 actions",
        ylabel="Fraction of seeds ≤ x",
    )
    axis.legend(frameon=False, loc="lower right")
    style_axes(axis)

    axis = axes[1, 1]
    task_x = np.arange(1, 11)
    for name, offset, label, color in (
        ("exact", -0.10, "Exact leader distribution", METHOD_COLORS["zero_sum_ols"]),
        ("paired", 0.10, "Paired sampled actions", METHOD_COLORS["zero_sum_paired"]),
    ):
        values = task_intervals[name]
        task_means = values[:, 0] * 1e4
        task_errors = np.vstack(
            (
                (values[:, 0] - values[:, 1]) * 1e4,
                (values[:, 2] - values[:, 0]) * 1e4,
            )
        )
        axis.errorbar(
            task_x + offset,
            task_means,
            yerr=task_errors,
            fmt="o",
            label=label,
            color=color,
            ecolor=color,
            alpha=0.9,
            capsize=2.5,
            markersize=3.5,
        )
    axis.axhline(0.0, color="#2E3440", linewidth=0.8)
    axis.set_xticks(task_x)
    axis.set(
        title="(d) Zero-sum protocol effect by task",
        xlabel="Protected transfer task",
        ylabel=r"Paired $\Delta$ SubOpt ($10^{-4}$)",
    )
    axis.legend(frameon=False, fontsize=7.2, loc="upper right")
    style_axes(axis)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mechanism_diagnostics.png"
    pdf_path = output_dir / "mechanism_diagnostics.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "warmup_effects": {
            str(float(multiplier)): {
                "mean": float(values[0]),
                "ci95": [float(values[1]), float(values[2])],
            }
            for multiplier, values in zip(multipliers, warm_intervals, strict=True)
        },
        "information_spectra_after_200": {
            name: values.tolist() for name, values in spectra.items()
        },
        "transfer_final_means": {
            method: float(np.mean(values)) for method, values in transfer_final.items()
        },
        "zero_sum_minus_active_taskwise": {
            protocol: {
                str(index + 1): {
                    "mean": float(values[0]),
                    "ci95": [float(values[1]), float(values[2])],
                }
                for index, values in enumerate(intervals)
            }
            for protocol, intervals in task_intervals.items()
        },
        "source": {
            "final_run": "a4b1b2a1cdf886dd",
            "final_seeds": list(range(30000, 30050)),
            "matched_lambda_grid_seeds": grid_seeds.tolist(),
            "taskwise_cache_verified_against_tracked_averages": True,
        },
    }
    (output_dir / "mechanism_diagnostics.json").write_text(
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
    summary = render(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
