"""Export reproducible warm-up and estimator-comparison plots from stored runs."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from mairl.exceptions import ConfigurationError
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.evaluation.mpl import plt
from mairl_exp.evaluation.statistics import mean_confidence_interval
from mairl_exp.execution.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

METHOD_LABELS = {"active": "Active IRL", "follower_ols": "Follower-only OLS", "zero_sum_ols": "Zero-sum OLS"}
_METRICS = (
    ("fixed_leader_fisher_error", "Fixed-leader Fisher error"),
    ("coefficient_l2_error", "Coefficient L2 error"),
    ("regularized_qre_leader_l1_error", "Regularized-QRE leader policy L1 error"),
)
ComparisonTrace = tuple[int, np.ndarray, dict[str, np.ndarray]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-run", type=Path, required=True)
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots/zero_sum_comparison"))
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--band-alpha", type=float, default=0.08)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _mean_interval(values: np.ndarray, confidence_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mean_confidence_interval(values, confidence_level, clip_lower=0.0)


def _load_comparison(run_dir: Path) -> tuple[dict[str, list[ComparisonTrace]], list[dict[str, str]], dict[str, object]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigurationError(f"comparison manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ConfigurationError("comparison run is not complete")

    traces: dict[str, list[ComparisonTrace]] = {}
    inputs = []
    for expected in manifest["expected_jobs"]:
        result_path = run_dir / "jobs" / expected["job_id"] / "result.npz"
        status_path = result_path.with_name("status.json")
        if not result_path.exists() or not status_path.exists():
            raise ConfigurationError(f"missing comparison job {expected['label']}")
        status = json.loads(status_path.read_text())
        digest = sha256_file(result_path)
        if status.get("status") != "complete" or status.get("result_sha256") != digest:
            raise ConfigurationError(f"invalid comparison job {expected['label']}")
        with np.load(result_path, allow_pickle=False) as arrays:
            checkpoints = arrays["evaluation_sample_counts"].copy()
            metrics = {metric: arrays[f"metric__{metric}"].copy() for metric, _ in _METRICS}
        method = str(expected["spec"]["method"])
        seed = int(expected["spec"]["seed"])
        traces.setdefault(method, []).append((seed, checkpoints, metrics))
        inputs.append({"job_id": str(expected["job_id"]), "result_sha256": digest})
    if set(traces) != set(METHOD_LABELS):
        raise ConfigurationError("comparison run does not contain all three estimators")
    for method_traces in traces.values():
        method_traces.sort(key=lambda item: item[0])
    return traces, inputs, manifest


def _render_comparison(traces: dict[str, list[ComparisonTrace]], *, confidence_level: float, band_alpha: float,
        dpi: int, output_path: Path) -> dict[str, dict[str, dict[str, float | int]]]:  # fmt: skip
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    active_checkpoints = traces["active"][0][1]
    warmup_end = int(active_checkpoints[0])
    total_samples = max(int(item[1][-1]) for method_traces in traces.values() for item in method_traces)
    for axis, (metric, label) in zip(axes, _METRICS, strict=True):
        summary[metric] = {}
        for method, method_label in METHOD_LABELS.items():
            method_traces = traces[method]
            checkpoints = method_traces[0][1]
            if not all(np.array_equal(checkpoints, item[1]) for item in method_traces):
                raise ConfigurationError(f"{method} uses inconsistent evaluation checkpoints")
            values = np.stack([item[2][metric] for item in method_traces])
            mean, lower, upper = _mean_interval(values, confidence_level)
            line = axis.plot(checkpoints, np.maximum(mean, 1e-12), label=method_label, linewidth=1.8)[0]
            axis.fill_between(
                checkpoints,
                np.maximum(lower, 1e-12),
                np.maximum(upper, 1e-12),
                color=line.get_color(),
                alpha=band_alpha,
            )
            summary[metric][method] = {
                "mean": float(mean[-1]),
                "lower": float(lower[-1]),
                "upper": float(upper[-1]),
                "seeds": int(values.shape[0]),
            }
        axis.set(xlim=(0, total_samples), xlabel="Follower samples", ylabel=label, yscale="log")
        axis.axvspan(0, warmup_end, color="black", alpha=0.035, zorder=0)
        axis.axvline(warmup_end, color="0.35", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.25)
    axes[0].text(
        warmup_end,
        0.03,
        f"Active warm-up ends: {warmup_end}",
        transform=axes[0].get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        color="0.35",
        fontsize=8,
    )
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Entropy-regularized zero-sum game "
        r"($\eta=1.25$ for both QRE players)"
        "\nEqual 200-follower-sample budget "
        f"(mean with {confidence_level:.0%} CI)",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output_path, buffer.getvalue())
    return summary


def _paired_summary(traces: dict[str, list[ComparisonTrace]],
        confidence_level: float) -> dict[str, dict[str, dict[str, float]]]:  # fmt: skip
    by_method = {
        method: {seed: metrics for seed, _, metrics in method_traces} for method, method_traces in traces.items()
    }
    pairs = (("active", "follower_ols"), ("active", "zero_sum_ols"), ("zero_sum_ols", "follower_ols"))
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for metric, _ in _METRICS:
        summary[metric] = {}
        for first, second in pairs:
            if set(by_method[first]) != set(by_method[second]):
                raise ConfigurationError("paired comparison methods use different seeds")
            seeds = sorted(by_method[first])
            first_values = np.asarray([by_method[first][seed][metric][-1] for seed in seeds], dtype=np.float64)
            second_values = np.asarray([by_method[second][seed][metric][-1] for seed in seeds], dtype=np.float64)
            differences = first_values - second_values
            standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
            critical = float(student_t.ppf(0.5 + confidence_level / 2.0, df=differences.size - 1))
            mean_difference = float(np.mean(differences))
            summary[metric][f"{first}_minus_{second}"] = {
                "mean_difference": mean_difference,
                "lower": mean_difference - critical * standard_error,
                "upper": mean_difference + critical * standard_error,
                "first_method_win_rate": float(np.mean(first_values < second_values)),
                "ratio_of_means": float(np.mean(first_values) / np.mean(second_values)),
            }
    return summary


def _render_tuning(tuning_run: Path, *, confidence_level: float, dpi: int,
        output_path: Path) -> tuple[dict[str, object], str]:  # fmt: skip
    selection_path = tuning_run / "warmup_selection.json"
    manifest_path = tuning_run / "manifest.json"
    if not selection_path.exists() or not manifest_path.exists():
        raise ConfigurationError("tuning run lacks its selection or manifest")
    selection = json.loads(selection_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if selection.get("status") != "selected" or manifest.get("status") != "complete":
        raise ConfigurationError("warm-up tuning run is not complete and selected")

    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for split, label, marker in (
        ("selection_statistics", "Selection seeds", "o"),
        ("confirmation_statistics", "Confirmation seeds", "s"),
    ):
        statistics = selection[split]
        budgets = np.asarray(sorted(int(value) for value in statistics), dtype=np.int64)
        means = np.asarray([statistics[str(int(budget))]["mean"] for budget in budgets], dtype=np.float64)
        errors = np.asarray([statistics[str(int(budget))]["standard_error"] for budget in budgets], dtype=np.float64)
        seed_count = int(statistics[str(int(budgets[0]))]["seeds"])
        critical = float(student_t.ppf(0.5 + confidence_level / 2.0, df=seed_count - 1))
        axis.errorbar(
            budgets,
            means,
            yerr=critical * errors,
            marker=marker,
            capsize=3,
            linewidth=1.5,
            label=f"{label} ({seed_count})",
        )
    selected = int(selection["selected_warmup_budget"])
    axis.set_xticks(budgets)
    axis.set_xticklabels([str(int(budget)) for budget in budgets])
    axis.axvline(selected, color="0.35", linestyle="--", linewidth=1.0)
    axis.text(
        selected,
        0.03,
        f"selected: {selected}",
        transform=axis.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        color="0.35",
        fontsize=8,
    )
    axis.set(
        xlabel="Warm-up follower samples (total budget: 200)",
        ylabel="Final fixed-leader Fisher error",
        yscale="log",
        title=(f"Active IRL warm-up tuning (mean with {confidence_level:.0%} CI; disjoint splits)"),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output_path, buffer.getvalue())
    return selection, sha256_file(manifest_path)


def main() -> None:
    args = _parser().parse_args()
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie in (0, 1)")
    if not 0 <= args.band_alpha <= 1:
        raise ValueError("--band-alpha must lie in [0, 1]")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")

    tuning_run = _resolve(args.tuning_run)
    comparison_run = _resolve(args.comparison_run)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces, inputs, comparison_manifest = _load_comparison(comparison_run)
    comparison_path = output_dir / "estimator_comparison.png"
    tuning_path = output_dir / "warmup_tuning.png"
    final_summary = _render_comparison(
        traces,
        confidence_level=args.confidence_level,
        band_alpha=args.band_alpha,
        dpi=args.dpi,
        output_path=comparison_path,
    )
    paired_summary = _paired_summary(traces, args.confidence_level)
    selection, tuning_manifest_digest = _render_tuning(
        tuning_run, confidence_level=args.confidence_level, dpi=args.dpi, output_path=tuning_path
    )
    inputs.sort(key=lambda item: item["job_id"])
    atomic_write_json(
        output_dir / "results_manifest.json",
        {
            "schema_version": 1,
            "tuning_run": str(tuning_run.relative_to(REPOSITORY_ROOT)),
            "tuning_manifest_sha256": tuning_manifest_digest,
            "comparison_run": str(comparison_run.relative_to(REPOSITORY_ROOT)),
            "comparison_manifest_sha256": sha256_file(comparison_run / "manifest.json"),
            "comparison_run_id": comparison_manifest["run_id"],
            "comparison_inputs_sha256": sha256_bytes(canonical_json(inputs).encode()),
            "confidence_level": args.confidence_level,
            "band_alpha": args.band_alpha,
            "selected_warmup_budget": selection["selected_warmup_budget"],
            "final_summary": final_summary,
            "paired_summary": paired_summary,
            "plots": {
                "warmup_tuning": {
                    "path": str(tuning_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": sha256_file(tuning_path),
                },
                "estimator_comparison": {
                    "path": str(comparison_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": sha256_file(comparison_path),
                },
            },
        },
    )
    print(f"plot={tuning_path}")
    print(f"plot={comparison_path}")
    print(f"manifest={output_dir / 'results_manifest.json'}")


if __name__ == "__main__":
    main()
