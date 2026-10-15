"""Export one full trajectory comparison for every measured warm-up budget."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

from experiments._config import REPOSITORY_ROOT
from mairl.exceptions import ConfigurationError
from mairl_exp.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from mairl_exp.mpl import plt
from mairl_exp.plotting import METHOD_LABELS
from mairl_exp.statistics import mean_confidence_interval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="completed run supplying warm-up budgets 0 and 120",
    )
    parser.add_argument(
        "--sweep-run",
        type=Path,
        required=True,
        help="completed run supplying the remaining warm-up budgets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plots/toy"),
    )
    parser.add_argument("--metric", default="fisher_transfer_error")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--band-alpha", type=float, default=0.08)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _load(
    run_dirs: tuple[Path, ...],
    metric: str,
) -> tuple[
    dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]],
    int,
    list[dict[str, str]],
    list[dict[str, str]],
]:
    traces: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    seen: set[tuple[int, str, int]] = set()
    total_samples: int | None = None
    input_records: list[dict[str, str]] = []
    source_runs: list[dict[str, str]] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ConfigurationError(f"run manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ConfigurationError(f"source run is not complete: {run_dir}")
        source_runs.append(
            {
                "run_id": str(manifest["run_id"]),
                "path": run_dir.relative_to(REPOSITORY_ROOT).as_posix()
                if run_dir.is_relative_to(REPOSITORY_ROOT)
                else run_dir.name,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        for expected in manifest["expected_jobs"]:
            job_dir = run_dir / "jobs" / expected["job_id"]
            status_path = job_dir / "status.json"
            summary_path = job_dir / "summary.json"
            result_path = job_dir / "result.npz"
            if not status_path.exists() or not summary_path.exists() or not result_path.exists():
                raise ConfigurationError(f"incomplete source job: {expected['label']}")
            status = json.loads(status_path.read_text())
            result_digest = sha256_file(result_path)
            if status.get("status") != "complete" or status.get("result_sha256") != result_digest:
                raise ConfigurationError(f"invalid source job: {expected['label']}")
            summary = json.loads(summary_path.read_text())
            job = summary["job"]
            budget = int(summary["warmup"]["samples"])
            method = str(job["method"])
            seed = int(job["seed"])
            key = (budget, method, seed)
            if key in seen:
                raise ConfigurationError(
                    f"duplicate source result for budget={budget}, method={method}, seed={seed}"
                )
            seen.add(key)
            samples = int(summary["samples"])
            if total_samples is None:
                total_samples = samples
            elif samples != total_samples:
                raise ConfigurationError("source runs use unequal total sample budgets")
            with np.load(result_path, allow_pickle=False) as arrays:
                metric_key = f"metric__{metric}"
                if metric_key not in arrays:
                    raise ConfigurationError(f"job {expected['label']} lacks metric {metric!r}")
                checkpoints = arrays["evaluation_sample_counts"].copy()
                values = arrays[metric_key].copy()
            if checkpoints.shape != values.shape or values.size == 0:
                raise ConfigurationError(f"job {expected['label']} has an invalid trace")
            traces.setdefault(budget, {}).setdefault(method, []).append((checkpoints, values))
            input_records.append(
                {
                    "run_id": str(manifest["run_id"]),
                    "job_id": str(expected["job_id"]),
                    "result_sha256": result_digest,
                }
            )
    if total_samples is None:
        raise ConfigurationError("no source results were loaded")
    return traces, total_samples, input_records, source_runs


def _interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mean_confidence_interval(values, confidence_level, clip_lower=0.0)


def _prepared_curves(
    traces: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]],
    confidence_level: float,
) -> tuple[
    dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    float,
    float,
    int,
]:
    curves = {}
    global_lower = np.inf
    global_upper = 0.0
    seed_counts: set[int] = set()
    for budget, methods in traces.items():
        if set(methods) != set(METHOD_LABELS):
            raise ConfigurationError(f"budget {budget} does not contain all three methods")
        curves[budget] = {}
        for method, method_traces in methods.items():
            checkpoints = method_traces[0][0]
            if not all(np.array_equal(checkpoints, item[0]) for item in method_traces):
                raise ConfigurationError(
                    f"budget {budget}, method {method} uses inconsistent checkpoints"
                )
            values = np.stack([item[1] for item in method_traces])
            mean, lower, upper = _interval(values, confidence_level)
            positive_lower = lower[lower > 0]
            if positive_lower.size:
                global_lower = min(global_lower, float(np.min(positive_lower)))
            global_upper = max(global_upper, float(np.max(upper)))
            seed_counts.add(values.shape[0])
            curves[budget][method] = checkpoints, mean, lower, upper
    if len(seed_counts) != 1:
        raise ConfigurationError("threshold plots require the same seed count everywhere")
    if not np.isfinite(global_lower) or not global_upper > 0:
        raise ConfigurationError("metric intervals do not define a positive log scale")
    return curves, 0.85 * global_lower, 1.15 * global_upper, seed_counts.pop()


def _render(
    *,
    budget: int,
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    total_samples: int,
    seed_count: int,
    metric: str,
    y_limits: tuple[float, float],
    confidence_level: float,
    band_alpha: float,
    dpi: int,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    for method, label in METHOD_LABELS.items():
        checkpoints, mean, lower, upper = curves[method]
        line = axis.plot(
            checkpoints,
            np.maximum(mean, 1e-12),
            label=label,
            linewidth=1.8,
            marker="o",
            markersize=2.5,
        )[0]
        axis.fill_between(
            checkpoints,
            np.maximum(lower, 1e-12),
            np.maximum(upper, 1e-12),
            color=line.get_color(),
            alpha=band_alpha,
        )
    if budget > 0:
        axis.axvspan(0, budget, color="black", alpha=0.035, zorder=0)
        axis.axvline(budget, color="0.35", linestyle="--", linewidth=1.0)
        axis.text(
            budget,
            0.03,
            "warm-up ends",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            color="0.35",
            fontsize=8,
        )
    endpoint_means = [curves[method][1][-1] for method in METHOD_LABELS]
    if len(curves["active"][0]) == 1 and np.ptp(endpoint_means) == 0:
        axis.annotate(
            f"all methods coincide: {endpoint_means[0]:.4f}",
            xy=(total_samples, endpoint_means[0]),
            xytext=(-12, 24),
            textcoords="offset points",
            ha="right",
            arrowprops={"arrowstyle": "-", "color": "0.35"},
            color="0.35",
            fontsize=8,
        )
    axis.set(
        xlim=(0, 1.02 * total_samples),
        ylim=y_limits,
        yscale="log",
        xlabel="Total follower samples",
        ylabel=metric.replace("_", " "),
        title=(
            ("Warm-up disabled" if budget == 0 else f"Warm-up {budget}/{total_samples} samples")
            + f" ({seed_count} seeds; "
            f"mean with {confidence_level:.0%} CI)"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output_path, buffer.getvalue())


def main() -> None:
    args = _parser().parse_args()
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie in (0, 1)")
    if not 0 <= args.band_alpha <= 1:
        raise ValueError("--band-alpha must lie in [0, 1]")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    output_dir = _resolve(args.output_dir)
    traces, total_samples, input_records, source_runs = _load(
        (_resolve(args.reference_run), _resolve(args.sweep_run)),
        args.metric,
    )
    curves, y_min, y_max, seed_count = _prepared_curves(
        traces,
        args.confidence_level,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for budget in sorted(curves):
        output_path = output_dir / f"warmup_{budget:04d}_{args.metric}.png"
        _render(
            budget=budget,
            curves=curves[budget],
            total_samples=total_samples,
            seed_count=seed_count,
            metric=args.metric,
            y_limits=(y_min, y_max),
            confidence_level=args.confidence_level,
            band_alpha=args.band_alpha,
            dpi=args.dpi,
            output_path=output_path,
        )
        generated.append(
            {
                "warmup_budget": budget,
                "path": str(output_path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(output_path),
            }
        )
        print(f"plot={output_path}")
    input_records.sort(key=lambda item: (item["run_id"], item["job_id"]))
    final_summary = {
        str(budget): {
            method: {
                "mean": float(curves[budget][method][1][-1]),
                "lower": float(curves[budget][method][2][-1]),
                "upper": float(curves[budget][method][3][-1]),
            }
            for method in METHOD_LABELS
        }
        for budget in sorted(curves)
    }
    manifest_path = output_dir / "warmup_threshold_plots_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "command": [
                str(argument).replace(str(REPOSITORY_ROOT) + "/", "") for argument in sys.argv
            ],
            "metric": args.metric,
            "confidence_level": args.confidence_level,
            "band_alpha": args.band_alpha,
            "total_samples": total_samples,
            "seed_count": seed_count,
            "source_runs": source_runs,
            "input_result_count": len(input_records),
            "input_results_sha256": sha256_bytes(canonical_json(input_records).encode()),
            "final_summary": final_summary,
            "plots": generated,
        },
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
