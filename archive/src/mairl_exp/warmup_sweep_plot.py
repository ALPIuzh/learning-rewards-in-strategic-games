"""Versioned final-metric plots across fixed-total-budget warm-up sweeps."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mairl.exceptions import ConfigurationError
from mairl_exp.artifacts import (
    RUN_SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    collect_git_provenance,
    sha256_bytes,
    sha256_file,
)
from mairl_exp.mpl import plt
from mairl_exp.plotting import METHOD_LABELS
from mairl_exp.statistics import mean_confidence_interval


@dataclass(frozen=True, slots=True)
class WarmupSweepPlot:
    """Plot final performance against warm-up samples from one or more runs."""

    reference_run_dirs: tuple[str, ...]
    name: str = "warmup-sweep"
    metric: str = "fisher_transfer_error"
    confidence_level: float = 0.95
    band_alpha: float = 0.08
    x_scale: Literal["linear", "symlog"] = "symlog"
    dpi: int = 180
    allow_incomplete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_run_dirs", tuple(self.reference_run_dirs))
        if not self.name or not self.metric:
            raise ConfigurationError("plot name and metric must be nonempty")
        if not 0 < self.confidence_level < 1:
            raise ConfigurationError("confidence_level must lie in (0, 1)")
        if not 0 <= self.band_alpha <= 1:
            raise ConfigurationError("band_alpha must lie in [0, 1]")
        if self.x_scale not in ("linear", "symlog"):
            raise ConfigurationError("x_scale must be 'linear' or 'symlog'")
        if self.dpi < 1:
            raise ConfigurationError("plot dpi must be positive")

    def render(
        self,
        run_dir: Path,
        *,
        repository: Path,
        semantic_plot_config: object,
        resolved_plot_config_yaml: str,
        require_clean_git: bool,
    ) -> Path:
        git = collect_git_provenance(repository)
        if require_clean_git and git.dirty:
            changed = "\n".join(f"  {line}" for line in git.status)
            raise ConfigurationError(
                "refusing to generate a versioned plot from a dirty Git worktree. "
                "Commit the changes, or explicitly set require_clean_git=false.\n"
                f"{changed}"
            )
        source_dirs = (
            run_dir,
            *(
                path if path.is_absolute() else repository / path
                for path in map(Path, self.reference_run_dirs)
            ),
        )
        inputs, values, total_samples = self._load_inputs(source_dirs)
        identity = {
            "schema_version": RUN_SCHEMA_VERSION,
            "plot_config": semantic_plot_config,
            "git_fingerprint": git.fingerprint,
            "inputs": inputs,
        }
        plot_id = sha256_bytes(canonical_json(identity).encode())[:16]
        output_dir = run_dir / "plots" / plot_id
        figure_path = output_dir / "figure.png"
        manifest_path = output_dir / "manifest.json"
        if figure_path.exists() and manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("status") == "complete" and existing.get(
                "figure_sha256"
            ) == sha256_file(figure_path):
                return figure_path

        summary: dict[str, dict[str, dict[str, float | int]]] = {}
        figure, axis = plt.subplots(figsize=(9.4, 5.4))
        methods = tuple(
            method
            for method in METHOD_LABELS
            if any(method in budget_values for budget_values in values.values())
        )
        for method in methods:
            budgets = np.asarray(
                sorted(budget for budget, items in values.items() if method in items),
                dtype=np.int64,
            )
            means = np.empty(budgets.size, dtype=np.float64)
            lowers = np.empty_like(means)
            uppers = np.empty_like(means)
            for index, budget in enumerate(budgets):
                samples = np.asarray(values[int(budget)][method], dtype=np.float64)
                mean, lower, upper = self._mean_interval(samples)
                means[index] = mean
                lowers[index] = lower
                uppers[index] = upper
                summary.setdefault(str(int(budget)), {})[method] = {
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                    "seeds": int(samples.size),
                }
            line = axis.plot(
                budgets,
                np.maximum(means, 1e-12),
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=METHOD_LABELS[method],
            )[0]
            axis.fill_between(
                budgets,
                np.maximum(lowers, 1e-12),
                np.maximum(uppers, 1e-12),
                color=line.get_color(),
                alpha=self.band_alpha,
            )

        all_budgets = sorted(values)
        if self.x_scale == "symlog":
            positive_budgets = [budget for budget in all_budgets if budget > 0]
            linear_threshold = min(positive_budgets, default=1)
            axis.set_xscale(
                "symlog",
                base=2,
                linthresh=linear_threshold,
                linscale=1.0,
            )
        axis.set_xticks(all_budgets)
        axis.set_xticklabels(
            [str(budget) for budget in all_budgets],
            rotation=25,
            ha="right",
        )
        scale_label = " (symmetric-log scale)" if self.x_scale == "symlog" else ""
        axis.set(
            xlabel=f"Warm-up follower samples{scale_label}",
            ylabel=self.metric.replace("_", " "),
            yscale="log",
            title=(
                f"Fixed {total_samples}-sample budget: final {self.metric} "
                f"(mean with {self.confidence_level:.0%} CI)"
            ),
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)

        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(figure_path, buffer.getvalue())
        atomic_write_text(output_dir / "resolved_plot_config.yaml", resolved_plot_config_yaml)
        atomic_write_json(output_dir / "summary.json", summary)
        primary_manifest = json.loads((run_dir / "manifest.json").read_text())
        atomic_write_json(
            manifest_path,
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "status": "complete",
                "plot_id": plot_id,
                "run_id": primary_manifest["run_id"],
                "git": git.manifest(),
                "command": list(sys.argv),
                "inputs": inputs,
                "semantic_plot_config": semantic_plot_config,
                "figure_sha256": sha256_file(figure_path),
            },
        )
        return figure_path

    def _load_inputs(
        self,
        run_dirs: tuple[Path, ...],
    ) -> tuple[
        list[dict[str, object]],
        dict[int, dict[str, list[float]]],
        int,
    ]:
        inputs: list[dict[str, object]] = []
        values: dict[int, dict[str, list[float]]] = {}
        seen: set[tuple[int, str, int]] = set()
        total_samples: int | None = None
        missing: list[str] = []
        for run_dir in run_dirs:
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                raise ConfigurationError(f"run manifest does not exist: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            expected_jobs = manifest.get("expected_jobs")
            if not isinstance(expected_jobs, list):
                raise ConfigurationError(f"invalid execution plan in {manifest_path}")
            for expected in expected_jobs:
                job_dir = run_dir / "jobs" / expected["job_id"]
                status_path = job_dir / "status.json"
                result_path = job_dir / "result.npz"
                summary_path = job_dir / "summary.json"
                if (
                    not status_path.exists()
                    or not result_path.exists()
                    or not summary_path.exists()
                ):
                    missing.append(str(expected["label"]))
                    continue
                status = json.loads(status_path.read_text())
                digest = sha256_file(result_path)
                if status.get("status") != "complete" or status.get("result_sha256") != digest:
                    missing.append(str(expected["label"]))
                    continue
                summary = json.loads(summary_path.read_text())
                job = summary["job"]
                method = str(job["method"])
                seed = int(job["seed"])
                budget = int(summary["warmup"]["samples"])
                samples = int(summary["samples"])
                if total_samples is None:
                    total_samples = samples
                elif samples != total_samples:
                    raise ConfigurationError("warm-up sweep inputs use unequal total budgets")
                key = (budget, method, seed)
                if key in seen:
                    raise ConfigurationError(
                        f"duplicate warm-up sweep result for budget={budget}, "
                        f"method={method}, seed={seed}"
                    )
                seen.add(key)
                with np.load(result_path, allow_pickle=False) as arrays:
                    metric_key = f"metric__{self.metric}"
                    if metric_key not in arrays:
                        raise ConfigurationError(
                            f"job {expected['label']} lacks metric {self.metric!r}"
                        )
                    metric = arrays[metric_key]
                    if metric.size == 0:
                        raise ConfigurationError(
                            f"job {expected['label']} has an empty metric trace"
                        )
                    final_value = float(metric[-1])
                values.setdefault(budget, {}).setdefault(method, []).append(final_value)
                inputs.append(
                    {
                        "run_id": manifest["run_id"],
                        "job_id": expected["job_id"],
                        "label": expected["label"],
                        "result_sha256": digest,
                    }
                )
        if missing and not self.allow_incomplete:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
            raise ConfigurationError(
                f"plot requires complete runs; missing or invalid jobs: {preview}{suffix}"
            )
        if not values or total_samples is None:
            raise ConfigurationError("no completed warm-up sweep results are available")
        return inputs, values, total_samples

    def _mean_interval(self, values: np.ndarray) -> tuple[float, float, float]:
        mean, lower, upper = mean_confidence_interval(
            values,
            self.confidence_level,
            clip_lower=0.0,
        )
        return float(mean), float(lower), float(upper)
