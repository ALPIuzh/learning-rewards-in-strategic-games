"""On-demand plots derived from durable experiment results."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mairl.exceptions import ConfigurationError
from mairl_exp.evaluation.mpl import plt
from mairl_exp.evaluation.statistics import mean_confidence_interval
from mairl_exp.execution.artifacts import (
    RUN_SCHEMA_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    collect_git_provenance,
    sha256_bytes,
    sha256_file,
)

METHOD_LABELS = {
    "active": "Active information",
    "fixed_random": "Fixed random policy",
    "random_each_round": "Random policy each round",
}


@dataclass(frozen=True, slots=True)
class FisherTransferPlot:
    """Mean trace with configurable uncertainty or run-distribution bands."""

    name: str = "fisher-transfer"
    metric: str = "fisher_transfer_error"
    band: Literal["mean_ci", "quantile"] = "mean_ci"
    confidence_level: float = 0.95
    band_alpha: float = 0.08
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    dpi: int = 180
    allow_incomplete: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.metric:
            raise ConfigurationError("plot name and metric must be nonempty")
        if self.band not in ("mean_ci", "quantile"):
            raise ConfigurationError("band must be 'mean_ci' or 'quantile'")
        if not 0 < self.confidence_level < 1:
            raise ConfigurationError("confidence_level must lie in (0, 1)")
        if not 0 <= self.band_alpha <= 1:
            raise ConfigurationError("band_alpha must lie in [0, 1]")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ConfigurationError("plot quantiles must satisfy 0 <= lower < upper <= 1")
        if self.dpi < 1:
            raise ConfigurationError("plot dpi must be positive")

    def render(self, run_dir: Path, *, repository: Path, semantic_plot_config: object, resolved_plot_config_yaml: str,
            require_clean_git: bool) -> Path:  # fmt: skip
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ConfigurationError(f"run manifest does not exist: {manifest_path}")
        run_manifest = json.loads(manifest_path.read_text())
        git = collect_git_provenance(repository)
        if require_clean_git and git.dirty:
            changed = "\n".join(f"  {line}" for line in git.status)
            raise ConfigurationError(
                "refusing to generate a versioned plot from a dirty Git worktree. "
                "Commit the changes, or explicitly set require_clean_git=false.\n"
                f"{changed}"
            )

        inputs, panels = self._load_inputs(run_dir, run_manifest)
        identity = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_manifest["run_id"],
            "plot_config": semantic_plot_config,
            "git_fingerprint": git.fingerprint,
            "inputs": inputs,
        }
        plot_id = sha256_bytes(canonical_json(identity).encode())[:16]
        output_dir = run_dir / "plots" / plot_id
        figure_path = output_dir / "figure.png"
        output_manifest_path = output_dir / "manifest.json"
        if figure_path.exists() and output_manifest_path.exists():
            existing = json.loads(output_manifest_path.read_text())
            if existing.get("status") == "complete" and existing.get("figure_sha256") == sha256_file(figure_path):
                return figure_path

        output_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
        summary: dict[str, dict[str, dict[str, float]]] = {}
        for axis, warmup_enabled in zip(axes, (True, False), strict=True):
            panel = panels.get(warmup_enabled, {})
            summary_name = "with_warmup" if warmup_enabled else "without_warmup"
            summary[summary_name] = {}
            seed_counts: set[int] = set()
            panel_max_samples = 0
            panel_first_samples: set[int] = set()
            for method, traces in panel.items():
                checkpoints = traces[0][0]
                values = np.stack([trace[1] for trace in traces])
                seed_counts.add(values.shape[0])
                panel_first_samples.add(int(checkpoints[0]))
                panel_max_samples = max(panel_max_samples, int(checkpoints[-1]))
                if not all(np.array_equal(checkpoints, trace[0]) for trace in traces):
                    raise ConfigurationError("plot inputs use inconsistent checkpoints")
                mean = np.mean(values, axis=0)
                lower, upper = self._band(values, mean)
                line = axis.plot(checkpoints, np.maximum(mean, 1e-12), label=METHOD_LABELS.get(method, method))[0]
                axis.fill_between(
                    checkpoints,
                    np.maximum(lower, 1e-12),
                    np.maximum(upper, 1e-12),
                    color=line.get_color(),
                    alpha=self.band_alpha,
                )
                summary[summary_name][method] = {
                    "mean": float(mean[-1]),
                    "lower": float(lower[-1]),
                    "upper": float(upper[-1]),
                    "seeds": int(values.shape[0]),
                    "band": self.band,
                }
            if len(panel_first_samples) > 1:
                raise ConfigurationError("plot methods start at inconsistent sample counts")
            if warmup_enabled and panel_first_samples:
                warmup_end = panel_first_samples.pop()
                axis.axvspan(0, warmup_end, color="black", alpha=0.035, zorder=0)
                axis.axvline(warmup_end, color="0.35", linestyle="--", linewidth=1.0)
                axis.text(
                    warmup_end,
                    0.03,
                    "warm-up ends",
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    va="bottom",
                    ha="right",
                    color="0.35",
                    fontsize=8,
                )
            if panel_max_samples:
                axis.set_xlim(0, panel_max_samples)
            if len(seed_counts) == 1:
                seed_count = seed_counts.pop()
                seed_label = f"{seed_count} seed" if seed_count == 1 else f"{seed_count} seeds"
            else:
                seed_label = "seed counts vary"
            warmup_label = "Warm-up enabled" if warmup_enabled else "Warm-up disabled"
            axis.set(
                title=f"{warmup_label} ({seed_label})",
                xlabel="Total follower samples",
                ylabel=self.metric.replace("_", " "),
                yscale="log",
            )
            axis.grid(alpha=0.25)
        legend_axis = next((axis for axis in axes if axis.lines), None)
        if legend_axis is not None:
            legend_axis.legend(fontsize=8)
        if self.band == "mean_ci":
            band_label = f"mean with {self.confidence_level:.0%} CI"
        else:
            band_label = f"mean with {self.lower_quantile:.0%} to {self.upper_quantile:.0%} seed quantiles"
        figure.suptitle(f"Equal-budget exploration comparison: {self.metric} ({band_label})", y=0.98)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)

        atomic_write_bytes(figure_path, buffer.getvalue())
        atomic_write_text(output_dir / "resolved_plot_config.yaml", resolved_plot_config_yaml)
        atomic_write_json(output_dir / "summary.json", summary)
        atomic_write_json(
            output_manifest_path,
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "status": "complete",
                "plot_id": plot_id,
                "run_id": run_manifest["run_id"],
                "git": git.manifest(),
                "command": list(sys.argv),
                "inputs": inputs,
                "semantic_plot_config": semantic_plot_config,
                "figure_sha256": sha256_file(figure_path),
            },
        )
        return figure_path

    def _band(self, values: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.band == "quantile":
            return (np.quantile(values, self.lower_quantile, axis=0), np.quantile(values, self.upper_quantile, axis=0))
        if values.shape[0] < 2:
            return mean.copy(), mean.copy()
        _, lower, upper = mean_confidence_interval(values, self.confidence_level, clip_lower=0.0)
        return lower, upper

    def _load_inputs(self, run_dir: Path, run_manifest: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[bool, dict[str, list[tuple[np.ndarray, np.ndarray]]]]]:  # fmt: skip
        inputs: list[dict[str, object]] = []
        panels: dict[bool, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
        missing: list[str] = []
        expected_jobs = run_manifest.get("expected_jobs")
        if not isinstance(expected_jobs, list):
            raise ConfigurationError("run manifest does not contain a valid execution plan")
        for expected in expected_jobs:
            job_id = expected["job_id"]
            job_dir = run_dir / "jobs" / job_id
            status_path = job_dir / "status.json"
            result_path = job_dir / "result.npz"
            if not status_path.exists() or not result_path.exists():
                missing.append(expected["label"])
                continue
            status = json.loads(status_path.read_text())
            digest = sha256_file(result_path)
            if status.get("status") != "complete" or status.get("result_sha256") != digest:
                missing.append(expected["label"])
                continue
            with np.load(result_path, allow_pickle=False) as arrays:
                metric_key = f"metric__{self.metric}"
                if metric_key not in arrays:
                    raise ConfigurationError(f"job {expected['label']} does not contain metric {self.metric!r}")
                checkpoints = arrays["evaluation_sample_counts"].copy()
                values = arrays[metric_key].copy()
            spec = expected["spec"]
            warmup_enabled = bool(spec["warmup_enabled"])
            method = str(spec["method"])
            panels.setdefault(warmup_enabled, {}).setdefault(method, []).append((checkpoints, values))
            inputs.append({"job_id": job_id, "label": expected["label"], "result_sha256": digest})
        if missing and not self.allow_incomplete:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
            raise ConfigurationError(f"plot requires a complete run; missing or invalid jobs: {preview}{suffix}")
        if not inputs:
            raise ConfigurationError("no completed jobs are available for plotting")
        return inputs, panels
