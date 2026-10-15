"""Paired seed-level contrasts for a completed dense behavioral-transfer pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from investigations.behavioral_transfer.pilot import (
    BUCKET_NAMES,
    METHODS,
    METRICS,
    _safe_spearman,
)
from mairl_exp.artifacts import (
    atomic_write_json,
    collect_git_provenance,
    find_repository,
)


def paired_contrast(values: np.ndarray) -> dict[str, Any]:
    """Summarize a vector of paired differences using a Student-t interval."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a finite vector of length at least two")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    radius = float(student_t.ppf(0.975, values.size - 1) * standard_error)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "student_t_95": [mean - radius, mean + radius],
        "positive_pair_count": int(np.sum(values > 0.0)),
        "pair_count": int(values.size),
        "paired_differences": values.tolist(),
    }


def _task_seed_tensor(
    evaluation: dict[str, np.ndarray],
    *,
    task_indices: list[int],
    slug: str,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(evaluation[f"task_{task}__{slug}__suboptimality"], dtype=np.float64)
            for task in task_indices
        ]
    )


def _oracle_structure(policies: np.ndarray) -> dict[str, np.ndarray]:
    positive = policies > 1.0e-8
    entropy = -np.sum(
        np.where(positive, policies * np.log(np.maximum(policies, 1.0e-300)), 0.0), axis=1
    )
    return {
        "support_size": np.sum(positive, axis=1),
        "entropy": entropy,
        "maximum_probability": np.max(policies, axis=1),
    }


def _render_structure_scatter(
    *,
    manifest: dict[str, Any],
    tensors: dict[str, np.ndarray],
    maximum_probability: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.4, 5.0), constrained_layout=True)
    norm = Normalize(vmin=float(np.min(maximum_probability)), vmax=1.0)
    titles = {
        "active_warm": "Active, warm-up",
        "active_no_warm": "Active, no warm-up",
        "fixed_expert_mle": "Fixed-expert MLE",
    }
    image = None
    for row, metric in enumerate(METRICS):
        distances = np.asarray(manifest["candidate_metrics"][metric], dtype=np.float64)
        for column, slug in enumerate(METHODS.values()):
            axis = axes[row, column]
            values = tensors[slug][:, :, -1].mean(axis=1)
            image = axis.scatter(
                distances,
                values,
                c=maximum_probability,
                cmap="viridis",
                norm=norm,
                s=16,
                alpha=0.68,
                linewidths=0,
            )
            distance_correlation = _safe_spearman(distances, values)["correlation"]
            peak_correlation = _safe_spearman(maximum_probability, values)["correlation"]
            axis.set_title(
                f"{titles[slug]} ($\\rho_d$={distance_correlation:.2f}, "
                f"$\\rho_p$={peak_correlation:.2f})",
                fontsize=8,
            )
            axis.set_yscale("log")
            if row == 1:
                axis.set_xlabel(f"{metric.upper()} from source task")
            if column == 0:
                axis.set_ylabel("task-mean SubOpt at final checkpoint")
    assert image is not None
    figure.colorbar(image, ax=axes, label="oracle maximum action probability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    output_path = args.output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite dense explanation: {output_path}")
    plot_path = args.plot_path.resolve()
    if plot_path.exists():
        raise FileExistsError(f"refusing to overwrite dense explanation plot: {plot_path}")
    derived_dir = args.derived_dir.resolve()
    manifest = json.loads((derived_dir / "task_manifest.json").read_text())
    if manifest["selection_mode"] != "all_candidates":
        raise ValueError("dense explanation requires an all-candidates evaluation")
    with np.load(derived_dir / "evaluation.npz") as stored:
        evaluation = {name: stored[name].copy() for name in stored.files}
    with np.load(derived_dir / "candidate_tasks.npz") as stored:
        oracle_policies = stored["oracle_policies"].copy()
    checkpoints = np.asarray(evaluation["checkpoints"], dtype=np.int64)
    structure = _oracle_structure(oracle_policies)
    source_policy = np.asarray(manifest["source_policy"], dtype=np.float64)
    source_overlap = np.minimum(oracle_policies, source_policy[None, :]).sum(axis=1)

    tensors = {
        slug: _task_seed_tensor(
            evaluation,
            task_indices=[int(task) for task in manifest["selected_union"]],
            slug=slug,
        )
        for slug in METHODS.values()
    }
    bucket_policy_structure: dict[str, Any] = {}
    for metric in METRICS:
        assignments = np.asarray(manifest["buckets"][metric]["assignments"], dtype=np.int64)
        bucket_policy_structure[metric] = {}
        for bucket, bucket_name in enumerate(BUCKET_NAMES):
            tasks = np.flatnonzero(assignments == bucket)
            bucket_policy_structure[metric][bucket_name] = {
                "task_count": int(tasks.size),
                "support_size_mean": float(np.mean(structure["support_size"][tasks])),
                "entropy_median": float(np.median(structure["entropy"][tasks])),
                "maximum_probability_median": float(
                    np.median(structure["maximum_probability"][tasks])
                ),
                "source_overlap_median": float(np.median(source_overlap[tasks])),
                "disjoint_source_support_fraction": float(np.mean(source_overlap[tasks] < 1.0e-8)),
            }

    methods: dict[str, Any] = {}
    for slug in METHODS.values():
        method_payload: dict[str, Any] = {"task_structure_correlations": {}}
        for checkpoint_position, checkpoint in enumerate(checkpoints):
            task_means = tensors[slug][:, :, checkpoint_position].mean(axis=1)
            method_payload["task_structure_correlations"][str(int(checkpoint))] = {
                name: _safe_spearman(values, task_means) for name, values in structure.items()
            }
        for metric in METRICS:
            assignments = np.asarray(manifest["buckets"][metric]["assignments"], dtype=np.int64)
            bucket_tensors = [
                tensors[slug][np.flatnonzero(assignments == bucket)]
                for bucket in range(len(BUCKET_NAMES))
            ]
            metric_payload: dict[str, Any] = {"bucket_variability": {}, "contrasts": {}}
            for bucket, bucket_name in enumerate(BUCKET_NAMES):
                final_values = bucket_tensors[bucket][:, :, -1]
                metric_payload["bucket_variability"][bucket_name] = {
                    "task_mean_standard_deviation": float(
                        np.std(final_values.mean(axis=1), ddof=1)
                    ),
                    "seed_mean_standard_deviation": float(
                        np.std(final_values.mean(axis=0), ddof=1)
                    ),
                }
            for checkpoint_position, checkpoint in enumerate(checkpoints):
                seed_means = [
                    tensor[:, :, checkpoint_position].mean(axis=0) for tensor in bucket_tensors
                ]
                metric_payload["contrasts"][str(int(checkpoint))] = {
                    "medium_minus_low": paired_contrast(seed_means[1] - seed_means[0]),
                    "high_minus_medium": paired_contrast(seed_means[2] - seed_means[1]),
                    "high_minus_low": paired_contrast(seed_means[2] - seed_means[0]),
                }
            method_payload[metric] = metric_payload
        methods[slug] = method_payload

    pairwise_methods = {}
    slugs = list(METHODS.values())
    for left_index, left in enumerate(slugs):
        for right in slugs[left_index + 1 :]:
            left_values = tensors[left]
            right_values = tensors[right]
            label = f"{left}_minus_{right}"
            pairwise_methods[label] = {}
            for checkpoint_position, checkpoint in enumerate(checkpoints):
                seed_difference = left_values[:, :, checkpoint_position].mean(
                    axis=0
                ) - right_values[:, :, checkpoint_position].mean(axis=0)
                task_mean_left = left_values[:, :, checkpoint_position].mean(axis=1)
                task_mean_right = right_values[:, :, checkpoint_position].mean(axis=1)
                pairwise_methods[label][str(int(checkpoint))] = {
                    "paired_seed_contrast": paired_contrast(seed_difference),
                    "left_better_task_fraction": float(np.mean(task_mean_left < task_mean_right)),
                }

    _render_structure_scatter(
        manifest=manifest,
        tensors=tensors,
        maximum_probability=structure["maximum_probability"],
        output_path=plot_path,
    )
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "derived_task_seed": manifest["task_seed"],
            "source_run": manifest["source_run"],
            "evaluation_git": collect_git_provenance(find_repository(Path.cwd())).manifest(),
            "bucket_policy_structure": bucket_policy_structure,
            "methods": methods,
            "pairwise_methods": pairwise_methods,
            "structure_plot": str(plot_path),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--plot-path", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
