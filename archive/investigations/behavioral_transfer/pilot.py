"""Small behavioral-transfer pilot using policy TV and source-policy gap.

This is deliberately an exploratory diagnostic rather than a publication-data
export.  It reuses a completed reduced Lobeke run, constructs downstream tasks
without looking at any learned estimate, stratifies the tasks independently by
two source-relative metrics, and then evaluates a small fixed task subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
from hydra.utils import instantiate
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.guard.paper.figures.data import MethodKey, SeedTrace, load_run_traces
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.task_sets import (
    policy_total_variation,
    quantile_bucket_assignments,
    source_policy_gap,
)
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from experiments.guard.paper.rewards import unit_interval_leader_reward
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
)
from mairl_exp.artifacts import (
    atomic_write_json,
    atomic_write_npz,
    collect_git_provenance,
    find_repository,
)

METHODS: dict[MethodKey, str] = {
    ("active", True): "active_warm",
    ("active", False): "active_no_warm",
    ("fixed_expert_mle", False): "fixed_expert_mle",
}
METRICS = ("tv", "gap")
BUCKET_NAMES = ("low", "medium", "high")


def select_bucket_tasks(
    assignments: np.ndarray,
    *,
    tasks_per_bucket: int,
    seed: int,
) -> np.ndarray:
    """Select tasks uniformly within buckets without method-performance inputs."""

    assignments = np.asarray(assignments, dtype=np.int64)
    bucket_count = int(np.max(assignments)) + 1
    rng = np.random.default_rng(seed)
    selected = np.empty((bucket_count, tasks_per_bucket), dtype=np.int64)
    for bucket in range(bucket_count):
        candidates = np.flatnonzero(assignments == bucket)
        if candidates.size < tasks_per_bucket:
            raise ValueError(
                f"bucket {bucket} has {candidates.size} tasks; need {tasks_per_bucket}"
            )
        selected[bucket] = rng.choice(candidates, size=tasks_per_bucket, replace=False)
    return selected


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_candidate_rewards(
    coverage: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target_values = rng.uniform(-1.0, 1.0, size=(count, coverage.shape[1]))
    rewards = (2.0 * coverage[None, :, :] - 1.0) * target_values[:, None, :]
    return np.stack([unit_interval_leader_reward(reward) for reward in rewards])


def align_common_checkpoints(
    traces: dict[MethodKey, list[SeedTrace]],
) -> tuple[dict[MethodKey, list[SeedTrace]], np.ndarray]:
    """Align protocols to their shared positive observed-sample checkpoints.

    No-warm-up Active IRL includes an estimate at zero observations, whereas a
    warm-up method cannot.  Comparing the methods therefore requires an explicit
    intersection, rather than assuming their stored checkpoint arrays coincide.
    """

    if any(not traces.get(key) for key in METHODS):
        raise ValueError("every pilot protocol must contain at least one seed trace")
    checkpoints = traces[next(iter(METHODS))][0].checkpoints.copy()
    for key in METHODS:
        for trace in traces[key]:
            checkpoints = np.intersect1d(checkpoints, trace.checkpoints)
    checkpoints = checkpoints[checkpoints > 0]
    if checkpoints.size == 0:
        raise ValueError("pilot protocols have no common positive checkpoint")

    aligned: dict[MethodKey, list[SeedTrace]] = {}
    for key in METHODS:
        aligned[key] = []
        for trace in traces[key]:
            positions = np.searchsorted(trace.checkpoints, checkpoints)
            if not np.array_equal(trace.checkpoints[positions], checkpoints):
                raise ValueError(f"failed to align pilot checkpoints for {key}")
            aligned[key].append(
                SeedTrace(
                    seed=trace.seed,
                    checkpoints=checkpoints.copy(),
                    metrics={
                        name: values[positions].copy() for name, values in trace.metrics.items()
                    },
                    omega_history=trace.omega_history[positions].copy(),
                )
            )
    return aligned, checkpoints


def _candidate_manifest(
    *,
    runtime: Any,
    rewards: np.ndarray,
    leader_config: LeaderOptimizationConfig,
    task_seed: int,
    tasks_per_bucket: int,
    evaluate_all: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    source_policy = np.asarray(runtime.fixed_mle_leader_policy, dtype=np.float64)
    policies = np.empty(
        (rewards.shape[0], runtime.problem.n_leader_actions),
        dtype=np.float64,
    )
    values = np.empty(rewards.shape[0], dtype=np.float64)
    tv = np.empty(rewards.shape[0], dtype=np.float64)
    gap = np.empty(rewards.shape[0], dtype=np.float64)
    started = perf_counter()
    for task_index, reward in enumerate(rewards):
        solution = optimize_leader_policy(
            runtime.problem,
            runtime.true_omega,
            reward,
            leader_config,
            seed=task_seed * 10_000 + task_index,
        )
        policies[task_index] = solution.policy
        values[task_index] = solution.value
        source_value = leader_value_and_gradient(
            runtime.problem,
            runtime.true_omega,
            reward,
            source_policy,
        )[0]
        tv[task_index] = policy_total_variation(solution.policy, source_policy)
        gap[task_index] = source_policy_gap(
            oracle_value=solution.value,
            source_policy_value=source_value,
        )

    metrics = {"tv": tv, "gap": gap}
    bucket_payload: dict[str, Any] = {}
    selected: list[int] = []
    assignments: dict[str, np.ndarray] = {}
    for metric_index, (name, metric_values) in enumerate(metrics.items()):
        edges, metric_assignments = quantile_bucket_assignments(metric_values)
        if evaluate_all:
            metric_selected = [
                np.flatnonzero(metric_assignments == bucket).tolist()
                for bucket in range(len(BUCKET_NAMES))
            ]
        else:
            metric_selected = select_bucket_tasks(
                metric_assignments,
                tasks_per_bucket=tasks_per_bucket,
                seed=task_seed + 100 + metric_index,
            ).tolist()
        assignments[name] = metric_assignments
        selected.extend(task for bucket_tasks in metric_selected for task in bucket_tasks)
        bucket_payload[name] = {
            "edges": edges.tolist(),
            "assignments": metric_assignments.tolist(),
            "selected_indices": metric_selected,
        }
    correlation = spearmanr(tv, gap)
    manifest = {
        "schema_version": 1,
        "candidate_count": int(rewards.shape[0]),
        "task_seed": task_seed,
        "selection_mode": "all_candidates" if evaluate_all else "sampled_within_bucket",
        "tasks_per_bucket": None if evaluate_all else tasks_per_bucket,
        "reward_family": (
            "unit-interval structured coverage rewards with signed target values "
            "sampled uniformly on [-1,1] before positive-affine normalization"
        ),
        "source_policy": source_policy.tolist(),
        "candidate_metrics": {
            "tv": tv.tolist(),
            "gap": gap.tolist(),
        },
        "buckets": bucket_payload,
        "selected_union": sorted(set(selected)),
        "spearman_tv_gap": {
            "correlation": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        },
        "oracle_policy_seconds": perf_counter() - started,
    }
    assignment_matrix = np.stack([assignments[name] for name in METRICS])
    return manifest, policies, assignment_matrix


def _evaluate_selected_tasks(
    *,
    runtime: Any,
    traces: dict[MethodKey, list[SeedTrace]],
    rewards: np.ndarray,
    selected: list[int],
    leader_config: LeaderOptimizationConfig,
    task_seed: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for task_index in selected:
        for key, slug in METHODS.items():
            suboptimality, fisher_error = task_transfer_metrics(
                runtime.problem,
                runtime.true_omega,
                rewards[task_index],
                leader_config,
                traces[key],
                task_seed=task_seed,
                task_index=task_index,
            )
            arrays[f"task_{task_index}__{slug}__suboptimality"] = suboptimality
            arrays[f"task_{task_index}__{slug}__fisher_error"] = fisher_error
        print(f"evaluated task {task_index}", flush=True)
    return arrays


def _heatmap_values(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, np.ndarray],
    metric: str,
    slug: str,
) -> np.ndarray:
    selected = manifest["buckets"][metric]["selected_indices"]
    rows = []
    for bucket_tasks in selected:
        values = np.stack(
            [evaluation[f"task_{task}__{slug}__suboptimality"] for task in bucket_tasks]
        )
        rows.append(values.mean(axis=(0, 1)))
    return np.asarray(rows)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return {"correlation": None, "pvalue": None}
    result = spearmanr(left, right)
    return {"correlation": float(result.statistic), "pvalue": float(result.pvalue)}


def _mean_bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 2_000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def _distribution_summary(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    ordered = np.sort(values)[::-1]
    top_count = max(1, int(np.ceil(0.1 * values.size)))
    total = float(np.sum(values))
    bootstrap_low, bootstrap_high = _mean_bootstrap_interval(values, seed=seed)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("minimum", "p10", "p25", "median", "p75", "p90", "maximum"),
                quantiles,
                strict=True,
            )
        },
        "mean_task_bootstrap_95": [bootstrap_low, bootstrap_high],
        "largest_task_share_of_sum": float(ordered[0] / total) if total > 0.0 else 0.0,
        "worst_ten_percent_share_of_sum": (
            float(np.sum(ordered[:top_count]) / total) if total > 0.0 else 0.0
        ),
    }


def _task_mean_matrix(
    evaluation: dict[str, np.ndarray],
    *,
    task_indices: list[int],
    slug: str,
    quantity: str,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(evaluation[f"task_{task}__{slug}__{quantity}"], dtype=np.float64).mean(
                axis=0
            )
            for task in task_indices
        ]
    )


def _policy_descriptor(policy: np.ndarray) -> dict[str, Any]:
    policy = np.asarray(policy, dtype=np.float64)
    positive = policy[policy > 1.0e-8]
    return {
        "support_size": int(positive.size),
        "maximum_probability": float(np.max(policy)),
        "entropy": float(-np.sum(positive * np.log(positive))),
        "dominant_action": int(np.argmax(policy)),
    }


def _extreme_descriptors(
    *,
    task_indices: list[int],
    final_task_means: np.ndarray,
    evaluation: dict[str, np.ndarray],
    slug: str,
    metrics: dict[str, np.ndarray],
    oracle_policies: np.ndarray,
) -> dict[str, Any]:
    order = np.argsort(final_task_means)
    positions = {
        "minimum": int(order[0]),
        "median": int(order[len(order) // 2]),
        "maximum": int(order[-1]),
    }
    payload = {}
    for label, local_index in positions.items():
        task_index = task_indices[local_index]
        seed_values = np.asarray(
            evaluation[f"task_{task_index}__{slug}__suboptimality"],
            dtype=np.float64,
        )[:, -1]
        payload[label] = {
            "task_index": task_index,
            "mean_suboptimality": float(np.mean(seed_values)),
            "seed_p10_p90": [float(item) for item in np.quantile(seed_values, [0.1, 0.9])],
            "tv": float(metrics["tv"][task_index]),
            "gap": float(metrics["gap"][task_index]),
            "oracle_policy": _policy_descriptor(oracle_policies[task_index]),
        }
    return payload


def _dense_diagnostics(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, np.ndarray],
    oracle_policies: np.ndarray,
    checkpoints: np.ndarray,
    traces: dict[MethodKey, list[SeedTrace]],
    true_omega: np.ndarray,
) -> dict[str, Any]:
    selected = [int(item) for item in manifest["selected_union"]]
    metrics = {
        name: np.asarray(manifest["candidate_metrics"][name], dtype=np.float64) for name in METRICS
    }
    policy_descriptors = [_policy_descriptor(policy) for policy in oracle_policies]
    pure = np.asarray([item["support_size"] == 1 for item in policy_descriptors])
    diagnostics: dict[str, Any] = {
        "task_count": len(selected),
        "oracle_policy_structure": {
            "pure_fraction": float(np.mean(pure[selected])),
            "support_size_counts": {
                str(size): int(
                    np.sum(
                        [policy_descriptors[index]["support_size"] == size for index in selected]
                    )
                )
                for size in sorted(
                    {policy_descriptors[index]["support_size"] for index in selected}
                )
            },
        },
        "coefficient_errors": {},
        "methods": {},
    }
    for key, slug in METHODS.items():
        errors = np.stack([trace.omega_history[-1] - true_omega for trace in traces[key]])
        diagnostics["coefficient_errors"][slug] = {
            "mean_error": np.mean(errors, axis=0).tolist(),
            "mean_l2": float(np.mean(np.linalg.norm(errors, axis=1))),
            "component_rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        }
        suboptimality = _task_mean_matrix(
            evaluation,
            task_indices=selected,
            slug=slug,
            quantity="suboptimality",
        )
        fisher = _task_mean_matrix(
            evaluation,
            task_indices=selected,
            slug=slug,
            quantity="fisher_error",
        )
        method_payload: dict[str, Any] = {
            "correlations": {},
            "buckets": {},
        }
        for checkpoint_position, checkpoint in enumerate(checkpoints):
            method_payload["correlations"][str(int(checkpoint))] = {
                "tv_vs_suboptimality": _safe_spearman(
                    metrics["tv"][selected], suboptimality[:, checkpoint_position]
                ),
                "gap_vs_suboptimality": _safe_spearman(
                    metrics["gap"][selected], suboptimality[:, checkpoint_position]
                ),
                "fisher_vs_suboptimality": _safe_spearman(
                    fisher[:, checkpoint_position], suboptimality[:, checkpoint_position]
                ),
            }
        for metric_index, metric in enumerate(METRICS):
            metric_payload = {}
            for bucket, bucket_name in enumerate(BUCKET_NAMES):
                bucket_tasks = [
                    int(task) for task in manifest["buckets"][metric]["selected_indices"][bucket]
                ]
                local_positions = [selected.index(task) for task in bucket_tasks]
                values = suboptimality[local_positions]
                other_metric = "gap" if metric == "tv" else "tv"
                bucket_payload = {
                    "task_indices": bucket_tasks,
                    "metric_range": [
                        float(np.min(metrics[metric][bucket_tasks])),
                        float(np.max(metrics[metric][bucket_tasks])),
                    ],
                    "other_metric_quantiles": [
                        float(item)
                        for item in np.quantile(
                            metrics[other_metric][bucket_tasks], [0.0, 0.5, 1.0]
                        )
                    ],
                    "pure_oracle_fraction": float(np.mean(pure[bucket_tasks])),
                    "checkpoints": {},
                    "final_extremes": _extreme_descriptors(
                        task_indices=bucket_tasks,
                        final_task_means=values[:, -1],
                        evaluation=evaluation,
                        slug=slug,
                        metrics=metrics,
                        oracle_policies=oracle_policies,
                    ),
                    "within_bucket_other_metric_vs_final_suboptimality": _safe_spearman(
                        metrics[other_metric][bucket_tasks], values[:, -1]
                    ),
                }
                for checkpoint_position, checkpoint in enumerate(checkpoints):
                    bucket_payload["checkpoints"][str(int(checkpoint))] = _distribution_summary(
                        values[:, checkpoint_position],
                        seed=(
                            90_000
                            + metric_index * 10_000
                            + list(METHODS.values()).index(slug) * 1_000
                            + bucket * 100
                            + checkpoint_position
                        ),
                    )
                metric_payload[bucket_name] = bucket_payload
            method_payload["buckets"][metric] = metric_payload
        diagnostics["methods"][slug] = method_payload
    return diagnostics


def _render_task_dispersion(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.4, 5.0), constrained_layout=True)
    rng = np.random.default_rng(72_001)
    titles = {
        "active_warm": "Active, warm-up",
        "active_no_warm": "Active, no warm-up",
        "fixed_expert_mle": "Fixed-expert MLE",
    }
    for row, metric in enumerate(METRICS):
        for column, slug in enumerate(METHODS.values()):
            axis = axes[row, column]
            groups = []
            for bucket in range(len(BUCKET_NAMES)):
                tasks = [
                    int(task) for task in manifest["buckets"][metric]["selected_indices"][bucket]
                ]
                groups.append(
                    _task_mean_matrix(
                        evaluation,
                        task_indices=tasks,
                        slug=slug,
                        quantity="suboptimality",
                    )[:, -1]
                )
            axis.boxplot(groups, showfliers=False, widths=0.55)
            for position, values in enumerate(groups, start=1):
                jitter = rng.uniform(-0.16, 0.16, size=values.size)
                axis.scatter(
                    position + jitter,
                    values,
                    s=9,
                    color="#2F6C8F",
                    alpha=0.52,
                    linewidths=0,
                )
            axis.set_yscale("log")
            axis.set_xticks(range(1, len(BUCKET_NAMES) + 1), BUCKET_NAMES)
            axis.set_title(titles[slug], fontsize=9)
            if row == 1:
                axis.set_xlabel("transfer-distance bucket")
            if column == 0:
                axis.set_ylabel(f"{metric.upper()}: task-mean SubOpt")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _render_task_scatter(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, np.ndarray],
    oracle_policies: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.4, 5.0), constrained_layout=True)
    pure = np.asarray(
        [_policy_descriptor(policy)["support_size"] == 1 for policy in oracle_policies]
    )
    titles = {
        "active_warm": "Active, warm-up",
        "active_no_warm": "Active, no warm-up",
        "fixed_expert_mle": "Fixed-expert MLE",
    }
    for row, metric in enumerate(METRICS):
        all_distances = np.asarray(manifest["candidate_metrics"][metric], dtype=np.float64)
        task_indices = [int(task) for task in manifest["selected_union"]]
        distances = all_distances[task_indices]
        selected_pure = pure[task_indices]
        for column, slug in enumerate(METHODS.values()):
            axis = axes[row, column]
            values = _task_mean_matrix(
                evaluation,
                task_indices=task_indices,
                slug=slug,
                quantity="suboptimality",
            )[:, -1]
            axis.scatter(
                distances[~selected_pure],
                values[~selected_pure],
                s=13,
                color="#2F6C8F",
                alpha=0.58,
                linewidths=0,
                label="mixed optimum",
            )
            axis.scatter(
                distances[selected_pure],
                values[selected_pure],
                s=16,
                color="#D07A1D",
                alpha=0.65,
                linewidths=0,
                label="pure optimum",
            )
            correlation = _safe_spearman(distances, values)["correlation"]
            correlation_label = "NA" if correlation is None else f"{correlation:.2f}"
            axis.set_title(f"{titles[slug]} ($\\rho$={correlation_label})", fontsize=9)
            axis.set_yscale("log")
            if row == 1:
                axis.set_xlabel(f"{metric.upper()} from source task")
            if column == 0:
                axis.set_ylabel("task-mean SubOpt at final checkpoint")
            if row == 0 and column == 2:
                axis.legend(frameon=False, fontsize=7)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _render_heatmaps(
    *,
    manifest: dict[str, Any],
    evaluation: dict[str, np.ndarray],
    checkpoints: np.ndarray,
    output_path: Path,
) -> dict[str, list[list[float]]]:
    values = {
        metric: {
            slug: _heatmap_values(
                manifest=manifest,
                evaluation=evaluation,
                metric=metric,
                slug=slug,
            )
            for slug in METHODS.values()
        }
        for metric in METRICS
    }
    positive = np.concatenate(
        [array.ravel() for metric_values in values.values() for array in metric_values.values()]
    )
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("all pilot SubOpt values are zero")
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    figure, axes = plt.subplots(2, 3, figsize=(9.4, 4.8), constrained_layout=True)
    titles = {
        "active_warm": "Active, warm-up",
        "active_no_warm": "Active, no warm-up",
        "fixed_expert_mle": "Fixed-expert MLE",
    }
    image = None
    for row, metric in enumerate(METRICS):
        for column, slug in enumerate(METHODS.values()):
            axis = axes[row, column]
            image = axis.imshow(values[metric][slug], aspect="auto", origin="lower", norm=norm)
            axis.set_title(titles[slug], fontsize=9)
            axis.set_xticks(range(checkpoints.size), checkpoints)
            axis.set_yticks(range(len(BUCKET_NAMES)), BUCKET_NAMES)
            if row == 1:
                axis.set_xlabel("follower samples")
            if column == 0:
                axis.set_ylabel(f"{metric.upper()} bucket")
    assert image is not None
    figure.colorbar(image, ax=axes, label="mean transfer SubOpt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        f"{metric}__{slug}": values[metric][slug].tolist()
        for metric in METRICS
        for slug in METHODS.values()
    }


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty pilot output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir.resolve()
    traces = load_run_traces(run_dir)
    missing = set(METHODS) - set(traces)
    if missing:
        raise ValueError(f"reduced run is missing pilot protocols: {sorted(missing)}")
    traces = {key: traces[key] for key in METHODS}
    traces, checkpoints = align_common_checkpoints(traces)
    runtime = load_runtime(run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    coverage = np.asarray(runtime.metadata["coverage_indicator"], dtype=np.float64)
    rewards = _normalized_candidate_rewards(
        coverage,
        count=args.candidate_count,
        seed=args.task_seed,
    )
    manifest, oracle_policies, assignments = _candidate_manifest(
        runtime=runtime,
        rewards=rewards,
        leader_config=leader_config,
        task_seed=args.task_seed,
        tasks_per_bucket=args.tasks_per_bucket,
        evaluate_all=args.evaluate_all,
    )
    manifest["source_run"] = {
        "run_id": json.loads((run_dir / "manifest.json").read_text())["run_id"],
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "checkpoints": checkpoints.tolist(),
        "methods": [slug for slug in METHODS.values()],
    }
    manifest["evaluation_git"] = collect_git_provenance(find_repository(Path.cwd())).manifest()
    atomic_write_json(output_dir / "task_manifest.json", manifest)
    atomic_write_npz(
        output_dir / "candidate_tasks.npz",
        {
            "rewards": rewards,
            "oracle_policies": oracle_policies,
            "metric_assignments": assignments,
        },
    )

    evaluation_started = perf_counter()
    evaluation = _evaluate_selected_tasks(
        runtime=runtime,
        traces=traces,
        rewards=rewards,
        selected=manifest["selected_union"],
        leader_config=leader_config,
        task_seed=args.task_seed,
    )
    evaluation["checkpoints"] = checkpoints
    atomic_write_npz(output_dir / "evaluation.npz", evaluation)
    heatmaps = _render_heatmaps(
        manifest=manifest,
        evaluation=evaluation,
        checkpoints=checkpoints,
        output_path=output_dir / "heatmaps.png",
    )
    diagnostics = _dense_diagnostics(
        manifest=manifest,
        evaluation=evaluation,
        oracle_policies=oracle_policies,
        checkpoints=checkpoints,
        traces=traces,
        true_omega=np.asarray(runtime.true_omega, dtype=np.float64),
    )
    atomic_write_json(output_dir / "diagnostics.json", diagnostics)
    _render_task_dispersion(
        manifest=manifest,
        evaluation=evaluation,
        output_path=output_dir / "task_dispersion.png",
    )
    _render_task_scatter(
        manifest=manifest,
        evaluation=evaluation,
        oracle_policies=oracle_policies,
        output_path=output_dir / "task_scatter.png",
    )
    summary = {
        "schema_version": 1,
        "source_run": manifest["source_run"],
        "candidate_count": args.candidate_count,
        "selection_mode": manifest["selection_mode"],
        "selected_task_count": len(manifest["selected_union"]),
        "tasks_per_bucket": manifest["tasks_per_bucket"],
        "spearman_tv_gap": manifest["spearman_tv_gap"],
        "oracle_policy_seconds": manifest["oracle_policy_seconds"],
        "evaluation_seconds": perf_counter() - evaluation_started,
        "heatmap_values": heatmaps,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=36)
    parser.add_argument("--tasks-per-bucket", type=int, default=4)
    parser.add_argument("--task-seed", type=int, default=63_001)
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help="evaluate every candidate and use every task in each bucket mean",
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
