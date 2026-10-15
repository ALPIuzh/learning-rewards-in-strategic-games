"""Export dense, source-relative transfer diagnostics for every Lobeke method.

The expensive learning trajectories are immutable inputs.  This module only
replays retained coefficient estimates on a fixed, algorithm-blind reward
suite, caching each method/task evaluation atomically so interrupted runs are
safe to resume.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from experiments.guard.paper.figures.common import (
    TRANSFER_CACHE_SCHEMA_VERSION,
    _array_sha256,
    _cached_transfer_metrics,
    _json_sha256,
    _portable_path,
    _source_record,
)
from experiments.guard.paper.figures.corrected_traces import comparison_config, load_figure_run_traces
from experiments.guard.paper.figures.data import METHOD_SLUGS, SeedTrace, load_runtime
from experiments.guard.paper.figures.task_sets import (
    BEHAVIORAL_TRANSFER_TASKS,
    StructuredTaskSet,
    normalized_structured_tasks,
    policy_total_variation,
    quantile_bucket_assignments,
    source_policy_gap,
)
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "data" / "lobeke" / "behavioral_transfer_v1"
DEFAULT_CACHE = REPOSITORY_ROOT / "artifacts" / "derived" / "guard-lobeke-behavioral-transfer"
DEFAULT_CHECKPOINTS = (200, 400, 800, 1_200, 1_600)
BUCKET_NAMES = ("low", "medium", "high")
METRICS = ("tv", "gap")


def _at_checkpoints(traces: list[SeedTrace], requested: tuple[int, ...]) -> list[SeedTrace]:
    """Copy traces at an explicit shared checkpoint grid."""

    if not requested or tuple(sorted(set(requested))) != requested or requested[0] <= 0:
        raise ValueError("requested checkpoints must be unique, increasing, and positive")
    selected: list[SeedTrace] = []
    target = np.asarray(requested, dtype=np.int64)
    for trace in traces:
        positions = np.searchsorted(trace.checkpoints, target)
        if np.any(positions >= trace.checkpoints.size) or not np.array_equal(trace.checkpoints[positions], target):
            raise ValueError(f"trace seed {trace.seed} is missing a requested checkpoint")
        selected.append(
            SeedTrace(
                seed=trace.seed,
                checkpoints=target.copy(),
                metrics={name: values[positions].copy() for name, values in trace.metrics.items()},
                omega_history=trace.omega_history[positions].copy(),
            )
        )
    return selected


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a finite rank correlation, or ``None`` for a constant input."""

    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    result = spearmanr(left, right)
    return None if not np.isfinite(result.statistic) else float(result.statistic)


def _task_geometry(runtime: Any, tasks: np.ndarray, leader_config: LeaderOptimizationConfig,
        *, task_seed: int) -> dict[str, np.ndarray]:  # fmt: skip
    """Compute source-relative, performance-free geometry for a reward suite."""

    source_reward = runtime.metadata.get("defender_reward")
    if source_reward is None:
        raise ValueError("application runtime is missing the source leader reward")
    source_solution = optimize_leader_policy(
        runtime.problem,
        runtime.true_omega,
        np.asarray(source_reward, dtype=np.float64),
        leader_config,
        seed=task_seed - 1,
    )
    oracle_policies = np.empty((tasks.shape[0], runtime.problem.n_leader_actions), dtype=np.float64)
    oracle_values = np.empty(tasks.shape[0], dtype=np.float64)
    tv = np.empty(tasks.shape[0], dtype=np.float64)
    gap = np.empty(tasks.shape[0], dtype=np.float64)
    for index, reward in enumerate(tasks):
        oracle = optimize_leader_policy(
            runtime.problem, runtime.true_omega, reward, leader_config, seed=task_seed + index
        )
        source_value = leader_value_and_gradient(runtime.problem, runtime.true_omega, reward, source_solution.policy)[0]
        oracle_policies[index] = oracle.policy
        oracle_values[index] = oracle.value
        tv[index] = policy_total_variation(oracle.policy, source_solution.policy)
        gap[index] = source_policy_gap(oracle_value=oracle.value, source_policy_value=source_value)
    return {
        "source_policy": np.asarray(source_solution.policy, dtype=np.float64),
        "oracle_policies": oracle_policies,
        "oracle_values": oracle_values,
        "tv": tv,
        "gap": gap,
    }


def _task_concentration(values: np.ndarray) -> dict[str, float]:
    """Summarize whether a bucket mean is dominated by a few tasks."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = float(np.sum(ordered))
    worst_count = max(1, int(np.ceil(0.1 * ordered.size)))
    if total <= 0.0:
        return {"largest_task_share": 0.0, "worst_ten_percent_share": 0.0}
    return {
        "largest_task_share": float(ordered[0] / total),
        "worst_ten_percent_share": float(np.sum(ordered[:worst_count]) / total),
    }


def _bucket_diagnostics(*, assignments: dict[str, np.ndarray], geometry: dict[str, np.ndarray],
        method_values: dict[str, np.ndarray]) -> dict[str, object]:  # fmt: skip
    """Describe cell means, within-bucket spread, and extreme tasks."""

    payload: dict[str, object] = {}
    for metric_name in METRICS:
        metric_values = geometry[metric_name]
        other_name = "gap" if metric_name == "tv" else "tv"
        metric_payload: dict[str, object] = {}
        for slug, values in method_values.items():
            bucket_payload: dict[str, object] = {}
            final_task_means = values[:, :, -1].mean(axis=1)
            for bucket_index, bucket_name in enumerate(BUCKET_NAMES):
                indices = np.flatnonzero(assignments[metric_name] == bucket_index)
                bucket_task_means = final_task_means[indices]
                order = np.argsort(bucket_task_means)
                representatives = {
                    "minimum": int(indices[order[0]]),
                    "median": int(indices[order[len(order) // 2]]),
                    "maximum": int(indices[order[-1]]),
                }
                bucket_payload[bucket_name] = {
                    "count": int(indices.size),
                    "distance_range": [float(np.min(metric_values[indices])), float(np.max(metric_values[indices]))],
                    "other_distance_quantiles": np.quantile(geometry[other_name][indices], (0.0, 0.5, 1.0)).tolist(),
                    "final_task_suboptimality_quantiles": np.quantile(
                        bucket_task_means, (0.0, 0.1, 0.5, 0.9, 1.0)
                    ).tolist(),
                    "concentration": _task_concentration(bucket_task_means),
                    "other_distance_vs_final_suboptimality_spearman": _finite_spearman(
                        geometry[other_name][indices], bucket_task_means
                    ),
                    "representative_task_indices": representatives,
                }
            metric_payload[slug] = bucket_payload
        payload[metric_name] = metric_payload
    return payload


def _source_records(run_dirs: tuple[Path, ...]) -> tuple[list[dict[str, object]], str]:
    """Validate immutable run composition and return its identity."""

    manifests = [json.loads((path / "manifest.json").read_text()) for path in run_dirs]
    records = [_source_record(path, manifest) for path, manifest in zip(run_dirs, manifests, strict=True)]
    if any(record["git_dirty"] for record in records):
        raise ValueError("behavioral-transfer sources must come from clean worktrees")
    commits = {str(record["git_commit"]) for record in records}
    if len(commits) != 1:
        raise ValueError("behavioral-transfer sources use different commits")
    configs = {_json_sha256(comparison_config(path)) for path in run_dirs}
    if len(configs) != 1:
        raise ValueError("behavioral-transfer sources use different data-generating configs")
    return records, _json_sha256([record["run_id"] for record in records])[:16]


def export(*, run_dirs: tuple[Path, ...], output_dir: Path | None, cache_dir: Path,
        task_set: StructuredTaskSet = BEHAVIORAL_TRANSFER_TASKS,
        checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS, methods: tuple[str, ...] = tuple(METHOD_SLUGS.values()),
        cache_only: bool = False) -> tuple[Path, Path] | None:  # fmt: skip
    """Evaluate selected methods and optionally write the complete dense export."""

    if not run_dirs:
        raise ValueError("at least one source run is required")
    unknown = sorted(set(methods) - set(METHOD_SLUGS.values()))
    if unknown:
        raise ValueError(f"unknown behavioral-transfer methods: {unknown}")
    if not methods:
        raise ValueError("at least one method is required")
    if not cache_only and set(methods) != set(METHOD_SLUGS.values()):
        raise ValueError("a complete export requires all nine methods")
    if task_set.candidate_pool != task_set.count:
        raise ValueError("dense behavioral suites must evaluate every i.i.d. candidate")

    records, composition_id = _source_records(run_dirs)
    traces = load_figure_run_traces(run_dirs)
    if set(traces) != set(METHOD_SLUGS):
        raise ValueError(f"unexpected Lobeke protocols: {sorted(traces)}")
    primary_run = run_dirs[0]
    runtime = load_runtime(primary_run)
    cfg = OmegaConf.load(primary_run / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    tasks = normalized_structured_tasks(runtime, task_set)
    geometry = _task_geometry(runtime, tasks, leader_config, task_seed=task_set.seed)
    assignments: dict[str, np.ndarray] = {}
    edges: dict[str, np.ndarray] = {}
    for metric_name in METRICS:
        edges[metric_name], assignments[metric_name] = quantile_bucket_assignments(geometry[metric_name])

    cache_namespace = (
        cache_dir
        / f"v{TRANSFER_CACHE_SCHEMA_VERSION}"
        / composition_id
        / f"{task_set.name}-seed{task_set.seed}-n{task_set.count}-k{len(checkpoints)}"
    )
    selected_by_slug = {slug: key for key, slug in METHOD_SLUGS.items() if slug in methods}
    method_values: dict[str, np.ndarray] = {}
    method_fisher: dict[str, np.ndarray] = {}
    common_seeds: tuple[int, ...] | None = None
    started = perf_counter()
    for slug, key in selected_by_slug.items():
        method_traces = _at_checkpoints(traces[key], checkpoints)
        seeds = tuple(trace.seed for trace in method_traces)
        if common_seeds is None:
            common_seeds = seeds
        elif seeds != common_seeds:
            raise ValueError(f"unpaired dense-transfer seeds for {key}")
        method_started = perf_counter()
        suboptimality, fisher_error = _cached_transfer_metrics(
            cache_dir=cache_namespace,
            slug=slug,
            runtime=runtime,
            traces=method_traces,
            leader_config=leader_config,
            transfer_tasks=tasks,
            task_seed=task_set.seed,
            task_label="dense transfer task",
            progress_every=max(1, min(20, task_set.count)),
        )
        method_values[slug] = suboptimality
        method_fisher[slug] = fisher_error
        print(f"{slug}: completed in {perf_counter() - method_started:.1f}s", flush=True)

    if cache_only:
        print(f"cache-only evaluation completed in {perf_counter() - started:.1f}s", flush=True)
        return None
    assert common_seeds is not None and output_dir is not None

    arrays: dict[str, np.ndarray] = {
        "tasks": tasks,
        "source_policy": geometry["source_policy"],
        "oracle_policies": geometry["oracle_policies"],
        "oracle_values": geometry["oracle_values"],
        "tv": geometry["tv"],
        "gap": geometry["gap"],
        "tv_bucket_edges": edges["tv"],
        "gap_bucket_edges": edges["gap"],
        "tv_bucket_assignments": assignments["tv"],
        "gap_bucket_assignments": assignments["gap"],
        "checkpoints": np.asarray(checkpoints, dtype=np.int64),
        "seeds": np.asarray(common_seeds, dtype=np.int64),
    }
    for slug in METHOD_SLUGS.values():
        arrays[f"{slug}__taskwise_transfer_suboptimality"] = method_values[slug]
        arrays[f"{slug}__taskwise_transfer_fisher_error"] = method_fisher[slug]
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"dense behavioral-transfer array {name} is non-finite")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite dense transfer export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"
    atomic_write_npz(data_path, arrays)
    bucket_counts = {name: np.bincount(assignments[name], minlength=len(BUCKET_NAMES)).tolist() for name in METRICS}
    provenance = {
        "schema_version": 1,
        "experiment": "Lobeke dense source-relative behavioral transfer",
        "purpose": ("algorithm-blind transfer-distance heatmaps from retained current-protocol coefficient histories"),
        "source": {
            "composition_id": composition_id,
            "git_commit": records[0]["git_commit"],
            "result_count": sum(int(record["result_count"]) for record in records),
            "runs": records,
        },
        "task_policy": {
            **asdict(task_set),
            "candidate_distribution": (
                "independent target priorities z_b ~ Uniform[-1,1], evaluated for every "
                "candidate without performance-based selection"
            ),
            "reward_definition": "r_z(a,b)=(2*c_ab-1)*z_b",
            "normalization": "positive-affine normalization to [0,1] per task; R_max=1",
            "disjoint_seeds": {
                "hyperparameter_tuning": 9_400,
                "ten_task_final_diagnostic": 202_609_700,
                "lambda_selection": 9_400,
                "lambda_confirmation": 9_500,
            },
        },
        "distance_metrics": {
            "tv": "half the L1 distance between true source- and task-optimal leader policies",
            "gap": (
                "task-optimal value minus the value of deploying the true source-optimal leader policy on that task"
            ),
            "performance_free": True,
            "bucket_rule": "tie-aware empirical terciles fixed before method evaluation",
            "bucket_names": list(BUCKET_NAMES),
            "bucket_counts": bucket_counts,
            "edges": {name: edges[name].tolist() for name in METRICS},
            "spearman_tv_gap": _finite_spearman(geometry["tv"], geometry["gap"]),
            "tv_saturation_count": int(np.sum(np.isclose(geometry["tv"], 1.0, atol=1e-10))),
        },
        "evaluation": {
            "checkpoints": list(checkpoints),
            "seed_count": len(common_seeds),
            "seeds": list(common_seeds),
            "methods": {slug: {"method": key[0], "warmup_enabled": key[1]} for key, slug in METHOD_SLUGS.items()},
            "cell_statistic": "arithmetic mean over every task in a bucket and all 50 seeds",
            "leader_optimizer": asdict(leader_config),
        },
        "diagnostics": _bucket_diagnostics(assignments=assignments, geometry=geometry, method_values=method_values),
        "cache": {
            "schema_version": TRANSFER_CACHE_SCHEMA_VERSION,
            "namespace": _portable_path(cache_namespace),
            "taskwise_atomic": True,
        },
        "arrays": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": _array_sha256(values)}
            for name, values in arrays.items()
        },
        "data_sha256": sha256_file(data_path),
        "elapsed_seconds": perf_counter() - started,
        "generation_command": (
            "python -m experiments.guard.paper.figures.export_behavioral_transfer "
            + " ".join(f"--run-dir {_portable_path(path)}" for path in run_dirs)
            + f" --output-dir {_portable_path(output_dir)}"
        ),
    }
    provenance_path = output_dir / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    return data_path, provenance_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--task-count", type=int, default=BEHAVIORAL_TRANSFER_TASKS.count)
    parser.add_argument("--task-seed", type=int, default=BEHAVIORAL_TRANSFER_TASKS.seed)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--method", action="append", choices=tuple(METHOD_SLUGS.values()))
    parser.add_argument("--cache-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    task_set = StructuredTaskSet(
        name=(
            BEHAVIORAL_TRANSFER_TASKS.name
            if args.task_count == BEHAVIORAL_TRANSFER_TASKS.count and args.task_seed == BEHAVIORAL_TRANSFER_TASKS.seed
            else "behavioral_transfer_smoke"
        ),
        count=args.task_count,
        seed=args.task_seed,
        candidate_pool=args.task_count,
    )
    result = export(
        run_dirs=tuple(args.run_dir),
        output_dir=None if args.cache_only else args.output_dir,
        cache_dir=args.cache_dir,
        task_set=task_set,
        checkpoints=tuple(args.checkpoints),
        methods=tuple(args.method or METHOD_SLUGS.values()),
        cache_only=args.cache_only,
    )
    if result is not None:
        print(f"wrote {result[0]}")
        print(f"wrote {result[1]}")


if __name__ == "__main__":
    main()
