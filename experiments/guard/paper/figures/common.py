"""Shared provenance and transfer-cache utilities for Lobeke figure exports."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.task_sets import TRANSFER_EVALUATION_TASKS
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.core.math import cumulative_fisher_information_batch
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy_batch
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import atomic_write_npz, sha256_file

TRANSFER_CACHE_SCHEMA_VERSION = 4


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        if "artifacts" in resolved.parts:
            index = resolved.parts.index("artifacts")
            return Path(*resolved.parts[index:]).as_posix()
        return resolved.as_posix()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256() -> str:
    """Hash the transfer implementation and numerical kernels used by the cache."""

    callables = (
        task_transfer_metrics,
        cumulative_fisher_information_batch,
        leader_value_and_gradient,
        optimize_leader_policy_batch,
    )
    sources = sorted({Path(inspect.getfile(item)).resolve() for item in callables})
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _transfer_context_sha256(runtime: Any, leader_config: LeaderOptimizationConfig,
        implementation_sha256: str) -> str:  # fmt: skip
    """Hash every non-task, non-estimate input that affects transfer diagnostics."""

    problem = runtime.problem
    return _json_sha256(
        {
            "features_sha256": _array_sha256(problem.features),
            "true_omega_sha256": _array_sha256(np.asarray(runtime.true_omega)),
            "eta": problem.eta,
            "radius": problem.radius,
            "leader_config": asdict(leader_config),
            "implementation_sha256": implementation_sha256,
        }
    )


def _trace_context_sha256(traces: list[SeedTrace], *, task_seed: int, task_index: int) -> str:
    """Hash trace identity and every seed input used by transfer optimization."""

    seeds = np.asarray([trace.seed for trace in traces], dtype=np.int64)
    checkpoints = np.asarray([trace.checkpoints for trace in traces], dtype=np.int64)
    return _json_sha256(
        {
            "seeds_sha256": _array_sha256(seeds),
            "checkpoints_sha256": _array_sha256(checkpoints),
            "task_seed": task_seed,
            "task_index": task_index,
        }
    )


def _valid_cached_transfer_arrays(suboptimality: np.ndarray, fisher_error: np.ndarray,
        *, expected_shape: tuple[int, int]) -> bool:  # fmt: skip
    """Reject incomplete, stale-shaped, or non-finite cached diagnostics."""

    return (
        suboptimality.shape == expected_shape
        and fisher_error.shape == expected_shape
        and np.issubdtype(suboptimality.dtype, np.floating)
        and np.issubdtype(fisher_error.dtype, np.floating)
        and np.all(np.isfinite(suboptimality))
        and np.all(np.isfinite(fisher_error))
    )


def _cached_transfer_metrics(*, cache_dir: Path, slug: str, runtime: Any, traces: list[SeedTrace],
        leader_config: LeaderOptimizationConfig, transfer_tasks: np.ndarray,
        task_seed: int = TRANSFER_EVALUATION_TASKS.seed, task_label: str = "fixed transfer task",
        progress_every: int = 10) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Load or atomically derive every task/protocol diagnostic."""

    if progress_every <= 0:
        raise ValueError("progress interval must be strictly positive")
    omega_history = np.asarray([trace.omega_history for trace in traces], dtype=np.float64)
    omega_sha256 = _array_sha256(omega_history)
    implementation_sha256 = _implementation_sha256()
    context_sha256 = _transfer_context_sha256(runtime, leader_config, implementation_sha256)
    expected_shape = (len(traces), traces[0].checkpoints.size)
    task_suboptimality: list[np.ndarray] = []
    task_fisher_error: list[np.ndarray] = []
    for task_index, leader_reward in enumerate(transfer_tasks):
        task_sha256 = _array_sha256(leader_reward)
        trace_context_sha256 = _trace_context_sha256(traces, task_seed=task_seed, task_index=task_index)
        cache_path = cache_dir / slug / f"task-{task_index:02d}.npz"
        cached_values: tuple[np.ndarray, np.ndarray] | None = None
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                required_metadata = {
                    "cache_schema_version",
                    "omega_sha256",
                    "task_sha256",
                    "context_sha256",
                    "implementation_sha256",
                    "trace_context_sha256",
                }
                if (
                    required_metadata.issubset(cached.files)
                    and int(cached["cache_schema_version"].item()) == TRANSFER_CACHE_SCHEMA_VERSION
                    and str(cached["omega_sha256"].item()) == omega_sha256
                    and str(cached["task_sha256"].item()) == task_sha256
                    and str(cached["context_sha256"].item()) == context_sha256
                    and str(cached["implementation_sha256"].item()) == implementation_sha256
                    and str(cached["trace_context_sha256"].item()) == trace_context_sha256
                ):
                    suboptimality = cached["suboptimality"].copy()
                    fisher_error = cached["fisher_error"].copy()
                    if _valid_cached_transfer_arrays(suboptimality, fisher_error, expected_shape=expected_shape):
                        cached_values = (suboptimality, fisher_error)
        if cached_values is None:
            suboptimality, fisher_error = task_transfer_metrics(
                runtime.problem,
                runtime.true_omega,
                leader_reward,
                leader_config,
                traces,
                task_seed=task_seed,
                task_index=task_index,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_npz(
                cache_path,
                {
                    "suboptimality": suboptimality,
                    "fisher_error": fisher_error,
                    "omega_sha256": np.asarray(omega_sha256),
                    "task_sha256": np.asarray(task_sha256),
                    "cache_schema_version": np.asarray(TRANSFER_CACHE_SCHEMA_VERSION),
                    "context_sha256": np.asarray(context_sha256),
                    "implementation_sha256": np.asarray(implementation_sha256),
                    "trace_context_sha256": np.asarray(trace_context_sha256),
                },
            )
        else:
            suboptimality, fisher_error = cached_values
        task_suboptimality.append(suboptimality)
        task_fisher_error.append(fisher_error)
        completed = task_index + 1
        if completed == 1 or completed == len(transfer_tasks) or completed % progress_every == 0:
            print(f"{slug}: {task_label} {completed}/{len(transfer_tasks)}", flush=True)
    return np.stack(task_suboptimality, axis=0), np.stack(task_fisher_error, axis=0)


def _result_set_sha256(run_dir: Path, expected_jobs: list[dict[str, object]]) -> str:
    """Hash the ordered ``job_id:result_sha256`` set after checking every status."""

    entries: list[str] = []
    for job in sorted(expected_jobs, key=lambda item: str(item["job_id"])):
        job_id = str(job["job_id"])
        status_path = run_dir / "jobs" / job_id / "status.json"
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            raise ValueError(f"Lobeke job {job_id} is not complete")
        digest = str(status["result_sha256"])
        if sha256_file(run_dir / "jobs" / job_id / "result.npz") != digest:
            raise ValueError(f"Lobeke job {job_id} failed its result checksum")
        entries.append(f"{job_id}:{digest}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def _source_record(run_dir: Path, manifest: dict[str, Any]) -> dict[str, object]:
    """Validate one immutable run and return its complete source record."""

    manifest_path = run_dir / "manifest.json"
    if manifest.get("status") != "complete" or manifest["progress"]["failed"] != 0:
        raise ValueError(f"source run is not complete: {run_dir}")
    methods = sorted({entry["spec"]["method"] for entry in manifest["expected_jobs"]})
    return {
        "study_name": manifest["study_name"],
        "run_id": manifest["run_id"],
        "methods": methods,
        "git_commit": manifest["git"]["commit"],
        "git_dirty": bool(manifest["git"]["dirty"]),
        "semantic_config_sha256": manifest["semantic_config_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "result_set_sha256": _result_set_sha256(run_dir, manifest["expected_jobs"]),
        "result_count": len(manifest["expected_jobs"]),
    }
