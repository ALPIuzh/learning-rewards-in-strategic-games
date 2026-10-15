"""Refit the protected fixed-expert samples under matched MLE ridge values.

This is a read-only counterfactual on the exact 50 final-evaluation trajectories.
It changes no samples, expert policy, task, or deployment optimizer.  The output
path must not already exist, preventing accidental artifact replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.task_sets import (
    TRANSFER_EVALUATION_TASKS,
    normalized_structured_tasks,
)
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.inference.mle import solve_mle_batch
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.artifacts import atomic_write_json

EXPECTED_SEEDS = np.arange(30_000, 30_050, dtype=np.int64)
EXPECTED_SAMPLES = 1_600


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_standard_error(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_error": float(np.std(array, ddof=1) / np.sqrt(array.size)),
        "values": array.tolist(),
    }


def _load_fixed_expert_samples(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("source run is not complete")
    entries = sorted(
        (
            entry
            for entry in manifest["expected_jobs"]
            if entry["spec"]["method"] == "fixed_expert_mle"
        ),
        key=lambda entry: int(entry["spec"]["seed"]),
    )
    seeds = np.asarray([entry["spec"]["seed"] for entry in entries], dtype=np.int64)
    if not np.array_equal(seeds, EXPECTED_SEEDS):
        raise ValueError("fixed-expert jobs do not use the protected final seed set")

    policies: list[np.ndarray] = []
    follower_actions: list[np.ndarray] = []
    for entry in entries:
        job_dir = run_dir / "jobs" / entry["job_id"]
        status = json.loads((job_dir / "status.json").read_text())
        result_path = job_dir / "result.npz"
        if status.get("status") != "complete" or status.get("result_sha256") != _sha256(
            result_path
        ):
            raise ValueError(f"job {entry['job_id']} failed integrity validation")
        with np.load(result_path, allow_pickle=False) as arrays:
            policies.append(arrays["policies"].copy())
            follower_actions.append(arrays["follower_actions"].copy())

    policy_array = np.asarray(policies, dtype=np.float64)
    action_array = np.asarray(follower_actions, dtype=np.int64)
    expected_policy_shape = (EXPECTED_SEEDS.size, EXPECTED_SAMPLES, policy_array.shape[-1])
    if policy_array.shape != expected_policy_shape or action_array.shape != (
        EXPECTED_SEEDS.size,
        EXPECTED_SAMPLES,
    ):
        raise ValueError("fixed-expert sample arrays have unexpected shapes")
    if not np.allclose(policy_array, policy_array[:, :1, :]):
        raise ValueError("fixed-expert policies change within a trajectory")
    if not np.allclose(policy_array, policy_array[:1, :, :]):
        raise ValueError("fixed-expert policies differ across protected seeds")
    return seeds, policy_array, action_array, _sha256(manifest_path)


def _evaluate_source(
    runtime: object,
    estimates: np.ndarray,
    seeds: np.ndarray,
) -> dict[str, object]:
    per_metric: dict[str, list[float]] = {}
    for estimate, seed in zip(estimates, seeds, strict=True):
        metrics = runtime.evaluate(estimate[None, :], int(seed))
        for name, values in metrics.items():
            per_metric.setdefault(name, []).append(float(values[0]))
    return {
        name: _mean_standard_error(np.asarray(values, dtype=np.float64))
        for name, values in per_metric.items()
    }


def _evaluate_transfer(
    runtime: object,
    estimates: np.ndarray,
    seeds: np.ndarray,
    leader_config: LeaderOptimizationConfig,
) -> dict[str, object]:
    tasks = normalized_structured_tasks(runtime, TRANSFER_EVALUATION_TASKS)
    traces = [
        SeedTrace(
            seed=int(seed),
            checkpoints=np.asarray([EXPECTED_SAMPLES], dtype=np.int64),
            metrics={},
            omega_history=estimate[None, :],
        )
        for seed, estimate in zip(seeds, estimates, strict=True)
    ]
    taskwise_suboptimality: list[np.ndarray] = []
    taskwise_fisher: list[np.ndarray] = []
    for task_index, reward in enumerate(tasks):
        suboptimality, fisher = task_transfer_metrics(
            runtime.problem,
            runtime.true_omega,
            reward,
            leader_config,
            traces,
            task_seed=TRANSFER_EVALUATION_TASKS.seed,
            task_index=task_index,
        )
        taskwise_suboptimality.append(suboptimality[:, 0])
        taskwise_fisher.append(fisher[:, 0])
    stacked_suboptimality = np.stack(taskwise_suboptimality)
    stacked_fisher = np.stack(taskwise_fisher)
    return {
        "averaged_transfer_suboptimality": _mean_standard_error(
            np.mean(stacked_suboptimality, axis=0)
        ),
        "averaged_transfer_fisher_error": _mean_standard_error(np.mean(stacked_fisher, axis=0)),
        "taskwise_transfer_suboptimality_mean": np.mean(
            stacked_suboptimality, axis=1
        ).tolist(),
        "taskwise_transfer_fisher_error_mean": np.mean(stacked_fisher, axis=1).tolist(),
    }


def run(run_dir: Path, output: Path, ridges: tuple[float, ...]) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {output}")
    if not ridges or any(not np.isfinite(ridge) or ridge < 0.0 for ridge in ridges):
        raise ValueError("ridges must be finite and nonnegative")

    seeds, policies, actions, manifest_sha256 = _load_fixed_expert_samples(run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    runtime = load_runtime(run_dir)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)

    results: list[dict[str, object]] = []
    for ridge in ridges:
        started = perf_counter()
        fits = solve_mle_batch(
            runtime.problem,
            policies,
            actions,
            replace(numerics.mle, ridge=ridge),
        )
        estimates = np.asarray([fit.omega for fit in fits], dtype=np.float64)
        results.append(
            {
                "ridge": ridge,
                "elapsed_seconds": perf_counter() - started,
                "boundary_fraction": float(
                    np.mean(np.linalg.norm(estimates, axis=1) >= runtime.problem.radius - 1.0e-8)
                ),
                "maximum_projected_gradient_residual": float(
                    max(fit.projected_gradient_residual for fit in fits)
                ),
                "mean_iterations": float(np.mean([fit.iterations for fit in fits])),
                "mean_estimate": np.mean(estimates, axis=0).tolist(),
                "source": _evaluate_source(runtime, estimates, seeds),
                "transfer": _evaluate_transfer(runtime, estimates, seeds, leader_config),
            }
        )
        print(f"completed ridge={ridge:g}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output,
        {
            "schema_version": 1,
            "purpose": "estimator-matched refit of protected fixed-expert samples",
            "source": {
                "run_dir": str(run_dir),
                "manifest_sha256": manifest_sha256,
                "seeds": seeds.tolist(),
                "sample_count": EXPECTED_SAMPLES,
                "samples_reused_identically_across_ridges": True,
            },
            "transfer_task_policy": {
                "name": TRANSFER_EVALUATION_TASKS.name,
                "seed": TRANSFER_EVALUATION_TASKS.seed,
                "count": TRANSFER_EVALUATION_TASKS.count,
            },
            "results": results,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=(0.0, 1.0e-4, 1.0e-2, 1.0e-1, 1.0),
    )
    args = parser.parse_args()
    print(run(args.run_dir, args.output, tuple(args.ridges)))


if __name__ == "__main__":
    main()
