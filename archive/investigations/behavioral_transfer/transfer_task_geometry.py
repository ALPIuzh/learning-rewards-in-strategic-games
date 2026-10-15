"""Relate downstream regret to task sensitivity and collection information.

This is a diagnostic, not a task-selection routine.  It retains every reward in
the supplied task set and evaluates pre-existing estimate/trajectory artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.downstream import load_runtime
from investigations.behavioral_transfer.small_k_geometry import finite_difference_hessian
from mairl.core.math import cumulative_fisher_information, fisher_information
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy_batch,
)
from mairl_exp.artifacts import atomic_write_json, collect_git_provenance, find_repository


def _regret_function(
    runtime: Any,
    leader_config: LeaderOptimizationConfig,
    leader_reward: np.ndarray,
    task_seed: int,
):
    true_solution = optimize_leader_policy_batch(
        runtime.problem,
        runtime.true_omega[None, :],
        leader_reward,
        leader_config,
        (task_seed,),
    )[0]

    def regret(error: np.ndarray) -> float:
        estimated_solution = optimize_leader_policy_batch(
            runtime.problem,
            (runtime.true_omega + error)[None, :],
            leader_reward,
            leader_config,
            (task_seed + 1,),
        )[0]
        deployed_value, _ = leader_value_and_gradient(
            runtime.problem,
            runtime.true_omega,
            leader_reward,
            estimated_solution.policy,
        )
        raw = float(true_solution.value - deployed_value)
        if raw < -1.0e-9:
            raise ValueError(f"task regret is unexpectedly negative: {raw:.3e}")
        return max(0.0, raw)

    return regret


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _load_entry(path: Path, runtime: Any) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        result = {
            "policies": arrays["policies"].copy(),
            "estimates": arrays["estimates"].copy(),
            "taskwise_suboptimality": arrays["taskwise_transfer_suboptimality"].copy(),
        }
    if result["taskwise_suboptimality"].ndim != 2:
        raise ValueError("taskwise transfer SubOpt must have shape (tasks, seeds)")
    result["information"] = np.stack(
        [
            cumulative_fisher_information(runtime.problem, policies, runtime.true_omega)
            for policies in result["policies"]
        ]
    )
    result["errors"] = result["estimates"] - runtime.true_omega[None, :]
    return result


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite task-geometry audit: {output}")

    run_dir = args.reference_run.resolve()
    runtime = load_runtime(run_dir)
    numerics = instantiate(OmegaConf.load(run_dir / "resolved_config.yaml").numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)

    tasks_dir = args.tasks_dir.resolve()
    with np.load(tasks_dir / "candidate_tasks.npz") as arrays:
        rewards = arrays["rewards"].copy()
    task_manifest = json.loads((tasks_dir / "task_manifest.json").read_text())
    task_seed = int(task_manifest["task_seed"])
    candidate_metrics = task_manifest.get("candidate_metrics", {})

    entries: dict[str, dict[str, np.ndarray]] = {}
    entry_paths: dict[str, str] = {}
    for specification in args.entry:
        if "=" not in specification:
            raise ValueError("entries must use LABEL=PATH syntax")
        label, raw_path = specification.split("=", 1)
        if not label or label in entries:
            raise ValueError("entry labels must be nonempty and unique")
        path = Path(raw_path).resolve()
        entries[label] = _load_entry(path, runtime)
        entry_paths[label] = str(path)
        if entries[label]["taskwise_suboptimality"].shape[0] != rewards.shape[0]:
            raise ValueError(f"{label} does not contain every supplied task")

    fixed_information = fisher_information(
        runtime.problem,
        runtime.fixed_mle_leader_policy,
        runtime.true_omega,
    )
    fixed_eigenvalues, fixed_eigenvectors = np.linalg.eigh(fixed_information)
    fixed_weak_direction = fixed_eigenvectors[:, 0]

    records: list[dict[str, Any]] = []
    for task_index, reward in enumerate(rewards):
        hessian = finite_difference_hessian(
            _regret_function(
                runtime,
                leader_config,
                reward,
                task_seed + 10_000 * task_index,
            ),
            runtime.problem.feature_dimension,
            args.step,
        )
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        sensitive_direction = eigenvectors[:, -1]
        record: dict[str, Any] = {
            "task_index": task_index,
            "hessian_eigenvalues": eigenvalues.tolist(),
            "sensitive_direction": sensitive_direction.tolist(),
            "absolute_fixed_weak_overlap": float(
                abs(sensitive_direction @ fixed_weak_direction)
            ),
        }
        for metric_name, values in candidate_metrics.items():
            record[metric_name] = float(values[task_index])
        for label, entry in entries.items():
            information = entry["information"]
            errors = entry["errors"]
            direction_information = np.einsum(
                "d,sde,e->s",
                sensitive_direction,
                information,
                sensitive_direction,
                optimize=True,
            )
            direction_errors = errors @ sensitive_direction
            record[label] = {
                "mean_suboptimality": float(
                    np.mean(entry["taskwise_suboptimality"][task_index])
                ),
                "mean_sensitive_information": float(np.mean(direction_information)),
                "sensitive_error_bias": float(np.mean(direction_errors)),
                "sensitive_error_rmse": float(
                    np.sqrt(np.mean(direction_errors**2))
                ),
            }
        records.append(record)

    pairwise: dict[str, Any] = {}
    labels = list(entries)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_subopt = np.asarray([row[left]["mean_suboptimality"] for row in records])
            right_subopt = np.asarray([row[right]["mean_suboptimality"] for row in records])
            left_information = np.asarray(
                [row[left]["mean_sensitive_information"] for row in records]
            )
            right_information = np.asarray(
                [row[right]["mean_sensitive_information"] for row in records]
            )
            difference = left_subopt - right_subopt
            log_information_ratio = np.log(
                np.maximum(left_information, 1.0e-300)
                / np.maximum(right_information, 1.0e-300)
            )
            overlaps = np.asarray([row["absolute_fixed_weak_overlap"] for row in records])
            pairwise[f"{left}_minus_{right}"] = {
                "mean_task_suboptimality_difference": float(np.mean(difference)),
                "left_better_task_count": int(np.sum(difference < 0.0)),
                "task_count": int(difference.size),
                "correlation_difference_with_fixed_weak_overlap": _correlation(
                    difference, overlaps
                ),
                "correlation_difference_with_log_information_ratio": _correlation(
                    difference, log_information_ratio
                ),
                "median_sensitive_information_ratio": float(
                    np.median(left_information / right_information)
                ),
            }

    payload = {
        "schema_version": 1,
        "reference_run": str(run_dir),
        "tasks_dir": str(tasks_dir),
        "task_seed": task_seed,
        "task_count": int(rewards.shape[0]),
        "finite_difference_step": args.step,
        "entry_paths": entry_paths,
        "fixed_expert_per_sample_fisher_eigenvalues": fixed_eigenvalues.tolist(),
        "fixed_expert_weak_direction": fixed_weak_direction.tolist(),
        "pairwise": pairwise,
        "tasks": records,
        "evaluation_git": collect_git_provenance(find_repository(Path.cwd())).manifest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=1.0e-2)
    parser.add_argument("--entry", action="append", required=True)
    args = parser.parse_args()
    if args.step <= 0.0:
        parser.error("step must be strictly positive")
    run(args)


if __name__ == "__main__":
    main()
