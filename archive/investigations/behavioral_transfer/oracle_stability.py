"""Check whether downstream optimal-policy TV is stable across optimizer seeds."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.downstream import load_runtime
from investigations.behavioral_transfer.pilot import METRICS, policy_total_variation
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.artifacts import (
    atomic_write_json,
    collect_git_provenance,
    find_repository,
)


def representative_task_indices(manifest: dict[str, object]) -> list[int]:
    """Select metric min/median/max within every bucket, independent of methods."""

    candidate_metrics = manifest["candidate_metrics"]
    buckets = manifest["buckets"]
    selected: set[int] = set()
    for metric in METRICS:
        values = np.asarray(candidate_metrics[metric], dtype=np.float64)
        assignments = np.asarray(buckets[metric]["assignments"], dtype=np.int64)
        for bucket in range(3):
            tasks = np.flatnonzero(assignments == bucket)
            order = tasks[np.argsort(values[tasks])]
            selected.update((int(order[0]), int(order[len(order) // 2]), int(order[-1])))
    return sorted(selected)


def _maximum_pairwise_tv(policies: list[np.ndarray]) -> float:
    return max(
        (policy_total_variation(left, right) for left, right in combinations(policies, 2)),
        default=0.0,
    )


def run(args: argparse.Namespace) -> None:
    output_path = args.output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite oracle-stability output: {output_path}")

    derived_dir = args.derived_dir.resolve()
    manifest = json.loads((derived_dir / "task_manifest.json").read_text())
    with np.load(derived_dir / "candidate_tasks.npz") as arrays:
        rewards = arrays["rewards"].copy()
        original_policies = arrays["oracle_policies"].copy()

    run_dir = args.run_dir.resolve()
    runtime = load_runtime(run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    task_indices = representative_task_indices(manifest)
    task_results = {}
    for task_index in task_indices:
        solutions = [
            optimize_leader_policy(
                runtime.problem,
                runtime.true_omega,
                rewards[task_index],
                leader_config,
                seed=args.seed + task_index * 100 + repeat,
            )
            for repeat in range(args.repeats)
        ]
        policies = [solution.policy for solution in solutions]
        values = np.asarray([solution.value for solution in solutions], dtype=np.float64)
        original = original_policies[task_index]
        task_results[str(task_index)] = {
            "tv": float(manifest["candidate_metrics"]["tv"][task_index]),
            "gap": float(manifest["candidate_metrics"]["gap"][task_index]),
            "value_range": [float(np.min(values)), float(np.max(values))],
            "value_spread": float(np.ptp(values)),
            "maximum_pairwise_policy_tv": _maximum_pairwise_tv(policies),
            "maximum_tv_from_original_policy": float(
                max(policy_total_variation(original, policy) for policy in policies)
            ),
            "support_sizes": [int(np.sum(policy > 1.0e-8)) for policy in policies],
            "projected_gradient_residual_maximum": float(
                max(solution.optimization.projected_gradient_residual for solution in solutions)
            ),
        }
        print(f"checked oracle stability for task {task_index}", flush=True)

    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "source_run": manifest["source_run"],
            "derived_task_seed": manifest["task_seed"],
            "evaluation_git": collect_git_provenance(find_repository(Path.cwd())).manifest(),
            "optimizer_repeats": args.repeats,
            "optimizer_seed": args.seed,
            "selection": "metric min/median/max within every TV and gap bucket",
            "task_count": len(task_indices),
            "tasks": task_results,
            "maximum_value_spread": max(item["value_spread"] for item in task_results.values()),
            "maximum_pairwise_policy_tv": max(
                item["maximum_pairwise_policy_tv"] for item in task_results.values()
            ),
            "maximum_tv_from_original_policy": max(
                item["maximum_tv_from_original_policy"] for item in task_results.values()
            ),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=82_000)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
