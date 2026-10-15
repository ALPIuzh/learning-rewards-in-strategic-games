"""Explain a pilot's taskwise transfer ranking using performance-free geometry.

The analysis reconstructs the exact normalized reward suite named by a pilot
summary, computes source-relative optimal-policy TV and source-policy value gap,
and summarizes every requested protocol within tie-aware metric terciles.  It
never selects or drops a reward based on an algorithm's score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.task_sets import (
    StructuredTaskSet,
    normalized_structured_tasks,
)
from investigations.behavioral_transfer.pilot import (
    policy_total_variation,
    quantile_bucket_assignments,
    source_policy_gap,
)
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
)
from mairl_exp.artifacts import atomic_write_json

BUCKET_NAMES = ("low", "medium", "high")


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    if not np.isfinite(result.statistic):
        raise ValueError("task geometry produced an undefined rank correlation")
    return float(result.statistic)


def _task_geometry(
    source_run: Path,
    task_policy: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runtime = load_runtime(source_run)
    numerics = instantiate(OmegaConf.load(source_run / "resolved_config.yaml").numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    task_set = StructuredTaskSet(
        name=str(task_policy["name"]),
        count=int(task_policy["count"]),
        seed=int(task_policy["seed"]),
        candidate_pool=int(task_policy["candidate_pool"]),
    )
    rewards = normalized_structured_tasks(runtime, task_set)
    source_policy = np.asarray(runtime.fixed_mle_leader_policy, dtype=np.float64)
    tv = np.empty(task_set.count, dtype=np.float64)
    gap = np.empty(task_set.count, dtype=np.float64)
    maximum_probability = np.empty(task_set.count, dtype=np.float64)
    for task_index, reward in enumerate(rewards):
        solution = optimize_leader_policy(
            runtime.problem,
            runtime.true_omega,
            reward,
            leader_config,
            seed=task_set.seed + task_index,
        )
        source_value, _ = leader_value_and_gradient(
            runtime.problem,
            runtime.true_omega,
            reward,
            source_policy,
        )
        tv[task_index] = policy_total_variation(solution.policy, source_policy)
        gap[task_index] = source_policy_gap(
            oracle_value=solution.value,
            source_policy_value=source_value,
        )
        maximum_probability[task_index] = float(np.max(solution.policy))
    return tv, gap, maximum_probability


def _distribution(values: np.ndarray) -> dict[str, object]:
    quantiles = np.quantile(values, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    total = float(np.sum(values))
    ordered = np.sort(values)[::-1]
    worst_count = max(1, int(np.ceil(0.1 * values.size)))
    return {
        "mean": float(np.mean(values)),
        "quantiles": quantiles.tolist(),
        "largest_task_share_of_sum": (None if total == 0.0 else float(ordered[0] / total)),
        "worst_ten_percent_share_of_sum": (
            None if total == 0.0 else float(np.sum(ordered[:worst_count]) / total)
        ),
    }


def run(
    summary_path: Path,
    source_run: Path,
    output: Path,
    protocols: tuple[str, ...],
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing task analysis: {output}")
    summary = json.loads(summary_path.read_text())
    available = summary["protocols"]
    missing = sorted(set(protocols) - set(available))
    if missing:
        raise ValueError(f"protocols missing from pilot summary: {missing}")

    tv, gap, maximum_probability = _task_geometry(source_run, summary["task_policy"])
    geometry = {"tv": tv, "gap": gap}
    bucket_payload: dict[str, object] = {}
    for metric_name, metric_values in geometry.items():
        edges, assignments = quantile_bucket_assignments(metric_values)
        bucket_payload[metric_name] = {
            "edges": edges.tolist(),
            "counts": np.bincount(assignments, minlength=len(BUCKET_NAMES)).tolist(),
            "protocols": {
                protocol: {
                    bucket_name: _distribution(
                        np.asarray(
                            available[protocol]["taskwise_transfer_suboptimality_mean"],
                            dtype=np.float64,
                        )[assignments == bucket_index]
                    )
                    for bucket_index, bucket_name in enumerate(BUCKET_NAMES)
                }
                for protocol in protocols
            },
        }

    protocol_values = {
        protocol: np.asarray(
            available[protocol]["taskwise_transfer_suboptimality_mean"],
            dtype=np.float64,
        )
        for protocol in protocols
    }
    protocol_payload = {}
    for protocol, values in protocol_values.items():
        protocol_payload[protocol] = {
            "distribution": _distribution(values),
            "spearman_tv": _finite_spearman(tv, values),
            "spearman_gap": _finite_spearman(gap, values),
            "spearman_oracle_max_probability": _finite_spearman(
                maximum_probability, values
            ),
        }

    pairwise = {}
    for left_index, left in enumerate(protocols):
        for right in protocols[left_index + 1 :]:
            differences = protocol_values[left] - protocol_values[right]
            pairwise[f"{left}_minus_{right}"] = {
                "mean": float(np.mean(differences)),
                "median": float(np.median(differences)),
                "left_better_task_count": int(np.sum(differences < 0.0)),
                "right_better_task_count": int(np.sum(differences > 0.0)),
                "task_count": int(differences.size),
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output,
        {
            "schema_version": 1,
            "purpose": "performance-free geometric explanation of a fixed reward suite",
            "source_summary": str(summary_path),
            "source_run": str(source_run),
            "task_policy": summary["task_policy"],
            "geometry": {
                "tv": _distribution(tv),
                "gap": _distribution(gap),
                "oracle_max_probability": _distribution(maximum_probability),
                "spearman_tv_gap": _finite_spearman(tv, gap),
            },
            "buckets": bucket_payload,
            "protocols": protocol_payload,
            "pairwise": pairwise,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", action="append", required=True)
    args = parser.parse_args()
    print(run(args.summary, args.source_run, args.output, tuple(args.protocol)))


if __name__ == "__main__":
    main()
