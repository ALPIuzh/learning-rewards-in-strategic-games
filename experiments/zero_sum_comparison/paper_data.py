"""Export the final zero-sum study as compact, tracked publication data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.zero_sum_comparison.downstream_tasks import (
    _draw_diverse_tasks,
    _evaluate_task,
    _final_estimates,
    _load_run_traces,
    _paired_summary,
)
from experiments.zero_sum_comparison.export_plots import _load_comparison
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.config import REPOSITORY_ROOT, RUNS_ROOT
from mairl_exp.execution.artifacts import atomic_write_json, canonical_json, sha256_bytes, sha256_file

DEFAULT_COMPARISON = RUNS_ROOT / "zero-sum-estimator-comparison" / "74204b3d1922f3ce"
DEFAULT_COARSE = RUNS_ROOT / "zero-sum-joint-selection" / "58d16f3d3f95c681"
DEFAULT_REFINEMENT = RUNS_ROOT / "zero-sum-joint-refinement" / "0912bb3d4fef6239"
DEFAULT_CONFIRMATION = RUNS_ROOT / "zero-sum-joint-confirmation" / "b64dbc092dba08fd"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "data" / "zero_sum" / "comparison.json"
METHODS = ("active", "zero_sum_ols", "follower_ols")
SOURCE_METRICS = ("fixed_leader_fisher_error", "regularized_qre_leader_l1_error")
DOWNSTREAM_METRICS = ("stackelberg_suboptimality", "fisher_transfer_error")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ConfigurationError(f"expected a JSON object in {path}")
    return payload


def _statistics(values: np.ndarray) -> dict[str, object]:
    if values.ndim not in (1, 2) or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ConfigurationError("publication summaries require at least two finite seeds")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "standard_error": (np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])).tolist(),
        "seeds": int(values.shape[0]),
    }


def _run_source(run_dir: Path, result_file: str | None = None) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    source = {
        "study_name": manifest["study_name"],
        "run_id": manifest["run_id"],
        "manifest_status": manifest["status"],
        "manifest_sha256": sha256_file(manifest_path),
        "semantic_config_sha256": manifest["semantic_config_sha256"],
        "git_commit": manifest["git"]["commit"],
    }
    if result_file is not None:
        result_path = run_dir / result_file
        source.update({"result_file": result_file, "result_sha256": sha256_file(result_path)})
    return source


def _validate_tuning_sources(coarse_dir: Path, refinement_dir: Path, confirmation_dir: Path) -> dict[str, object]:
    coarse_manifest = _load_json(coarse_dir / "manifest.json")
    refinement_manifest = _load_json(refinement_dir / "manifest.json")
    confirmation_manifest = _load_json(confirmation_dir / "manifest.json")
    _load_json(coarse_dir / "joint_discovery.json")
    _load_json(refinement_dir / "joint_discovery.json")
    confirmation = _load_json(confirmation_dir / "confirmation.json")
    if coarse_manifest["status"] != "complete" or refinement_manifest["status"] != "complete":
        raise ConfigurationError("zero-sum selection and refinement must be complete")
    if confirmation["selection_run_id"] != refinement_manifest["run_id"]:
        raise ConfigurationError("confirmation does not reference the supplied refinement")
    if confirmation["status"] != "rejected":
        raise ConfigurationError("the recorded frozen zero-sum candidate was not rejected")
    progress = confirmation_manifest["progress"]
    if progress["complete"] != progress["total"] or progress["failed"] != 0:
        raise ConfigurationError("confirmation jobs did not all complete successfully")
    return {
        "coarse": _run_source(coarse_dir, "joint_discovery.json"),
        "refinement": _run_source(refinement_dir, "joint_discovery.json"),
        "confirmation": _run_source(confirmation_dir, "confirmation.json"),
        "frozen_candidate": confirmation["candidate"],
        "validated_final": confirmation["reference"],
        "decision": "frozen candidate rejected; use predeclared reference",
        "checks_passed": confirmation["checks_passed"],
        "paired_differences_candidate_minus_reference": confirmation["paired_differences_candidate_minus_reference"],
        "selection_and_confirmation_seeds_disjoint": confirmation["selection_and_confirmation_seeds_disjoint"],
    }


def export(*, comparison_dir: Path, coarse_dir: Path, refinement_dir: Path, confirmation_dir: Path,
        output: Path, num_tasks: int = 3) -> None:  # fmt: skip
    comparison_traces, comparison_inputs, comparison_manifest = _load_comparison(comparison_dir)
    reward_traces, reward_inputs = _load_run_traces(comparison_dir)
    if sorted(comparison_inputs, key=lambda item: item["job_id"]) != sorted(
        reward_inputs, key=lambda item: item["job_id"]
    ):
        raise ConfigurationError("source and downstream loaders disagree on comparison inputs")
    if comparison_manifest["status"] != "complete":
        raise ConfigurationError("final comparison run is not complete")

    source: dict[str, object] = {}
    for metric in SOURCE_METRICS:
        source[metric] = {}
        for method in METHODS:
            traces = comparison_traces[method]
            checkpoints = traces[0][1]
            if not all(np.array_equal(checkpoints, trace[1]) for trace in traces):
                raise ConfigurationError(f"{method} uses inconsistent source checkpoints")
            values = np.stack([trace[2][metric] for trace in traces])
            source[metric][method] = {"checkpoints": checkpoints.tolist(), **_statistics(values)}

    config = OmegaConf.load(comparison_dir / "resolved_config.yaml")
    runtime = instantiate(config.application).build(instantiate(config.numerics))
    tasks, pairwise_cosine = _draw_diverse_tasks(
        shape=(runtime.problem.n_leader_actions, runtime.problem.n_follower_actions),
        num_tasks=num_tasks,
        seed=20260731,
        scale=1.0,
        candidate_pool=200,
    )
    estimates = _final_estimates(reward_traces)
    task_results = [
        _evaluate_task(runtime.problem, runtime.true_omega, task, LeaderOptimizationConfig(), estimates)
        for task in tasks
    ]
    downstream: dict[str, object] = {}
    for metric in DOWNSTREAM_METRICS:
        downstream[metric] = {}
        for method in METHODS:
            values = np.stack([result[method][metric] for result in task_results], axis=1)
            downstream[metric][method] = _statistics(values)

    sorted_inputs = sorted(comparison_inputs, key=lambda item: item["job_id"])
    payload = {
        "schema_version": 1,
        "sources": {
            "comparison": _run_source(comparison_dir),
            "comparison_inputs_sha256": sha256_bytes(canonical_json(sorted_inputs).encode()),
            "tuning": _validate_tuning_sources(coarse_dir, refinement_dir, confirmation_dir),
        },
        "methods": {"active": "Active IRL", "zero_sum_ols": "Zero-sum OLS", "follower_ols": "Follower-only OLS"},
        "source": source,
        "downstream": {
            "task_seed": 20260731,
            "task_scale": 1.0,
            "task_candidate_pool": 200,
            "tasks": tasks.tolist(),
            "pairwise_cosine_similarity": pairwise_cosine.tolist(),
            "metrics": downstream,
            "paired": _paired_summary(task_results, 0.95),
        },
        "confidence_level": 0.95,
        "seed_policy": {
            "selection": "40000..40063",
            "confirmation": "50000..50063",
            "final_evaluation": "60000..60099",
            "all_splits_disjoint": True,
        },
    }
    atomic_write_json(output, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--coarse-dir", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--refinement-dir", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--confirmation-dir", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-tasks", type=int, default=3, help="Number of deterministic diverse downstream tasks to export."
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    export(
        comparison_dir=args.comparison_dir,
        coarse_dir=args.coarse_dir,
        refinement_dir=args.refinement_dir,
        confirmation_dir=args.confirmation_dir,
        output=args.output,
        num_tasks=args.num_tasks,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
