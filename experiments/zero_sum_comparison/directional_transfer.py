"""Build a target-dependent Fisher-transfer diagnostic from frozen estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.linalg import eigh, eigvalsh

from experiments.zero_sum_comparison.application import ZeroSumApplicationRuntime
from experiments.zero_sum_comparison.downstream_tasks import _evaluate_task, _final_estimates, _load_run_traces
from mairl.core.math import fisher_information
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, canonical_json, sha256_bytes

DEFAULT_COMPARISON = REPOSITORY_ROOT / "artifacts" / "runs" / "zero-sum-estimator-comparison" / "74204b3d1922f3ce"
DEFAULT_OUTPUT_STEM = REPOSITORY_ROOT / "reproducibility" / "data" / "zero_sum" / "directional_transfer"
METHODS = ("active", "zero_sum_ols", "follower_ols")


@dataclass(frozen=True, slots=True)
class DirectionalTaskSelection:
    """Two tasks selected exclusively from their ground-truth Fisher geometry."""

    tasks: np.ndarray
    policies: np.ndarray
    fisher_matrices: np.ndarray
    candidate_indices: tuple[int, int]
    policy_l1_distance: float
    generalized_contrast_ratio: float
    controlled_error_direction: np.ndarray


def select_directional_tasks(runtime: ZeroSumApplicationRuntime, *, seed: int = 20_260_810,
        candidate_pool: int = 256, minimum_policy_l1_distance: float = 0.75,
        fisher_ridge: float = 1.0e-6) -> DirectionalTaskSelection:  # fmt: skip
    """Select a pair without consulting any learned estimate or method outcome."""

    if candidate_pool < 2:
        raise ConfigurationError("directional task candidate_pool must be at least two")
    if not 0 < minimum_policy_l1_distance <= 2:
        raise ConfigurationError("minimum policy distance must lie in (0, 2]")
    if fisher_ridge <= 0:
        raise ConfigurationError("directional Fisher ridge must be strictly positive")
    rng = np.random.default_rng(seed)
    shape = (runtime.problem.n_leader_actions, runtime.problem.n_follower_actions)
    candidates = rng.uniform(-1.0, 1.0, size=(candidate_pool, *shape))
    policies = np.empty((candidate_pool, runtime.problem.n_leader_actions), dtype=np.float64)
    matrices = np.empty(
        (candidate_pool, runtime.problem.feature_dimension, runtime.problem.feature_dimension), dtype=np.float64
    )
    leader_config = LeaderOptimizationConfig()
    for index, reward in enumerate(candidates):
        policy = optimize_leader_policy(
            runtime.problem, runtime.true_omega, reward, leader_config, seed=seed + index
        ).policy
        policies[index] = policy
        matrices[index] = fisher_information(runtime.problem, policy, runtime.true_omega)

    identity = np.eye(runtime.problem.feature_dimension)
    best: tuple[float, int, int, float] | None = None
    for first, second in combinations(range(candidate_pool), 2):
        distance = float(np.linalg.norm(policies[first] - policies[second], ord=1))
        if distance < minimum_policy_l1_distance:
            continue
        eigenvalues = eigvalsh(matrices[first] + fisher_ridge * identity, matrices[second] + fisher_ridge * identity)
        if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError("regularized Fisher pencil is not positive definite")
        contrast = float(max(np.max(eigenvalues), 1.0 / np.min(eigenvalues)))
        candidate = contrast, -first, -second, distance
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ConfigurationError("no sufficiently distinct directional task pair was found")
    contrast, negative_first, negative_second, distance = best
    indices = -negative_first, -negative_second
    selected_matrices = matrices[list(indices)]
    eigenvalues, eigenvectors = eigh(
        selected_matrices[0] + fisher_ridge * identity, selected_matrices[1] + fisher_ridge * identity
    )
    del eigenvalues
    # Once the unordered pair is selected by its symmetric contrast, its stable
    # candidate-index order defines Task 1 versus Task 2. The maximum generalized
    # eigenvector is therefore the direction emphasized by Task 1 relative to Task 2.
    direction = eigenvectors[:, -1]
    direction /= np.linalg.norm(direction)
    # Fix the otherwise arbitrary eigenvector sign by pointing the perturbation inward,
    # which guarantees the controlled estimate stays away from the ball boundary.
    if float(runtime.true_omega @ direction) > 0:
        direction = -direction
    return DirectionalTaskSelection(
        tasks=candidates[list(indices)],
        policies=policies[list(indices)],
        fisher_matrices=selected_matrices,
        candidate_indices=indices,
        policy_l1_distance=distance,
        generalized_contrast_ratio=contrast,
        controlled_error_direction=direction,
    )


def export(comparison_dir: Path, output_stem: Path) -> tuple[Path, Path]:
    """Evaluate the frozen estimates on the geometry-selected target pair."""

    manifest_path = comparison_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    progress = manifest.get("progress", {})
    required = {"complete": 300, "failed": 0, "pending": 0, "running": 0, "total": 300}
    if manifest.get("status") != "complete" or any(progress.get(key) != value for key, value in required.items()):
        raise ConfigurationError("zero-sum comparison run is incomplete")
    traces, raw_inputs = _load_run_traces(comparison_dir)
    for method in METHODS:
        if [item[0] for item in traces[method]] != list(range(60_000, 60_100)):
            raise ConfigurationError(f"{method} does not use the protected final seeds")
    estimates = _final_estimates(traces)

    config = OmegaConf.load(comparison_dir / "resolved_config.yaml")
    runtime = instantiate(config.application).build(instantiate(config.numerics))
    if not isinstance(runtime, ZeroSumApplicationRuntime):
        raise ConfigurationError("directional export requires the zero-sum runtime")
    selection = select_directional_tasks(runtime)
    if selection.generalized_contrast_ratio < 2.0:
        raise ConfigurationError("selected Fisher geometry does not provide a material contrast")

    coefficient_error = np.empty((len(METHODS), 100), dtype=np.float64)
    for method_index, method in enumerate(METHODS):
        coefficient_error[method_index] = [
            np.linalg.norm(estimate - runtime.true_omega) for _seed, estimate in estimates[method]
        ]
    task_results = [
        _evaluate_task(runtime.problem, runtime.true_omega, task, LeaderOptimizationConfig(), estimates)
        for task in selection.tasks
    ]
    controlled_error_norm = 0.25
    controlled_estimate = runtime.true_omega + controlled_error_norm * selection.controlled_error_direction
    if np.linalg.norm(controlled_estimate) >= runtime.problem.radius:
        raise ConfigurationError("controlled directional estimate unexpectedly leaves the ball")
    controlled_results = [
        _evaluate_task(
            runtime.problem,
            runtime.true_omega,
            task,
            LeaderOptimizationConfig(),
            {"controlled": [(0, controlled_estimate)]},
        )["controlled"]
        for task in selection.tasks
    ]
    fisher_error = np.empty((2, len(METHODS), 100), dtype=np.float64)
    suboptimality = np.empty_like(fisher_error)
    for task_index, result in enumerate(task_results):
        for method_index, method in enumerate(METHODS):
            fisher_error[task_index, method_index] = result[method]["fisher_transfer_error"]
            suboptimality[task_index, method_index] = result[method]["stackelberg_suboptimality"]

    arrays = {
        "methods": np.asarray(METHODS),
        "seeds": np.arange(60_000, 60_100),
        "tasks": selection.tasks,
        "oracle_policies": selection.policies,
        "oracle_fisher_matrices": selection.fisher_matrices,
        "controlled_error_direction": selection.controlled_error_direction,
        "controlled_estimate": controlled_estimate,
        "controlled_fisher_transfer_error": np.asarray(
            [result["fisher_transfer_error"][0] for result in controlled_results]
        ),
        "controlled_stackelberg_suboptimality": np.asarray(
            [result["stackelberg_suboptimality"][0] for result in controlled_results]
        ),
        "coefficient_l2_error": coefficient_error,
        "fisher_transfer_error": fisher_error,
        "stackelberg_suboptimality": suboptimality,
    }
    npz_path, json_path = output_stem.with_suffix(".npz"), output_stem.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(npz_path, arrays)
    sorted_inputs = sorted(raw_inputs, key=lambda item: item["job_id"])
    atomic_write_json(
        json_path,
        {
            "schema_version": 1,
            "experiment": "zero-sum target-dependent Fisher-transfer diagnostic",
            "source_run_id": manifest["run_id"],
            "source_commit": manifest["git"]["commit"],
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_inputs_sha256": sha256_bytes(canonical_json(sorted_inputs).encode()),
            "derived_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "methods": list(METHODS),
            "seed_policy": {"paired": True, "seeds": list(range(60_000, 60_100)), "count": 100},
            "selection": {
                "uses_learned_estimates": False,
                "uses_method_outcomes": False,
                "candidate_seed": 20_260_810,
                "candidate_pool": 256,
                "candidate_indices": list(selection.candidate_indices),
                "minimum_policy_l1_distance": 0.75,
                "observed_policy_l1_distance": selection.policy_l1_distance,
                "fisher_ridge": 1.0e-6,
                "generalized_contrast_ratio": selection.generalized_contrast_ratio,
                "controlled_error_l2_norm": controlled_error_norm,
                "controlled_direction_rule": (
                    "maximum generalized eigenvector of Task 1 relative to Task 2, "
                    "with sign pointing inward relative to true omega"
                ),
                "criterion": (
                    "maximize the symmetric regularized generalized-eigenvalue contrast "
                    "between oracle-policy Fisher matrices"
                ),
            },
        },
    )
    return json_path, npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()
    json_path, npz_path = export(args.comparison_dir, args.output_stem)
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
