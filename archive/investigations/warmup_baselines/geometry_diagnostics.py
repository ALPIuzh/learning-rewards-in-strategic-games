"""Information-geometry diagnostics for the protected Lobeke comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.downstream import load_runtime
from mairl.core.math import cumulative_fisher_information, fisher_information
from mairl.exploration.design import WarmupConfig, compute_warmup_design
from mairl.exploration.policy import select_information_policy
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.fixed_leader_ols import FixedLeaderOLSConfig, solve_fixed_leader_ols
from mairl.zero_sum.liao import solve_liao_zero_sum_ols


def condition(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values[-1] / values[0])


def job_entries(run_dir: Path, method: str, warmup_enabled: bool) -> list[dict[str, object]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return [
        entry
        for entry in manifest["expected_jobs"]
        if entry["spec"]["method"] == method
        and bool(entry["spec"]["warmup_enabled"]) == warmup_enabled
    ]


def information_prefixes(run_dir: Path, runtime: object, method: str, warm: bool) -> np.ndarray:
    rows = []
    for entry in job_entries(run_dir, method, warm):
        result_path = run_dir / "jobs" / str(entry["job_id"]) / "result.npz"
        with np.load(result_path, allow_pickle=False) as arrays:
            policies = arrays["policies"][:200]
        information = cumulative_fisher_information(runtime.problem, policies, runtime.true_omega)
        rows.append(np.linalg.eigvalsh(information))
    return np.asarray(rows)


def contrast_systems(runtime: object, config: FixedLeaderOLSConfig) -> None:
    problem = runtime.problem
    leader = runtime.expert.max_policy
    follower = runtime.expert.min_policy
    expected_features = np.einsum("a,abd->bd", leader, problem.features, optimize=True)

    follower_conditions = []
    for reference in range(problem.n_follower_actions):
        indices = np.delete(np.arange(problem.n_follower_actions), reference)
        matrix = expected_features[indices] - expected_features[reference]
        singular = np.linalg.svd(matrix, compute_uv=False)
        follower_conditions.append((reference, singular, float(np.linalg.cond(matrix))))
    follower_conditions.sort(key=lambda item: item[2])
    best_reference, best_singular, best_condition = follower_conditions[0]
    configured = next(
        item for item in follower_conditions if item[0] == config.follower_reference_action
    )
    print("\nFOLLOWER CONTRAST GEOMETRY")
    print(
        f"configured_reference={configured[0]} singular={configured[1].tolist()} "
        f"condition={configured[2]:.8g}"
    )
    print(
        f"best_reference={best_reference} singular={best_singular.tolist()} "
        f"condition={best_condition:.8g}"
    )

    follower_result = solve_fixed_leader_ols(
        problem,
        runtime.fixed_response,
        follower,
        config,
    )
    zero_sum_result = solve_liao_zero_sum_ols(
        runtime.game,
        runtime.expert,
        leader,
        follower,
        config,
    )
    print(
        f"exact follower-only omega={follower_result.omega.tolist()} "
        f"error={np.linalg.norm(follower_result.omega - runtime.true_omega):.3e} "
        f"singular={follower_result.singular_values.tolist()}"
    )
    print(
        f"exact zero-sum omega={zero_sum_result.omega.tolist()} "
        f"error={np.linalg.norm(zero_sum_result.omega - runtime.true_omega):.3e} "
        f"singular={zero_sum_result.singular_values.tolist()} "
        f"condition={np.linalg.cond(zero_sum_result.design_matrix):.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "artifacts/runs/"
            "guard-lobeke-paper/a4b1b2a1cdf886dd"
        ),
    )
    args = parser.parse_args()
    runtime = load_runtime(args.run_dir)
    problem = runtime.problem
    true_omega = runtime.true_omega
    expert_policy = runtime.expert.max_policy

    print("LOBeke GEOMETRY")
    print(f"shape={problem.features.shape} radius={problem.radius:g}")
    print(f"true_omega={true_omega.tolist()} norm={np.linalg.norm(true_omega):.8g}")
    expert_entropy = -np.sum(expert_policy * np.log(expert_policy))
    print(
        f"expert_leader_min={np.min(expert_policy):.8g} "
        f"max={np.max(expert_policy):.8g} entropy={expert_entropy:.8g}"
    )

    fixed_information = fisher_information(problem, expert_policy, true_omega)
    fixed_eigenvalues = np.linalg.eigvalsh(fixed_information)
    print("\nFIXED-EXPERT FISHER GEOMETRY")
    print(f"per_sample_eigenvalues={fixed_eigenvalues.tolist()}")
    print(f"condition={condition(fixed_eigenvalues):.8g}")
    print(
        f"cramer_rao_directional_sd_at_1600={(1.0 / np.sqrt(1600.0 * fixed_eigenvalues)).tolist()}"
    )
    ridge = 0.1
    shrinkage = 1600.0 * fixed_eigenvalues / (1600.0 * fixed_eigenvalues + ridge)
    print(f"local_ridge_0.1_shrinkage_at_1600={shrinkage.tolist()}")

    ols_config = FixedLeaderOLSConfig(
        pseudocount=0.5,
        smoothing_warning_ratio=0.1,
        identifiability="require_unique",
        constraint="ball",
        follower_reference_action=0,
        leader_reference_action=0,
        rank_relative_tolerance=1.0e-10,
        rank_absolute_tolerance=1.0e-12,
        ball_tolerance=1.0e-6,
        ball_root_tolerance=1.0e-12,
        ball_max_iterations=400,
    )
    contrast_systems(runtime, ols_config)

    design = compute_warmup_design(
        problem,
        WarmupConfig(budget=200, design_tolerance=0.15),
    )
    print("\nWARM-UP DESIGN")
    print(f"support={design.support_actions.tolist()}")
    print(f"weights={design.support_weights.tolist()}")
    print(f"counts={design.allocation_counts.tolist()}")
    print(f"coverage_eigenvalues={np.linalg.eigvalsh(design.coverage_matrix).tolist()}")

    initial = select_information_policy(
        problem,
        np.zeros(problem.feature_dimension),
        problem.eta**2 * np.eye(problem.feature_dimension),
        # Match the resolved final numerical configuration.
        PolicyOptimizationConfig(
            method="multistart",
            tolerance=1.0e-9,
            projected_gradient_tolerance=1.0e-5,
            max_iterations=1000,
            random_starts=16,
            batch_prefilter_iterations=4,
            batch_refine_starts=2,
            refine_grid_solution=True,
        ),
        np.random.default_rng(30000),
    )
    print("\nNO-WARM INITIAL ACTIVE POLICY")
    print(f"support={np.flatnonzero(initial.policy > 1e-8).tolist()}")
    print(f"weights={initial.policy[initial.policy > 1e-8].tolist()}")
    print(f"fisher_eigenvalues={np.linalg.eigvalsh(initial.fisher_information).tolist()}")
    print(f"information_gain={initial.information_gain:.8g}")

    print("\nTRUE-FISHER INFORMATION AFTER 200 OBSERVATIONS")
    for method, warm in (("active", True), ("active", False), ("fixed_random", True)):
        eigenvalues = information_prefixes(args.run_dir, runtime, method, warm)
        logdet = np.sum(np.log(np.maximum(eigenvalues, np.finfo(float).tiny)), axis=1)
        print(
            f"method={method} warm={warm} "
            f"mean_eigenvalues={np.mean(eigenvalues, axis=0).tolist()} "
            f"mean_logdet={np.mean(logdet):.8g} sd_logdet={np.std(logdet, ddof=1):.3g}"
        )


if __name__ == "__main__":
    main()
