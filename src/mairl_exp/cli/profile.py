"""Repeatable component-level MAIRL profiler."""

from __future__ import annotations

import argparse
import cProfile
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from mairl.algorithm import ActiveIRL, ActiveIRLConfig
from mairl.core.math import cumulative_fisher_information, cumulative_fisher_information_batch
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import CompiledProblem, compile_problem
from mairl.exploration.design import WarmupConfig, compute_warmup_design
from mairl.exploration.policy import select_information_policy, select_information_policy_batch
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, solve_mle, solve_mle_batch
from mairl_exp.studies.trials import TuningTrial, run_tuning


def _scenario(seed: int):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(8, 7, 6))
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features /= np.maximum(1.0, 1.1 * norms)
    problem = compile_problem(features, eta=1.1, radius=2.0)
    true_omega = rng.normal(size=6)
    true_omega *= 0.9 / np.linalg.norm(true_omega)
    oracle = SoftmaxFollowerOracle(problem, true_omega)
    policies = rng.dirichlet(np.ones(problem.n_leader_actions), size=400)
    actions = np.array([oracle.sample(policy, rng) for policy in policies])
    dataset = InteractionDataset(policies, actions)
    return rng, problem, oracle, dataset


def _timed(function, repeats: int) -> tuple[float, Any]:
    durations = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        durations.append(perf_counter() - start)
    return median(durations), result


def _growing_mle_workload(problem: CompiledProblem, dataset: InteractionDataset, config: MLEConfig,
        *, first_size: int = 160, last_size: int = 220) -> dict[str, Any]:  # fmt: skip
    """Refit after each added observation, as in the online algorithm."""

    initial = None
    iterations = []
    residuals = []
    boundary_hits = 0
    result = None
    for size in range(first_size, last_size + 1):
        prefix = InteractionDataset(dataset.policies[:size], dataset.follower_actions[:size])
        result = solve_mle(problem, prefix, config, initial=initial)
        initial = result.omega
        iterations.append(result.iterations)
        residuals.append(result.projected_gradient_residual)
        boundary_hits += int(np.linalg.norm(result.omega) >= problem.radius - config.constraint_tolerance)

    return {
        "result": result,
        "fits": len(iterations),
        "median_iterations": float(np.median(iterations)),
        "max_iterations": int(np.max(iterations)),
        "max_residual": float(np.max(residuals)),
        "boundary_hits": boundary_hits,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--batch-seeds", type=int, default=16)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--cprofile", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    rng, problem, oracle, dataset = _scenario(args.seed)
    warmup_config = WarmupConfig(budget=160, design_tolerance=0.2)
    mle_config = MLEConfig(ridge=1e-4)
    slsqp_mle_config = replace(mle_config, solver="slsqp")
    policy_config = PolicyOptimizationConfig(random_starts=8)
    if args.batch_seeds < 1:
        raise ValueError("--batch-seeds must be positive")

    warmup_seconds, warmup = _timed(lambda: compute_warmup_design(problem, warmup_config), args.repeats)
    mle_seconds, mle = _timed(lambda: solve_mle(problem, dataset, mle_config), args.repeats)
    slsqp_mle_seconds, slsqp_mle = _timed(lambda: solve_mle(problem, dataset, slsqp_mle_config), args.repeats)
    growing_mle_seconds, growing_mle = _timed(lambda: _growing_mle_workload(problem, dataset, mle_config), args.repeats)
    growing_slsqp_seconds, growing_slsqp = _timed(
        lambda: _growing_mle_workload(problem, dataset, slsqp_mle_config), args.repeats
    )
    information = problem.eta**2 * np.eye(problem.feature_dimension) + cumulative_fisher_information(
        problem, dataset.policies, mle.omega
    )
    policy_seconds, selection = _timed(
        lambda: select_information_policy(problem, mle.omega, information, policy_config, rng), args.repeats
    )
    batch_policies = np.repeat(dataset.policies[None, :, :], args.batch_seeds, axis=0)
    batch_actions = np.repeat(dataset.follower_actions[None, :], args.batch_seeds, axis=0)
    scalar_seed_mle_seconds, scalar_seed_mles = _timed(
        lambda: tuple(
            solve_mle(problem, InteractionDataset(batch_policies[index], batch_actions[index]), mle_config)
            for index in range(args.batch_seeds)
        ),
        args.repeats,
    )
    batched_seed_mle_seconds, batched_seed_mles = _timed(
        lambda: solve_mle_batch(problem, batch_policies, batch_actions, mle_config), args.repeats
    )
    batch_omegas = np.stack([result.omega for result in batched_seed_mles])
    batch_information = problem.eta**2 * np.eye(problem.feature_dimension)[
        None, :, :
    ] + cumulative_fisher_information_batch(problem, batch_policies, batch_omegas)

    def scalar_seed_policies():
        generators = tuple(np.random.default_rng(args.seed + 10_000 + index) for index in range(args.batch_seeds))
        return tuple(
            select_information_policy(
                problem, batch_omegas[index], batch_information[index], policy_config, generators[index]
            )
            for index in range(args.batch_seeds)
        )

    def batched_seed_policies():
        generators = tuple(np.random.default_rng(args.seed + 10_000 + index) for index in range(args.batch_seeds))
        return select_information_policy_batch(problem, batch_omegas, batch_information, policy_config, generators)

    scalar_seed_policy_seconds, scalar_seed_policies_result = _timed(scalar_seed_policies, args.repeats)
    batched_seed_policy_seconds, batched_seed_policies_result = _timed(batched_seed_policies, args.repeats)
    grid_problem = compile_problem(problem.features[:3], eta=problem.eta, radius=problem.radius)
    grid_config = PolicyOptimizationConfig(
        method="grid", grid_resolution=100, grid_max_points=10_000, refine_grid_solution=False
    )
    grid_information = problem.eta**2 * np.eye(problem.feature_dimension)
    start = perf_counter()
    grid_selection = select_information_policy(grid_problem, mle.omega, grid_information, grid_config, rng)
    grid_cold_seconds = perf_counter() - start
    grid_cached_seconds, _ = _timed(
        lambda: select_information_policy(grid_problem, mle.omega, grid_information, grid_config, rng), args.repeats
    )
    active_config = ActiveIRLConfig(
        horizon=12,
        lambda_information=problem.eta**2,
        lambda_warm=0.01,
        confidence_delta=0.05,
        mle=mle_config,
        warmup=warmup_config,
        policy=policy_config,
        seed=args.seed,
    )
    runner = ActiveIRL(problem, active_config)
    end_to_end_seconds, active_result = _timed(lambda: runner.run(oracle), args.repeats)
    tuning_trials = tuple(
        TuningTrial(f"ridge-{ridge:.0e}", replace(active_config, horizon=4, mle=replace(mle_config, ridge=ridge)))
        for ridge in (1e-5, 3e-5, 1e-4, 3e-4)
    )
    tuning_seconds, tuning_result = _timed(
        lambda: run_tuning(
            problem,
            tuning_trials,
            oracle_factory=lambda _: SoftmaxFollowerOracle(problem, oracle.true_omega),
            evaluate=lambda _, result: {"omega_norm": np.linalg.norm(result.omega)},
        ),
        args.repeats,
    )

    report = {
        "scenario": {
            "leader_actions": problem.n_leader_actions,
            "follower_actions": problem.n_follower_actions,
            "feature_dimension": problem.feature_dimension,
            "identifiable_dimension": problem.identifiable_dimension,
            "dataset_size": dataset.size,
            "batch_seeds": args.batch_seeds,
            "repeats": args.repeats,
        },
        "median_seconds": {
            "warmup_design": warmup_seconds,
            "mle_400_newton": mle_seconds,
            "mle_400_slsqp": slsqp_mle_seconds,
            "growing_mle_61_newton": growing_mle_seconds,
            "growing_mle_61_slsqp": growing_slsqp_seconds,
            "policy_selection": policy_seconds,
            "seed_mle_scalar": scalar_seed_mle_seconds,
            "seed_mle_vectorized": batched_seed_mle_seconds,
            "seed_policy_scalar": scalar_seed_policy_seconds,
            "seed_policy_vectorized": batched_seed_policy_seconds,
            "grid_policy_5151_cold": grid_cold_seconds,
            "grid_policy_5151_cached": grid_cached_seconds,
            "end_to_end_160_plus_12": end_to_end_seconds,
            "tuning_4_trials": tuning_seconds,
        },
        "checks": {
            "warmup_iterations": warmup.diagnostics.iterations,
            "warmup_gap": warmup.diagnostics.final_gap,
            "mle_residual": mle.projected_gradient_residual,
            "mle_solver_difference": np.linalg.norm(mle.omega - slsqp_mle.omega),
            "mle_newton_iterations": mle.iterations,
            "mle_slsqp_iterations": slsqp_mle.iterations,
            "growing_mle_fits": growing_mle["fits"],
            "growing_newton_median_iterations": growing_mle["median_iterations"],
            "growing_slsqp_median_iterations": growing_slsqp["median_iterations"],
            "growing_newton_max_residual": growing_mle["max_residual"],
            "growing_slsqp_max_residual": growing_slsqp["max_residual"],
            "growing_newton_boundary_hits": growing_mle["boundary_hits"],
            "growing_slsqp_boundary_hits": growing_slsqp["boundary_hits"],
            "growing_newton_milliseconds_per_fit": (1_000.0 * growing_mle_seconds / growing_mle["fits"]),
            "growing_slsqp_milliseconds_per_fit": (1_000.0 * growing_slsqp_seconds / growing_slsqp["fits"]),
            "growing_mle_speedup": growing_slsqp_seconds / growing_mle_seconds,
            "policy_residual": selection.optimization.projected_gradient_residual,
            "seed_mle_speedup": scalar_seed_mle_seconds / batched_seed_mle_seconds,
            "seed_mle_max_difference": max(
                np.linalg.norm(left.omega - right.omega)
                for left, right in zip(scalar_seed_mles, batched_seed_mles, strict=True)
            ),
            "seed_policy_speedup": (scalar_seed_policy_seconds / batched_seed_policy_seconds),
            "seed_policy_max_objective_difference": max(
                abs(left.information_gain - right.information_gain)
                for left, right in zip(scalar_seed_policies_result, batched_seed_policies_result, strict=True)
            ),
            "grid_points": grid_selection.optimization.grid_points,
            "final_samples": active_result.dataset.size,
            "tuning_warmup_designs": tuning_result.unique_warmup_designs,
        },
    }

    print("MAIRL profile (median wall-clock seconds; no concurrency)")
    for name, duration in report["median_seconds"].items():
        print(f"  {name:28s} {duration:10.6f}")
    print("Post-solve checks")
    for name, value in report["checks"].items():
        print(f"  {name:28s} {value}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"json_report={args.json.resolve()}")

    if args.cprofile is not None:
        args.cprofile.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        runner.run(oracle)
        profiler.disable()
        profiler.dump_stats(args.cprofile)
        print(f"cprofile={args.cprofile.resolve()}")


if __name__ == "__main__":
    main()
