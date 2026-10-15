"""Internal vectorized execution across independent experiment seeds."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

from mairl.core.math import cumulative_fisher_information_batch
from mairl.core.oracle import sample_follower_action
from mairl.core.problem import FloatArray
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.exploration.policy import select_information_policy_batch
from mairl.exploration.warmup import PreparedWarmup
from mairl.inference.mle import solve_mle_batch
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.jobs import StudyJobResult, StudyMethod

ProgressCallback = Callable[[int, int, int], None]


def run_seed_batch(*, method: StudyMethod, warmup_enabled: bool, seeds: tuple[int, ...], total_samples: int,
        evaluation_interval: int, lambda_information_multiplier: float,
        application: ApplicationRuntime, numerics: ExperimentNumerics, prepared_warmup: PreparedWarmup,
        on_progress: ProgressCallback | None = None) -> tuple[StudyJobResult, ...]:  # fmt: skip
    """Run one method and warm-up mode over a leading seed dimension."""

    seed_count = len(seeds)
    problem = application.problem
    warmup_samples = prepared_warmup.config.effective_budget
    online_samples = total_samples - warmup_samples
    oracles = tuple(application.oracle_factory(seed) for seed in seeds)
    policies = np.empty((seed_count, total_samples, problem.n_leader_actions), dtype=np.float64)
    actions = np.empty((seed_count, total_samples), dtype=np.int64)

    warmup_policies = _warmup_policies(problem.n_leader_actions, prepared_warmup)
    if warmup_samples:
        policies[:, :warmup_samples] = warmup_policies[None, :, :]
        for seed_index in range(seed_count):
            warmup_rng = experiment_rng(seeds[seed_index], RandomPurpose.WARMUP_FOLLOWER)
            for sample_index in range(warmup_samples):
                actions[seed_index, sample_index] = sample_follower_action(
                    oracles[seed_index], problem, warmup_policies[sample_index], warmup_rng
                )

    started = perf_counter()
    mle_results = solve_mle_batch(problem, policies[:, :warmup_samples], actions[:, :warmup_samples], numerics.mle)
    omegas = np.stack([result.omega for result in mle_results])
    fixed_policies: FloatArray | None = None
    if method == "fixed_random":
        fixed_policies = np.stack(
            [
                experiment_rng(seed, RandomPurpose.FIXED_RANDOM_POLICY).dirichlet(np.ones(problem.n_leader_actions))
                for seed in seeds
            ]
        )

    if method == "active":
        trace_sample_counts = np.arange(warmup_samples, total_samples + 1, dtype=np.int64)
        omega_history = np.empty((seed_count, online_samples + 1, problem.feature_dimension), dtype=np.float64)
        omega_history[:, 0] = omegas
        mle_iterations = np.empty((seed_count, online_samples), dtype=np.int64)
        mle_residuals = np.empty((seed_count, online_samples), dtype=np.float64)
        policy_residuals = np.empty((seed_count, online_samples), dtype=np.float64)
        information_gains = np.empty((seed_count, online_samples), dtype=np.float64)

        # lambda_warm remains part of WARM configuration and run provenance, but it
        # does not regularize online policy selection.
        regularizer = lambda_information_multiplier * problem.eta**2
        identity = np.eye(problem.feature_dimension)[None, :, :]
        for online_index in range(online_samples):
            sample_index = warmup_samples + online_index
            information = regularizer * identity + cumulative_fisher_information_batch(
                problem, policies[:, :sample_index], omegas
            )
            selections = select_information_policy_batch(
                problem,
                omegas,
                information,
                numerics.active_policy,
                tuple(experiment_rng(seed, RandomPurpose.ACTIVE_POLICY, sample_index) for seed in seeds),
            )
            selected_policies = np.stack([selection.policy for selection in selections])
            information_gains[:, online_index] = [selection.information_gain for selection in selections]
            policy_residuals[:, online_index] = [
                selection.optimization.projected_gradient_residual for selection in selections
            ]
            policies[:, sample_index] = selected_policies
            for seed_index in range(seed_count):
                actions[seed_index, sample_index] = sample_follower_action(
                    oracles[seed_index],
                    problem,
                    selected_policies[seed_index],
                    experiment_rng(seeds[seed_index], RandomPurpose.ONLINE_FOLLOWER, sample_index),
                )
            mle_results = solve_mle_batch(
                problem, policies[:, : sample_index + 1], actions[:, : sample_index + 1], numerics.mle, initial=omegas
            )
            omegas = np.stack([result.omega for result in mle_results])
            omega_history[:, online_index + 1] = omegas
            mle_iterations[:, online_index] = [result.iterations for result in mle_results]
            mle_residuals[:, online_index] = [result.projected_gradient_residual for result in mle_results]
            completed_samples = sample_index + 1
            if on_progress is not None and (
                completed_samples == total_samples or completed_samples % evaluation_interval == 0
            ):
                for seed_index in range(seed_count):
                    on_progress(seed_index, completed_samples, total_samples)
    else:
        # Random policies do not depend on the current reward estimate. Collect the
        # complete trajectory first, then fit only the estimates that are evaluated.
        # This is exactly the same estimator at every reported checkpoint without the
        # quadratic cost of solving unused intermediate MLEs.
        for online_index in range(online_samples):
            sample_index = warmup_samples + online_index
            if fixed_policies is not None:
                selected_policies = fixed_policies
            else:
                selected_policies = np.stack(
                    [
                        experiment_rng(seed, RandomPurpose.ONLINE_RANDOM_POLICY, sample_index).dirichlet(
                            np.ones(problem.n_leader_actions)
                        )
                        for seed in seeds
                    ]
                )
            policies[:, sample_index] = selected_policies
            for seed_index in range(seed_count):
                actions[seed_index, sample_index] = sample_follower_action(
                    oracles[seed_index],
                    problem,
                    selected_policies[seed_index],
                    experiment_rng(seeds[seed_index], RandomPurpose.ONLINE_FOLLOWER, sample_index),
                )

        trace_sample_counts = evaluation_checkpoints(warmup_samples, total_samples, evaluation_interval)
        omega_history = np.empty((seed_count, trace_sample_counts.size, problem.feature_dimension), dtype=np.float64)
        omega_history[:, 0] = omegas
        for checkpoint_index, checkpoint in enumerate(trace_sample_counts[1:], start=1):
            mle_results = solve_mle_batch(
                problem, policies[:, :checkpoint], actions[:, :checkpoint], numerics.mle, initial=omegas
            )
            omegas = np.stack([result.omega for result in mle_results])
            omega_history[:, checkpoint_index] = omegas
            if on_progress is not None:
                for seed_index in range(seed_count):
                    on_progress(seed_index, int(checkpoint), total_samples)
        mle_iterations = np.empty((seed_count, 0), dtype=np.int64)
        mle_residuals = np.empty((seed_count, 0), dtype=np.float64)
        policy_residuals = np.empty((seed_count, 0), dtype=np.float64)
        information_gains = np.empty((seed_count, 0), dtype=np.float64)

    elapsed = perf_counter() - started
    checkpoints = evaluation_checkpoints(warmup_samples, total_samples, evaluation_interval)
    checkpoint_indices = checkpoint_positions(trace_sample_counts, checkpoints)
    design = prepared_warmup.design
    coverage_eigenvalues = np.linalg.eigvalsh(design.coverage_matrix)
    amortized_seconds = elapsed / len(seeds)
    outputs: list[StudyJobResult] = []
    for seed_index, seed in enumerate(seeds):
        estimates = omega_history[seed_index, checkpoint_indices]
        metrics = application.evaluate(estimates, seed)
        if method == "active":
            diagnostics = {
                "diagnostic__information_gain": information_gains[seed_index],
                "diagnostic__mle_iterations": mle_iterations[seed_index],
                "diagnostic__mle_residual": mle_residuals[seed_index],
                "diagnostic__policy_residual": policy_residuals[seed_index],
            }
        else:
            diagnostics = {
                "diagnostic__information_gain": np.empty(0, dtype=np.float64),
                "diagnostic__mle_residual": np.asarray([mle_results[seed_index].projected_gradient_residual]),
                "diagnostic__policy_residual": np.empty(0, dtype=np.float64),
            }
        arrays: dict[str, np.ndarray] = {
            "trace_sample_counts": trace_sample_counts,
            "omega_history": omega_history[seed_index],
            "evaluation_sample_counts": checkpoints,
            "policies": policies[seed_index],
            "follower_actions": actions[seed_index],
            "warmup__design_weights": design.weights,
            "warmup__support_actions": design.support_actions,
            "warmup__allocation_counts": design.allocation_counts,
            "warmup__coverage_eigenvalues": coverage_eigenvalues,
            **diagnostics,
        }
        arrays.update({f"metric__{name}": values for name, values in metrics.items()})
        outputs.append(
            StudyJobResult(
                arrays=arrays,
                summary={
                    "job": {"method": method, "warmup_enabled": warmup_enabled, "seed": seed},
                    "algorithm_seconds": amortized_seconds,
                    "batch_algorithm_seconds": elapsed,
                    "batch_seed_count": len(seeds),
                    "samples": total_samples,
                    "final_metrics": {name: float(values[-1]) for name, values in metrics.items()},
                    "warmup": {
                        "enabled": warmup_enabled,
                        "samples": warmup_samples,
                        "support_size": design.support_size,
                        "design_weights": design.weights.tolist(),
                        "allocation_counts": design.allocation_counts.tolist(),
                        "coverage_eigenvalues": coverage_eigenvalues.tolist(),
                    },
                },
            )
        )
    return tuple(outputs)


def _warmup_policies(n_leader_actions: int, prepared: PreparedWarmup) -> FloatArray:
    design = prepared.design
    if design.support_size == 0:
        return np.empty((0, n_leader_actions), dtype=np.float64)
    vertices = np.eye(n_leader_actions, dtype=np.float64)[design.support_actions]
    return np.repeat(vertices, design.allocation_counts, axis=0)


def evaluation_checkpoints(first_sample: int, total_samples: int, interval: int) -> np.ndarray:
    """Return the sample counts at which estimates are evaluated and stored."""

    values = np.arange(first_sample, total_samples + 1, interval, dtype=np.int64)
    if values.size == 0 or values[-1] != total_samples:
        values = np.append(values, total_samples)
    return values


def checkpoint_positions(trace_sample_counts: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    """Locate each checkpoint in a stored trace, rejecting any that is absent."""

    positions = np.searchsorted(trace_sample_counts, checkpoints)
    if np.any(positions >= trace_sample_counts.size) or not np.array_equal(trace_sample_counts[positions], checkpoints):
        raise RuntimeError("evaluation checkpoint missing from stored estimate trace")
    return positions
