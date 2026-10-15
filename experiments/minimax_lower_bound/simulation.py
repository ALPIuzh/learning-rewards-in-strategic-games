"""Fast, explicit Active IRL loop for the lower-bound sufficient statistics."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from scipy.special import expit

from experiments.minimax_lower_bound.application import exact_hard_instance_metrics, make_minimax_instance
from experiments.minimax_lower_bound.solver import solve_one_hot_mle_batch
from mairl.core.math import project_simplex_rows
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.jobs import StudyJobResult

ProgressCallback = Callable[[int, int, int], None]


def run_hard_instance_batch(*, dimension: int, total_samples: int, seeds: tuple[int, ...],
        active_coordinates: tuple[int, ...], numerics: ExperimentNumerics,
        on_progress: ProgressCallback | None = None) -> tuple[StudyJobResult, ...]:  # fmt: skip
    """Run Algorithm 2 using the exact one-hot hard-instance sufficient statistics.

    The control flow mirrors Active IRL. Policy selection evaluates every vertex's
    information gain in closed form and checks stationarity before updating Bernoulli
    counts and the constrained MLE.
    """

    seed_count = len(seeds)
    if seed_count < 1 or len(active_coordinates) != seed_count:
        raise ConfigurationError("one active coordinate is required for every seed")
    if len(set(seeds)) != seed_count or any(seed < 0 for seed in seeds):
        raise ConfigurationError("execution seeds must be unique and nonnegative")
    if any(not 0 <= coordinate < dimension for coordinate in active_coordinates):
        raise ConfigurationError("active coordinates lie outside the hard-instance dimension")

    template = make_minimax_instance(dimension, total_samples, seed=0, active_coordinate=0)
    rngs = tuple(np.random.default_rng(seed) for seed in seeds)
    tie_uniforms = np.stack([rng.random(total_samples) for rng in rngs])
    outcome_uniforms = np.stack([rng.random(total_samples) for rng in rngs])
    active = np.asarray(active_coordinates, dtype=np.int64)
    counts = np.zeros((seed_count, dimension), dtype=np.int64)
    successes = np.zeros((seed_count, dimension), dtype=np.int64)
    omegas = np.zeros((seed_count, dimension), dtype=np.float64)

    selected_action_history = np.empty((seed_count, total_samples), dtype=np.int64)
    follower_actions = np.empty((seed_count, total_samples), dtype=np.int64)
    trace_interval = max(1, total_samples // 20)
    trace_sample_counts = np.arange(0, total_samples + 1, trace_interval, dtype=np.int64)
    if trace_sample_counts[-1] != total_samples:
        trace_sample_counts = np.append(trace_sample_counts, total_samples)
    omega_history = np.empty((seed_count, trace_sample_counts.size, dimension), dtype=np.float64)
    omega_history[:, 0] = omegas
    trace_index = 1
    information_gains = np.empty((seed_count, total_samples), dtype=np.float64)
    policy_residuals = np.empty((seed_count, total_samples), dtype=np.float64)
    mle_residuals = np.empty((seed_count, total_samples), dtype=np.float64)
    mle_iterations = np.empty((seed_count, total_samples), dtype=np.int64)

    progress_interval = max(1, total_samples // 20)
    started = perf_counter()
    for sample_index in range(total_samples):
        variances = expit(omegas) * (1.0 - expit(omegas))
        information_diagonal = 1.0 + counts * variances
        selected_coordinates, gains, residuals = select_hard_information_vertices(
            omegas,
            information_diagonal,
            rngs,
            tolerance=numerics.active_policy.projected_gradient_tolerance,
            tie_uniforms=tie_uniforms[:, sample_index],
        )
        selected_action_history[:, sample_index] = selected_coordinates
        information_gains[:, sample_index] = gains
        policy_residuals[:, sample_index] = residuals

        true_logits = template.epsilon * (selected_coordinates == active)
        probabilities = expit(true_logits)
        actions = (outcome_uniforms[:, sample_index] < probabilities).astype(np.int64)
        follower_actions[:, sample_index] = actions
        rows = np.arange(seed_count)
        counts[rows, selected_coordinates] += 1
        successes[rows, selected_coordinates] += actions

        mle = solve_one_hot_mle_batch(
            counts,
            successes,
            ridge=numerics.mle.ridge,
            radius=template.problem.radius,
            initial=omegas,
            tolerance=min(numerics.mle.objective_tolerance, 1e-10),
            projected_gradient_tolerance=numerics.mle.projected_gradient_tolerance,
            max_iterations=numerics.mle.max_iterations,
        )
        omegas = mle.omegas
        mle_residuals[:, sample_index] = mle.projected_gradient_residuals
        mle_iterations[:, sample_index] = mle.iterations

        completed = sample_index + 1
        if trace_index < trace_sample_counts.size and completed == trace_sample_counts[trace_index]:
            omega_history[:, trace_index] = omegas
            trace_index += 1
        if on_progress is not None and (completed == total_samples or completed % progress_interval == 0):
            for seed_index in range(seed_count):
                on_progress(seed_index, completed, total_samples)

    if trace_index != trace_sample_counts.size:
        raise RuntimeError("internal error: lower-bound coefficient trace is incomplete")
    elapsed = perf_counter() - started
    return _build_outputs(
        dimension=dimension,
        total_samples=total_samples,
        seeds=seeds,
        active_coordinates=active_coordinates,
        numerics=numerics,
        trace_sample_counts=trace_sample_counts,
        selected_action_history=selected_action_history,
        follower_actions=follower_actions,
        omega_history=omega_history,
        final_counts=counts,
        final_successes=successes,
        information_gains=information_gains,
        policy_residuals=policy_residuals,
        mle_residuals=mle_residuals,
        mle_iterations=mle_iterations,
        elapsed=elapsed,
    )


def select_hard_information_vertices(omegas: np.ndarray, information_diagonal: np.ndarray,
        rngs: tuple[np.random.Generator, ...], *, tolerance: float,
        tie_uniforms: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Evaluate every vertex exactly and verify the selected vertex is stationary."""

    probabilities = expit(omegas)
    variances = probabilities * (1.0 - probabilities)
    gains = variances / information_diagonal
    maximum = np.max(gains, axis=1, keepdims=True)
    tied = np.isclose(gains, maximum, rtol=1e-12, atol=1e-15)
    if tie_uniforms is None:
        tie_breakers = np.stack([rng.random(omegas.shape[1]) for rng in rngs])
        selected = np.argmax(np.where(tied, tie_breakers, -1.0), axis=1)
    else:
        uniforms = np.asarray(tie_uniforms, dtype=np.float64)
        if uniforms.shape != (omegas.shape[0],) or np.any((uniforms < 0.0) | (uniforms >= 1.0)):
            raise ConfigurationError("tie uniforms must contain one value in [0, 1) per seed")
        tie_counts = np.sum(tied, axis=1)
        ranks = np.minimum((uniforms * tie_counts).astype(np.int64), tie_counts - 1)
        selected = np.argmax(np.cumsum(tied, axis=1) > ranks[:, None], axis=1)
    rows = np.arange(omegas.shape[0])
    selected_gain = gains[rows, selected]

    selected_probabilities = probabilities[rows, selected]
    selected_variances = variances[rows, selected]
    inverse_information = 1.0 / information_diagonal[rows, selected]
    variance_derivative = selected_variances * (1.0 - 2.0 * selected_probabilities)
    gradients = variance_derivative[:, None] * omegas * inverse_information[:, None]
    gradients[rows, selected] += 2.0 * selected_variances * inverse_information
    scales = np.maximum(1.0, np.linalg.norm(gradients, axis=1))
    vertices = np.eye(omegas.shape[1], dtype=np.float64)[selected]
    projected = project_simplex_rows(vertices + gradients / scales[:, None])
    residuals = np.linalg.norm(projected - vertices, axis=1)
    failures = np.flatnonzero(residuals > tolerance)
    if failures.size:
        failed = int(failures[0])
        raise ConvergenceError(
            "closed-form hard-instance vertex is not a stationary information-policy "
            f"solution for seed index {failed}: residual={residuals[failed]:.3e}"
        )
    return selected, selected_gain, residuals


def _build_outputs(*, dimension: int, total_samples: int, seeds: tuple[int, ...], active_coordinates: tuple[int, ...],
        numerics: ExperimentNumerics, trace_sample_counts: np.ndarray, selected_action_history: np.ndarray,
        follower_actions: np.ndarray, omega_history: np.ndarray, final_counts: np.ndarray, final_successes: np.ndarray,
        information_gains: np.ndarray, policy_residuals: np.ndarray, mle_residuals: np.ndarray,
        mle_iterations: np.ndarray, elapsed: float) -> tuple[StudyJobResult, ...]:  # fmt: skip
    checkpoints = np.asarray([0, total_samples], dtype=np.int64)
    outputs: list[StudyJobResult] = []
    for index, (seed, coordinate) in enumerate(zip(seeds, active_coordinates, strict=True)):
        instance = make_minimax_instance(dimension, total_samples, seed, active_coordinate=coordinate)
        estimates = omega_history[index, [0, -1]]
        metrics = exact_hard_instance_metrics(instance, estimates)
        arrays: dict[str, np.ndarray] = {
            "trace_sample_counts": trace_sample_counts,
            "omega_history": omega_history[index],
            "evaluation_sample_counts": checkpoints,
            "selected_leader_actions": selected_action_history[index],
            "follower_actions": follower_actions[index],
            "final_action_counts": final_counts[index],
            "final_action_successes": final_successes[index],
            "warmup__design_weights": np.empty(0, dtype=np.float64),
            "warmup__support_actions": np.empty(0, dtype=np.int64),
            "warmup__allocation_counts": np.empty(0, dtype=np.int64),
            "warmup__coverage_eigenvalues": np.zeros(dimension, dtype=np.float64),
            "diagnostic__information_gain": information_gains[index],
            "diagnostic__mle_iterations": mle_iterations[index],
            "diagnostic__mle_residual": mle_residuals[index],
            "diagnostic__policy_residual": policy_residuals[index],
        }
        arrays.update({f"metric__{name}": values for name, values in metrics.items()})
        outputs.append(
            StudyJobResult(
                arrays=arrays,
                summary={
                    "job": {"method": "active", "warmup_enabled": False, "seed": seed},
                    "algorithm_seconds": elapsed / len(seeds),
                    "batch_algorithm_seconds": elapsed,
                    "batch_seed_count": len(seeds),
                    "mle_solver": "one-hot-sufficient-statistics",
                    "mle_ridge": numerics.mle.ridge,
                    "samples": total_samples,
                    "final_metrics": {name: float(values[-1]) for name, values in metrics.items()},
                    "warmup": {
                        "enabled": False,
                        "samples": 0,
                        "support_size": 0,
                        "design_weights": [],
                        "allocation_counts": [],
                        "coverage_eigenvalues": [0.0] * dimension,
                    },
                },
            )
        )
    return tuple(outputs)
