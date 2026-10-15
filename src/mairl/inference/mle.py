"""Public flow for strict constrained maximum-likelihood estimation.

The functions in this module deliberately read like the algorithm: validate and
prepare the observations, try exact Newton, verify the candidate, and fall back to
an exact constrained solve when the coefficient ball is active. Numerical details
live in the private sibling modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.math import project_ball, project_ball_rows
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError
from mairl.inference._mle_likelihood import (
    PreparedLikelihood,
    evaluate_prepared,
    evaluate_prepared_batch,
    likelihood_and_gradient_prepared,
    prepare_likelihood,
    prepare_likelihood_batch,
    validate_dataset,
)
from mairl.inference._mle_solver import (
    NewtonResult,
    solution_failures,
    solve_constrained,
    unconstrained_newton,
    unconstrained_newton_batch,
)
from mairl.inference._mle_types import MLEConfig, MLEResult, MLESolver
from mairl.inference.data import InteractionDataset

__all__ = [
    "MLEConfig",
    "MLEResult",
    "MLESolver",
    "likelihood_and_gradient",
    "likelihood_gradient_and_hessian",
    "solve_mle",
    "solve_mle_batch",
]


def likelihood_and_gradient(problem: CompiledProblem, dataset: InteractionDataset,
        omega: ArrayLike, *, ridge: float) -> tuple[float, float, FloatArray]:  # fmt: skip
    """Return unpenalized likelihood, penalized objective, and objective gradient."""

    coefficients = np.asarray(omega, dtype=np.float64)
    if coefficients.shape != (problem.feature_dimension,):
        raise ConfigurationError("omega has the wrong shape")
    if dataset.size == 0:
        penalty = 0.5 * ridge * float(coefficients @ coefficients)
        return 0.0, -penalty, -ridge * coefficients

    prepared = prepare_likelihood(problem, dataset)
    return likelihood_and_gradient_prepared(problem, prepared, coefficients, ridge)


def likelihood_gradient_and_hessian(problem: CompiledProblem, dataset: InteractionDataset,
        omega: ArrayLike, *, ridge: float) -> tuple[float, float, FloatArray, FloatArray]:  # fmt: skip
    """Return likelihood, penalized objective, gradient, and exact Hessian."""

    coefficients = np.asarray(omega, dtype=np.float64)
    if coefficients.shape != (problem.feature_dimension,):
        raise ConfigurationError("omega has the wrong shape")
    if dataset.size == 0:
        penalty = 0.5 * ridge * float(coefficients @ coefficients)
        hessian = -ridge * np.eye(problem.feature_dimension)
        return 0.0, -penalty, -ridge * coefficients, hessian

    prepared = prepare_likelihood(problem, dataset)
    log_likelihood, penalized, gradient, hessian = evaluate_prepared(
        problem, prepared, coefficients, ridge, with_hessian=True
    )
    if hessian is None:
        raise RuntimeError("internal error: Hessian was not computed")
    return log_likelihood, penalized, gradient, hessian


def solve_mle(problem: CompiledProblem, dataset: InteractionDataset, config: MLEConfig,
        *, initial: ArrayLike | None = None) -> MLEResult:  # fmt: skip
    """Solve the optionally ridge-regularized MLE and verify optimality.

    Newton first solves the unconstrained problem using the exact Hessian. An
    interior solution is already the constrained optimum. Otherwise, its Euclidean
    projection is only a warm start for the exact constrained solver.
    """

    validate_dataset(problem, dataset)
    dimension = problem.feature_dimension
    if problem.is_degenerate or dataset.size == 0:
        omega = np.zeros(dimension, dtype=np.float64)
        log_likelihood, penalized, gradient = likelihood_and_gradient(problem, dataset, omega, ridge=config.ridge)
        return MLEResult(omega, log_likelihood, penalized, gradient, 0.0, 0, 1)

    initial_point = _initial_point(problem, initial)
    prepared = prepare_likelihood(problem, dataset)
    if config.solver != "newton":
        return solve_constrained(problem, prepared, config, initial_point)

    newton = unconstrained_newton(problem, prepared, config, initial_point)
    if newton.converged:
        verified = _verified_newton_result(problem, prepared, config, newton)
        if verified is not None:
            return verified

    return solve_constrained(
        problem,
        prepared,
        config,
        project_ball(newton.omega, problem.radius),
        prior_iterations=newton.iterations,
        prior_evaluations=newton.function_evaluations,
    )


def solve_mle_batch(problem: CompiledProblem, policies: ArrayLike, follower_actions: ArrayLike,
        config: MLEConfig, *, initial: ArrayLike | None = None) -> tuple[MLEResult, ...]:  # fmt: skip
    """Solve equally long seeded MLEs with masked Newton and strict scalar fallbacks."""

    policy_array, action_array = _validate_batch(problem, policies, follower_actions)
    seed_count = policy_array.shape[0]
    initial_points = _initial_points(problem, seed_count, initial)
    if problem.is_degenerate or policy_array.shape[1] == 0 or config.solver != "newton":
        return tuple(
            solve_mle(
                problem,
                InteractionDataset(policy_array[index], action_array[index]),
                config,
                initial=initial_points[index],
            )
            for index in range(seed_count)
        )

    prepared = prepare_likelihood_batch(problem, policy_array, action_array)
    omegas, converged, iterations, evaluations = unconstrained_newton_batch(problem, prepared, config, initial_points)
    log_likelihoods, penalized_objectives, gradients, _ = evaluate_prepared_batch(
        problem, prepared, omegas, config.ridge, with_hessian=False
    )
    return tuple(
        _finish_batched_seed(
            problem,
            prepared.select(index),
            config,
            omegas[index],
            gradients[index],
            float(log_likelihoods[index]),
            float(penalized_objectives[index]),
            bool(converged[index]),
            int(iterations[index]),
            int(evaluations[index]),
        )
        for index in range(seed_count)
    )


def _initial_point(problem: CompiledProblem, initial: ArrayLike | None) -> FloatArray:
    if initial is None:
        return np.zeros(problem.feature_dimension, dtype=np.float64)
    candidate = np.asarray(initial, dtype=np.float64)
    if candidate.shape != (problem.feature_dimension,) or not np.all(np.isfinite(candidate)):
        raise ConfigurationError("initial MLE point is invalid")
    return project_ball(candidate, problem.radius)


def _initial_points(problem: CompiledProblem, seed_count: int, initial: ArrayLike | None) -> FloatArray:
    if initial is None:
        return np.zeros((seed_count, problem.feature_dimension), dtype=np.float64)
    candidates = np.asarray(initial, dtype=np.float64)
    if candidates.shape != (seed_count, problem.feature_dimension) or not np.all(np.isfinite(candidates)):
        raise ConfigurationError("batched initial MLE points must have shape (seeds, feature_dimension)")
    return project_ball_rows(candidates, problem.radius)


def _validate_batch(problem: CompiledProblem, policies: ArrayLike,
        follower_actions: ArrayLike) -> tuple[FloatArray, np.ndarray]:  # fmt: skip
    policy_array = np.asarray(policies, dtype=np.float64)
    action_array = np.asarray(follower_actions, dtype=np.int64)
    if policy_array.ndim != 3 or policy_array.shape[2] != problem.n_leader_actions:
        raise ConfigurationError("batched policies must have shape (seeds, observations, leader_actions)")
    if action_array.shape != policy_array.shape[:2]:
        raise ConfigurationError("batched follower actions must have shape (seeds, observations)")
    if action_array.size and (np.min(action_array) < 0 or np.max(action_array) >= problem.n_follower_actions):
        raise ConfigurationError("batched datasets contain an invalid follower action")
    return policy_array, action_array


def _verified_newton_result(problem: CompiledProblem, prepared: PreparedLikelihood,
        config: MLEConfig, newton: NewtonResult) -> MLEResult | None:  # fmt: skip
    log_likelihood, penalized, gradient = likelihood_and_gradient_prepared(
        problem, prepared, newton.omega, config.ridge
    )
    failures, residual = solution_failures(problem, config, newton.omega, gradient)
    if failures:
        return None
    return MLEResult(
        omega=newton.omega,
        log_likelihood=log_likelihood,
        penalized_objective=penalized,
        gradient=gradient,
        projected_gradient_residual=residual,
        iterations=newton.iterations,
        function_evaluations=newton.function_evaluations,
    )


def _finish_batched_seed(problem: CompiledProblem, prepared: PreparedLikelihood, config: MLEConfig,
        omega: FloatArray, gradient: FloatArray, log_likelihood: float, penalized_objective: float,
        converged: bool, iterations: int, evaluations: int) -> MLEResult:  # fmt: skip
    if converged:
        failures, residual = solution_failures(problem, config, omega, gradient)
        if not failures:
            return MLEResult(
                omega=omega.copy(),
                log_likelihood=log_likelihood,
                penalized_objective=penalized_objective,
                gradient=gradient.copy(),
                projected_gradient_residual=residual,
                iterations=iterations,
                function_evaluations=evaluations,
            )
    return solve_constrained(
        problem,
        prepared,
        config,
        project_ball(omega, problem.radius),
        prior_iterations=iterations,
        prior_evaluations=evaluations,
    )
