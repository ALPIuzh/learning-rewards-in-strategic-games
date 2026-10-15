"""Exact sufficient-statistic MLE for the lower-bound hard instance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from mairl.core.math import project_ball_rows
from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError, ConvergenceError


@dataclass(frozen=True, slots=True)
class OneHotMLEResult:
    """Verified batched constrained MLEs and their numerical diagnostics."""

    omegas: FloatArray
    projected_gradient_residuals: FloatArray
    iterations: np.ndarray


def solve_one_hot_mle_batch(counts: np.ndarray, successes: np.ndarray, *, ridge: float, radius: float,
        initial: FloatArray | None = None, tolerance: float = 1e-10, projected_gradient_tolerance: float = 1e-6,
        max_iterations: int = 50) -> OneHotMLEResult:  # fmt: skip
    r"""Solve the ridge-logistic MLE from per-coordinate Bernoulli counts.

    With one-hot leader policies, the log likelihood is

    ``sum_j successes_j omega_j - counts_j log(1 + exp(omega_j))``.

    The unconstrained coordinate roots are independent. If they leave the Euclidean
    ball, the boundary KKT system has one scalar multiplier and a diagonal-plus-arrow
    Jacobian, so every Newton step is vectorized across seeds and coordinates.
    """

    observations = np.asarray(counts, dtype=np.int64)
    positive = np.asarray(successes, dtype=np.int64)
    if observations.ndim != 2 or positive.shape != observations.shape:
        raise ConfigurationError("counts and successes must have the same two-dimensional shape")
    if np.any(observations < 0) or np.any(positive < 0) or np.any(positive > observations):
        raise ConfigurationError("Bernoulli sufficient statistics are invalid")
    if ridge <= 0 or radius <= 0 or tolerance <= 0 or projected_gradient_tolerance <= 0:
        raise ConfigurationError("MLE ridge, radius, and tolerances must be positive")
    if max_iterations < 1:
        raise ConfigurationError("max_iterations must be positive")

    shape = observations.shape
    if initial is None:
        omega = np.zeros(shape, dtype=np.float64)
    else:
        omega = np.asarray(initial, dtype=np.float64).copy()
        if omega.shape != shape or not np.all(np.isfinite(omega)):
            raise ConfigurationError("initial MLE values have the wrong shape or are non-finite")
        omega = np.clip(omega, -radius, radius)

    unconstrained_iterations = _solve_clipped_coordinate_roots(
        observations, positive, ridge, radius, omega, tolerance, max_iterations
    )
    norms = np.linalg.norm(omega, axis=1)
    boundary = norms >= radius - 10.0 * tolerance
    iterations = unconstrained_iterations.copy()
    if np.any(boundary):
        boundary_omega = omega[boundary].copy()
        boundary_iterations = _solve_boundary_kkt(
            observations[boundary], positive[boundary], ridge, radius, boundary_omega, tolerance, max_iterations
        )
        omega[boundary] = boundary_omega
        iterations[boundary] += boundary_iterations

    gradients = positive - observations * expit(omega) - ridge * omega
    residuals = _projected_gradient_residuals(omega, gradients, radius)
    violations = np.maximum(0.0, np.linalg.norm(omega, axis=1) - radius)
    failures = np.flatnonzero((residuals > projected_gradient_tolerance) | (violations > 10.0 * tolerance))
    if failures.size:
        index = int(failures[0])
        raise ConvergenceError(
            "one-hot MLE failed verified postconditions for seed index "
            f"{index}: residual={residuals[index]:.3e}, violation={violations[index]:.3e}"
        )
    return OneHotMLEResult(omega, residuals, iterations)


def _solve_clipped_coordinate_roots(counts: np.ndarray, successes: np.ndarray, ridge: float, radius: float,
        omega: FloatArray, tolerance: float, max_iterations: int) -> np.ndarray:  # fmt: skip
    """Safeguarded Newton solve of the independent unconstrained roots."""

    lower = np.full_like(omega, -radius)
    upper = np.full_like(omega, radius)
    iterations = np.zeros(omega.shape[0], dtype=np.int64)
    converged = np.zeros(omega.shape[0], dtype=np.bool_)
    for iteration in range(1, max_iterations + 1):
        probabilities = expit(omega)
        gradient = successes - counts * probabilities - ridge * omega
        curvature = counts * probabilities * (1.0 - probabilities) + ridge
        lower = np.where(gradient > 0.0, omega, lower)
        upper = np.where(gradient < 0.0, omega, upper)
        candidate = omega + gradient / curvature
        candidate = np.where((candidate > lower) & (candidate < upper), candidate, 0.5 * (lower + upper))
        candidate = np.clip(candidate, -radius, radius)
        steps = np.max(np.abs(candidate - omega), axis=1)
        active = ~converged
        omega[active] = candidate[active]
        newly_converged = active & (steps <= tolerance)
        iterations[newly_converged] = iteration
        converged |= newly_converged
        if np.all(converged):
            return iterations
    raise ConvergenceError("one-hot unconstrained coordinate solve did not converge")


def _solve_boundary_kkt(counts: np.ndarray, successes: np.ndarray, ridge: float, radius: float,
        omega: FloatArray, tolerance: float, max_iterations: int) -> np.ndarray:  # fmt: skip
    """Solve the active-ball KKT equations with vectorized arrowhead Newton steps."""

    omega[:] = project_ball_rows(omega, radius)
    probabilities = expit(omega)
    likelihood_gradient = successes - counts * probabilities - ridge * omega
    multiplier = np.maximum(0.0, np.einsum("sd,sd->s", omega, likelihood_gradient, optimize=True) / radius**2)

    iterations = np.zeros(omega.shape[0], dtype=np.int64)
    solved = np.zeros(omega.shape[0], dtype=np.bool_)
    for iteration in range(1, max_iterations + 1):
        probabilities = expit(omega)
        likelihood_gradient = successes - counts * probabilities - ridge * omega
        stationarity = likelihood_gradient - multiplier[:, None] * omega
        norm_residual = np.einsum("sd,sd->s", omega, omega, optimize=True) - radius**2
        residual = np.maximum(np.linalg.norm(stationarity, axis=1), np.abs(norm_residual))
        converged = residual <= tolerance
        newly_solved = ~solved & converged
        iterations[newly_solved] = iteration
        solved |= newly_solved
        if np.all(solved):
            return iterations

        diagonal = counts * probabilities * (1.0 - probabilities) + ridge + multiplier[:, None]
        inverse_diagonal = 1.0 / diagonal
        denominator = 2.0 * np.sum(omega * omega * inverse_diagonal, axis=1)
        delta_multiplier = (norm_residual + 2.0 * np.sum(omega * stationarity * inverse_diagonal, axis=1)) / denominator
        delta_omega = (stationarity - omega * delta_multiplier[:, None]) * inverse_diagonal

        steps = np.ones(omega.shape[0], dtype=np.float64)
        decreasing = delta_multiplier < 0.0
        steps[decreasing] = np.minimum(
            1.0, 0.9 * multiplier[decreasing] / np.maximum(-delta_multiplier[decreasing], np.finfo(np.float64).tiny)
        )
        accepted = solved.copy()
        for _ in range(20):
            candidate_omega = omega + steps[:, None] * delta_omega
            candidate_multiplier = multiplier + steps * delta_multiplier
            candidate_probabilities = expit(candidate_omega)
            candidate_gradient = successes - counts * candidate_probabilities - ridge * candidate_omega
            candidate_stationarity = candidate_gradient - candidate_multiplier[:, None] * candidate_omega
            candidate_norm = np.einsum("sd,sd->s", candidate_omega, candidate_omega, optimize=True) - radius**2
            candidate_residual = np.maximum(np.linalg.norm(candidate_stationarity, axis=1), np.abs(candidate_norm))
            newly_accepted = (
                ~accepted
                & (candidate_multiplier >= 0.0)
                & np.isfinite(candidate_residual)
                & (candidate_residual < residual)
            )
            omega[newly_accepted] = candidate_omega[newly_accepted]
            multiplier[newly_accepted] = candidate_multiplier[newly_accepted]
            accepted |= newly_accepted
            if np.all(accepted):
                break
            steps[~accepted] *= 0.5
        if not np.all(accepted):
            failed = int(np.flatnonzero(~accepted)[0])
            raise ConvergenceError(f"one-hot boundary KKT line search failed for seed index {failed}")
    raise ConvergenceError("one-hot boundary KKT solve did not converge")


def _projected_gradient_residuals(omega: FloatArray, gradient: FloatArray, radius: float) -> FloatArray:
    scales = np.maximum(1.0, np.linalg.norm(gradient, axis=1))
    projected = project_ball_rows(omega + gradient / scales[:, None], radius)
    return np.linalg.norm(projected - omega, axis=1)
