"""Vectorized mathematical primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError


def stable_softmax(logits: FloatArray, axis: int = -1) -> FloatArray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def validate_policy(policy: ArrayLike, n_actions: int, *, tolerance: float = 1e-10) -> FloatArray:
    mu = np.asarray(policy, dtype=np.float64)
    if mu.shape != (n_actions,):
        raise ConfigurationError(f"policy must have shape ({n_actions},); got {mu.shape}")
    if not np.all(np.isfinite(mu)):
        raise ConfigurationError("policy contains NaN or infinite values")
    if np.min(mu) < -tolerance or abs(float(np.sum(mu)) - 1.0) > tolerance:
        raise ConfigurationError(f"policy must be a probability vector; min={np.min(mu):.3g}, sum={np.sum(mu):.17g}")
    clipped = np.maximum(mu, 0.0)
    return clipped / np.sum(clipped)


def project_ball(vector: FloatArray, radius: float) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= radius:
        return vector.copy()
    return vector * (radius / norm)


def project_ball_rows(vectors: FloatArray, radius: float) -> FloatArray:
    """Project a matrix of row vectors onto the same Euclidean ball."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ConfigurationError("batched ball projection requires a two-dimensional array")
    norms = np.linalg.norm(values, axis=1)
    scales = np.minimum(1.0, radius / np.maximum(norms, np.finfo(np.float64).tiny))
    return values * scales[:, None]


def project_simplex(vector: FloatArray) -> FloatArray:
    """Euclidean projection onto the probability simplex."""

    values = np.asarray(vector, dtype=np.float64)
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, values.size + 1)
    valid = sorted_values - (cumulative - 1.0) / indices > 0
    rho = int(np.flatnonzero(valid)[-1])
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - theta, 0.0)


def project_simplex_rows(vectors: FloatArray) -> FloatArray:
    """Project arrays of vectors onto the simplex along their final axis."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim < 2:
        raise ConfigurationError("batched simplex projection requires at least two dimensions")
    sorted_values = np.sort(values, axis=-1)[..., ::-1]
    cumulative = np.cumsum(sorted_values, axis=-1)
    indices = np.arange(1, values.shape[-1] + 1)
    valid = sorted_values - (cumulative - 1.0) / indices > 0
    rho = np.sum(valid, axis=-1) - 1
    selected = np.take_along_axis(cumulative, rho[..., None], axis=-1)[..., 0]
    theta = (selected - 1.0) / (rho + 1)
    return np.maximum(values - theta[..., None], 0.0)


def follower_features(problem: CompiledProblem, policy: ArrayLike) -> FloatArray:
    mu = validate_policy(policy, problem.n_leader_actions)
    return np.einsum("a,abd->bd", mu, problem.features, optimize=True)


def follower_probabilities(problem: CompiledProblem, policy: ArrayLike, omega: ArrayLike) -> FloatArray:
    coefficients = np.asarray(omega, dtype=np.float64)
    if coefficients.shape != (problem.feature_dimension,):
        raise ConfigurationError(f"omega must have shape ({problem.feature_dimension},); got {coefficients.shape}")
    psi = follower_features(problem, policy)
    return stable_softmax(problem.eta * (psi @ coefficients))


def fisher_information(problem: CompiledProblem, policy: ArrayLike, omega: ArrayLike) -> FloatArray:
    """Return ``eta^2 Cov[psi_mu(b)]`` in ambient coefficient coordinates."""

    psi = follower_features(problem, policy)
    probabilities = follower_probabilities(problem, policy, omega)
    mean = probabilities @ psi
    centered = psi - mean
    return problem.eta**2 * np.einsum("b,bd,be->de", probabilities, centered, centered, optimize=True)


def cumulative_fisher_information(problem: CompiledProblem, policies: ArrayLike, omega: ArrayLike) -> FloatArray:
    policies_array = np.asarray(policies, dtype=np.float64)
    if policies_array.size == 0:
        return np.zeros((problem.feature_dimension, problem.feature_dimension))
    if policies_array.ndim != 2 or policies_array.shape[1] != problem.n_leader_actions:
        raise ConfigurationError(f"policies must have shape (observations, leader_actions); got {policies_array.shape}")
    coefficients = np.asarray(omega, dtype=np.float64)
    psi = np.einsum("na,abd->nbd", policies_array, problem.features, optimize=True)
    probabilities = stable_softmax(problem.eta * np.einsum("nbd,d->nb", psi, coefficients))
    means = np.einsum("nb,nbd->nd", probabilities, psi, optimize=True)
    centered = psi - means[:, None, :]
    return problem.eta**2 * np.einsum("nb,nbd,nbe->de", probabilities, centered, centered, optimize=True)


def cumulative_fisher_information_batch(problem: CompiledProblem,
        policies: FloatArray, omegas: FloatArray) -> FloatArray:  # fmt: skip
    """Evaluate cumulative Fisher matrices for equally long seeded trajectories."""

    policy_array = np.asarray(policies, dtype=np.float64)
    coefficients = np.asarray(omegas, dtype=np.float64)
    if policy_array.ndim != 3 or policy_array.shape[2] != problem.n_leader_actions:
        raise ConfigurationError("batched policies must have shape (seeds, observations, leader_actions)")
    if coefficients.shape != (policy_array.shape[0], problem.feature_dimension):
        raise ConfigurationError("batched coefficients must have shape (seeds, feature_dimension)")
    if policy_array.shape[1] == 0:
        return np.zeros((policy_array.shape[0], problem.feature_dimension, problem.feature_dimension))
    psi = np.einsum("sna,abd->snbd", policy_array, problem.features, optimize=True)
    logits = problem.eta * np.einsum("snbd,sd->snb", psi, coefficients, optimize=True)
    probabilities = stable_softmax(logits)
    means = np.einsum("snb,snbd->snd", probabilities, psi, optimize=True)
    centered = psi - means[:, :, None, :]
    return problem.eta**2 * np.einsum("snb,snbd,snbe->sde", probabilities, centered, centered, optimize=True)


def projected_simplex_gradient_residual(policy: FloatArray, gradient: FloatArray) -> float:
    return float(np.linalg.norm(policy - project_simplex(policy + gradient)))
