"""Active information-gain policy selection."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from mairl.core.math import stable_softmax
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError
from mairl.exploration.simplex import (
    _DENSE_VERTEX_LIMIT,
    PolicyOptimizationConfig,
    SimplexOptimizationResult,
    _batch_starts,
    _local_solve,
    _prefilter_batch,
    maximize_on_simplex,
)


@dataclass(frozen=True, slots=True)
class PolicySelectionResult:
    policy: FloatArray
    information_gain: float
    fisher_information: FloatArray
    optimization: SimplexOptimizationResult


def information_gain_and_gradient(problem: CompiledProblem, omega: FloatArray,
        inverse_information: FloatArray, policy: FloatArray) -> tuple[float, FloatArray]:  # fmt: skip
    """Evaluate ``trace(G^-1 I_mu)`` and its analytic policy gradient."""

    pure_rewards = np.tensordot(problem.features, omega, axes=([2], [0]))
    return _information_gain_and_gradient_precomputed(
        problem.features, problem.eta, inverse_information, pure_rewards, policy
    )


def _information_gain_and_gradient_precomputed(phi: FloatArray, eta: float, inverse_information: FloatArray,
        pure_rewards: FloatArray, policy: FloatArray) -> tuple[float, FloatArray]:  # fmt: skip
    """Hot-path implementation with policy-independent quantities cached."""

    psi = np.tensordot(policy, phi, axes=([0], [0]))
    logits = eta * (policy @ pure_rewards)
    probabilities = stable_softmax(logits)
    mean = probabilities @ psi
    centered = psi - mean
    h_centered = centered @ inverse_information
    gain = eta**2 * float(np.sum(probabilities[:, None] * centered * h_centered))

    # Differentiate p_b and psi_b with respect to every mu_a.
    dz = eta * pure_rewards
    expected_dz = dz @ probabilities
    dp = probabilities[None, :] * (dz - expected_dz[:, None])

    h_psi = psi @ inverse_information
    psi_quadratic = np.sum(psi * h_psi, axis=1)
    first_probability = dp @ psi_quadratic
    first_features = 2.0 * np.tensordot(phi, probabilities[:, None] * h_psi, axes=([1, 2], [0, 1]))

    d_mean = dp @ psi + np.tensordot(phi, probabilities, axes=([1], [0]))
    second = 2.0 * (d_mean @ (inverse_information @ mean))
    gradient = eta**2 * (first_probability + first_features - second)
    return gain, gradient


def _information_gain_batch_precomputed(phi: FloatArray, eta: float, inverse_information: FloatArray,
        pure_rewards: FloatArray, policies: FloatArray) -> FloatArray:  # fmt: skip
    """Evaluate the information criterion for many policies in one tensor operation."""

    follower_features = np.tensordot(policies, phi, axes=([1], [0]))
    probabilities = stable_softmax(eta * (policies @ pure_rewards))
    means = np.einsum("nb,nbd->nd", probabilities, follower_features, optimize=True)
    centered = follower_features - means[:, None, :]
    transformed = centered @ inverse_information
    return eta**2 * np.einsum("nb,nbd,nbd->n", probabilities, centered, transformed, optimize=True)


def _information_gain_and_gradient_seed_batch(phi: FloatArray, eta: float, inverse_information: FloatArray,
        pure_rewards: FloatArray, policies: FloatArray, *,
        with_gradient: bool = True) -> tuple[FloatArray, FloatArray | None]:  # fmt: skip
    """Evaluate information objectives for seed-by-multistart policy arrays."""

    psi = np.einsum("ska,abd->skbd", policies, phi, optimize=True)
    logits = eta * np.einsum("ska,sab->skb", policies, pure_rewards, optimize=True)
    probabilities = stable_softmax(logits)
    mean = np.einsum("skb,skbd->skd", probabilities, psi, optimize=True)
    centered = psi - mean[:, :, None, :]
    h_centered = np.einsum("skbd,sde->skbe", centered, inverse_information, optimize=True)
    gain = eta**2 * np.einsum("skb,skbd,skbd->sk", probabilities, centered, h_centered, optimize=True)
    if not with_gradient:
        return gain, None

    dz = eta * pure_rewards
    expected_dz = np.einsum("sab,skb->ska", dz, probabilities, optimize=True)
    dp = probabilities[:, :, None, :] * (dz[:, None, :, :] - expected_dz[:, :, :, None])
    h_psi = np.einsum("skbd,sde->skbe", psi, inverse_information, optimize=True)
    psi_quadratic = np.einsum("skbd,skbd->skb", psi, h_psi, optimize=True)
    first_probability = np.einsum("skab,skb->ska", dp, psi_quadratic, optimize=True)
    first_features = 2.0 * np.einsum("abd,skbd->ska", phi, probabilities[:, :, :, None] * h_psi, optimize=True)
    d_mean = np.einsum("skab,skbd->skad", dp, psi, optimize=True) + np.einsum(
        "abd,skb->skad", phi, probabilities, optimize=True
    )
    inverse_mean = np.einsum("sde,ske->skd", inverse_information, mean, optimize=True)
    second = 2.0 * np.einsum("skad,skd->ska", d_mean, inverse_mean, optimize=True)
    return gain, eta**2 * (first_probability + first_features - second)


def _vertex_information_gain(phi: FloatArray, eta: float, inverse_information: FloatArray,
        pure_rewards: FloatArray) -> FloatArray:  # fmt: skip
    """Exact information gain at every pure-action vertex e_a, one value per vertex.

    At policy=e_a, the average follower feature psi is exactly phi[a, :, :] -- no
    (n_leader, n_leader) one-hot matrix or matrix contraction over the leader-action
    axis is needed. O(n_leader * n_follower * d), matching the closed form used by
    optimize_leader_policy's vertex_scores.
    """

    probabilities = stable_softmax(eta * pure_rewards)
    mean = np.einsum("ab,abd->ad", probabilities, phi, optimize=True)
    centered = phi - mean[:, None, :]
    transformed = np.einsum("abd,de->abe", centered, inverse_information, optimize=True)
    return eta**2 * np.einsum("ab,abd,abd->a", probabilities, centered, transformed, optimize=True)


def select_information_policy(problem: CompiledProblem, omega: FloatArray, information_matrix: FloatArray,
        config: PolicyOptimizationConfig, rng: np.random.Generator) -> PolicySelectionResult:  # fmt: skip
    """Approximately maximize the paper's active Fisher-information criterion."""

    expected_shape = (problem.feature_dimension, problem.feature_dimension)
    if information_matrix.shape != expected_shape:
        raise ConfigurationError(f"information_matrix must have shape {expected_shape}; got {information_matrix.shape}")
    eigenvalues = np.linalg.eigvalsh(information_matrix)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise ConfigurationError("information_matrix must be finite and positive definite")
    inverse = np.linalg.inv(information_matrix)
    pure_rewards = np.tensordot(problem.features, omega, axes=([2], [0]))

    def objective(policy: FloatArray) -> tuple[float, FloatArray]:
        return _information_gain_and_gradient_precomputed(problem.features, problem.eta, inverse, pure_rewards, policy)

    def batch_objective(policies: FloatArray) -> FloatArray:
        return _information_gain_batch_precomputed(problem.features, problem.eta, inverse, pure_rewards, policies)

    vertex_scores = _vertex_information_gain(problem.features, problem.eta, inverse, pure_rewards)

    optimization = maximize_on_simplex(
        objective,
        n_actions=problem.n_leader_actions,
        config=config,
        rng=rng,
        batch_objective=batch_objective,
        vertex_scores=vertex_scores,
    )
    psi = np.einsum("a,abd->bd", optimization.policy, problem.features, optimize=True)
    probabilities = stable_softmax(problem.eta * (psi @ omega))
    centered = psi - probabilities @ psi
    fisher = problem.eta**2 * np.einsum("b,bd,be->de", probabilities, centered, centered, optimize=True)
    return PolicySelectionResult(
        policy=optimization.policy,
        information_gain=optimization.objective,
        fisher_information=fisher,
        optimization=optimization,
    )


def _vertex_information_gain_batch(phi: FloatArray, eta: float,
        inverse_information: FloatArray, pure_rewards: FloatArray) -> FloatArray:  # fmt: skip
    """Per-problem exact information gain at every pure-action vertex e_a.

    Batched analogue of _vertex_information_gain: phi is shared across problems
    (shape (n_leader, n_follower, d)); inverse_information and pure_rewards carry a
    leading per-problem axis. O(n_problems * n_leader * n_follower * d) -- linear in
    n_leader, so scoring every vertex is cheap even though selecting the top ones is
    what keeps the downstream dense arrays bounded.
    """

    probabilities = stable_softmax(eta * pure_rewards)
    mean = np.einsum("sab,abd->sad", probabilities, phi, optimize=True)
    centered = phi[None, :, :, :] - mean[:, :, None, :]
    transformed = np.einsum("sabd,sde->sabe", centered, inverse_information, optimize=True)
    return eta**2 * np.einsum("sab,sabd,sabd->sa", probabilities, centered, transformed, optimize=True)


def select_information_policy_batch(problem: CompiledProblem, omegas: FloatArray,
        information_matrices: FloatArray, config: PolicyOptimizationConfig,
        rngs: tuple[np.random.Generator, ...]) -> tuple[PolicySelectionResult, ...]:  # fmt: skip
    """Select seeded policies with masked projected ascent and scalar fallbacks."""

    seed_count = omegas.shape[0]
    dimension = problem.feature_dimension
    if omegas.shape != (seed_count, dimension):
        raise ConfigurationError("batched omegas have the wrong shape")
    if information_matrices.shape != (seed_count, dimension, dimension):
        raise ConfigurationError("batched information matrices have the wrong shape")
    if len(rngs) != seed_count:
        raise ConfigurationError("one random generator is required per seed")
    if config.method != "multistart":
        return tuple(
            select_information_policy(problem, omegas[index], information_matrices[index], config, rngs[index])
            for index in range(seed_count)
        )
    eigenvalues = np.linalg.eigvalsh(information_matrices)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues[:, 0] <= 0):
        raise ConfigurationError("batched information matrices must be positive definite")

    inverse = np.linalg.inv(information_matrices)
    pure_rewards = np.einsum("abd,sd->sab", problem.features, omegas, optimize=True)
    saved_states = [copy.deepcopy(rng.bit_generator.state) for rng in rngs]

    n_leader = problem.n_leader_actions
    vertex_scores = None
    if n_leader > _DENSE_VERTEX_LIMIT:
        vertex_scores = _vertex_information_gain_batch(problem.features, problem.eta, inverse, pure_rewards)
    policies = _batch_starts(n_leader, config, rngs, vertex_scores)
    start_count = policies.shape[1]

    def batch_objective(policies: FloatArray, *, with_gradient: bool = True) -> tuple[FloatArray, FloatArray | None]:
        return _information_gain_and_gradient_seed_batch(
            problem.features, problem.eta, inverse, pure_rewards, policies, with_gradient=with_gradient
        )

    values = _prefilter_batch(policies, batch_objective, config)
    refine_count = min(config.batch_refine_starts, start_count)

    results: list[PolicySelectionResult] = []
    for seed_index in range(seed_count):
        scalar_rewards = np.tensordot(problem.features, omegas[seed_index], axes=([2], [0]))
        candidates = np.argsort(values[seed_index])[-refine_count:][::-1]
        successful: list[tuple[FloatArray, float, FloatArray, float]] = []
        failures: list[str] = []
        for start_index in candidates:
            policy, value, gradient, residual, failure = _local_solve(
                lambda candidate, index=seed_index, rewards=scalar_rewards: _information_gain_and_gradient_precomputed(
                    problem.features, problem.eta, inverse[index], rewards, candidate
                ),
                policies[seed_index, start_index],
                config,
            )
            if failure is None:
                successful.append((policy, value, gradient, residual))
            else:
                failures.append(f"prefiltered start {start_index}: {failure}")
        if not successful:
            rngs[seed_index].bit_generator.state = saved_states[seed_index]
            results.append(
                select_information_policy(
                    problem, omegas[seed_index], information_matrices[seed_index], config, rngs[seed_index]
                )
            )
            continue
        policy, value, gradient, residual = max(successful, key=lambda item: item[1])
        psi = np.einsum("a,abd->bd", policy, problem.features, optimize=True)
        probabilities = stable_softmax(problem.eta * (psi @ omegas[seed_index]))
        centered = psi - probabilities @ psi
        fisher = problem.eta**2 * np.einsum("b,bd,be->de", probabilities, centered, centered, optimize=True)
        optimization = SimplexOptimizationResult(
            policy=policy,
            objective=float(value),
            gradient=gradient,
            projected_gradient_residual=residual,
            method="batched_multistart",
            attempted_starts=start_count,
            converged_starts=len(successful),
            failed_start_messages=tuple(failures),
            grid_points=0,
        )
        results.append(
            PolicySelectionResult(
                policy=policy,
                information_gain=optimization.objective,
                fisher_information=fisher,
                optimization=optimization,
            )
        )
    return tuple(results)
