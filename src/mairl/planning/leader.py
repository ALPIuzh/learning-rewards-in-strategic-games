"""Optional leader optimization after reward estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

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
class LeaderOptimizationConfig:
    """Optional optimization of the leader after reward estimation."""

    policy: PolicyOptimizationConfig = field(default_factory=PolicyOptimizationConfig)


@dataclass(frozen=True, slots=True)
class LeaderOptimizationResult:
    policy: FloatArray
    follower_policy: FloatArray
    value: float
    optimization: SimplexOptimizationResult


def leader_value_and_gradient(problem: CompiledProblem, omega: FloatArray,
        leader_reward: FloatArray, policy: FloatArray) -> tuple[float, FloatArray]:  # fmt: skip
    """Evaluate the Stackelberg leader objective and its analytic gradient."""

    rewards = np.einsum("abd,d->ab", problem.features, omega, optimize=True)
    return _leader_value_and_gradient_precomputed(problem, omega, leader_reward, policy, rewards)


def _leader_value_and_gradient_precomputed(problem: CompiledProblem, omega: FloatArray,
        leader_reward: FloatArray, policy: FloatArray,
        pure_follower_rewards: FloatArray) -> tuple[float, FloatArray]:  # fmt: skip
    psi = np.einsum("a,abd->bd", policy, problem.features, optimize=True)
    logits = problem.eta * (psi @ omega)
    follower_policy = stable_softmax(logits)
    reward_by_follower_action = policy @ leader_reward
    value = float(follower_policy @ reward_by_follower_action)

    dz = problem.eta * pure_follower_rewards
    expected_dz = dz @ follower_policy
    d_follower_policy = follower_policy[None, :] * (dz - expected_dz[:, None])
    gradient = d_follower_policy @ reward_by_follower_action + leader_reward @ follower_policy
    return value, gradient


def optimize_leader_policy(problem: CompiledProblem, omega: ArrayLike, leader_reward: ArrayLike,
        config: LeaderOptimizationConfig | None = None, *, seed: int = 0) -> LeaderOptimizationResult:  # fmt: skip
    """Optimize the leader independently of reward estimation."""

    config = config or LeaderOptimizationConfig()
    coefficients = np.asarray(omega, dtype=np.float64)
    expected_omega_shape = (problem.feature_dimension,)
    if coefficients.shape != expected_omega_shape or not np.all(np.isfinite(coefficients)):
        raise ConfigurationError(f"omega must be finite with shape {expected_omega_shape}; got {coefficients.shape}")
    reward = np.asarray(leader_reward, dtype=np.float64)
    expected_reward_shape = (problem.n_leader_actions, problem.n_follower_actions)
    if reward.shape != expected_reward_shape or not np.all(np.isfinite(reward)):
        raise ConfigurationError(f"leader_reward must be finite with shape {expected_reward_shape}; got {reward.shape}")

    scalar_rewards = np.einsum("abd,d->ab", problem.features, coefficients, optimize=True)

    def objective(policy: FloatArray) -> tuple[float, FloatArray]:
        return _leader_value_and_gradient_precomputed(problem, coefficients, reward, policy, scalar_rewards)

    pure_follower_rewards = np.tensordot(problem.features, coefficients, axes=([2], [0]))

    def batch_objective(policies: FloatArray) -> FloatArray:
        follower_policies = stable_softmax(problem.eta * (policies @ pure_follower_rewards))
        rewards_by_follower_action = policies @ reward
        return np.einsum("nb,nb->n", follower_policies, rewards_by_follower_action, optimize=True)

    # Exact value of every pure-action vertex e_a, without ever forming an (n_leader,
    # n_leader) one-hot matrix: at policy=e_a, follower logits are just row a of
    # pure_follower_rewards. See maximize_on_simplex's vertex_scores docstring.
    follower_policy_at_vertices = stable_softmax(problem.eta * pure_follower_rewards)
    vertex_scores = np.einsum("ab,ab->a", follower_policy_at_vertices, reward, optimize=True)

    optimization = maximize_on_simplex(
        objective,
        n_actions=problem.n_leader_actions,
        config=config.policy,
        rng=np.random.default_rng(seed),
        batch_objective=batch_objective,
        vertex_scores=vertex_scores,
    )
    psi = np.einsum("a,abd->bd", optimization.policy, problem.features, optimize=True)
    follower_policy = stable_softmax(problem.eta * (psi @ coefficients))
    return LeaderOptimizationResult(
        policy=optimization.policy,
        follower_policy=follower_policy,
        value=optimization.objective,
        optimization=optimization,
    )


def _leader_value_and_gradient_batch(eta: float, leader_reward: FloatArray,
        pure_follower_rewards: FloatArray, policies: FloatArray, *,
        with_gradient: bool = True) -> tuple[FloatArray, FloatArray | None]:  # fmt: skip
    """Evaluate the (non-concave) leader objective for a batch of independent problems.

    ``pure_follower_rewards`` has shape ``(n_problems, n_leader, n_follower)`` (one
    slice per distinct omega); ``policies`` has shape ``(n_problems, n_starts, n_leader)``.
    Mirrors ``_information_gain_and_gradient_seed_batch`` in ``mairl.exploration.policy``.
    """

    logits = eta * np.einsum("nka,nab->nkb", policies, pure_follower_rewards, optimize=True)
    follower_policy = stable_softmax(logits)
    reward_by_follower_action = np.einsum("nka,ab->nkb", policies, leader_reward, optimize=True)
    value = np.einsum("nkb,nkb->nk", follower_policy, reward_by_follower_action, optimize=True)
    if not with_gradient:
        return value, None

    dz = eta * pure_follower_rewards
    expected_dz = np.einsum("nab,nkb->nka", dz, follower_policy, optimize=True)
    d_follower_policy = follower_policy[:, :, None, :] * (dz[:, None, :, :] - expected_dz[:, :, :, None])
    gradient = np.einsum("nkab,nkb->nka", d_follower_policy, reward_by_follower_action, optimize=True) + np.einsum(
        "ab,nkb->nka", leader_reward, follower_policy, optimize=True
    )
    return value, gradient


def optimize_leader_policy_batch(problem: CompiledProblem, omegas: FloatArray, leader_reward: ArrayLike,
        config: LeaderOptimizationConfig, seeds: tuple[int, ...]) -> tuple[LeaderOptimizationResult, ...]:  # fmt: skip
    """Batched, approximate version of :func:`optimize_leader_policy`.

    Optimizing the leader's non-concave objective independently for every checkpoint
    of an estimate trace dominates evaluation cost for security games with large leader
    action spaces, because :func:`maximize_on_simplex` runs one full SLSQP solve per
    pure-action vertex start. This batches the same set of starts across problems with a
    cheap vectorized projected-gradient prefilter (mirroring
    :func:`mairl.exploration.policy.select_information_policy_batch`), and only spends
    exact SLSQP solves on the top ``config.policy.batch_refine_starts`` candidates per
    problem. Falls back to :func:`optimize_leader_policy` for non-multistart configs and
    for any problem whose refinement fails entirely.
    """

    n_problems = omegas.shape[0]
    reward = np.asarray(leader_reward, dtype=np.float64)
    expected_reward_shape = (problem.n_leader_actions, problem.n_follower_actions)
    if omegas.shape != (n_problems, problem.feature_dimension):
        raise ConfigurationError("batched omegas have the wrong shape")
    if reward.shape != expected_reward_shape or not np.all(np.isfinite(reward)):
        raise ConfigurationError(f"leader_reward must be finite with shape {expected_reward_shape}; got {reward.shape}")
    if not np.all(np.isfinite(omegas)):
        raise ConfigurationError("batched omegas must be finite")
    if len(seeds) != n_problems:
        raise ConfigurationError("one seed is required per problem")

    if config.policy.method != "multistart":
        return tuple(
            optimize_leader_policy(problem, omegas[index], reward, config, seed=seeds[index])
            for index in range(n_problems)
        )

    n_leader = problem.n_leader_actions
    phi = problem.features
    pure_follower_rewards = np.einsum("abd,nd->nab", phi, omegas, optimize=True)

    vertex_scores = None
    if n_leader > _DENSE_VERTEX_LIMIT:
        follower_policy_at_vertices = stable_softmax(problem.eta * pure_follower_rewards)
        vertex_scores = np.einsum("nab,ab->na", follower_policy_at_vertices, reward, optimize=True)
    rngs = tuple(np.random.default_rng(seed) for seed in seeds)
    policies = _batch_starts(n_leader, config.policy, rngs, vertex_scores)
    start_count = policies.shape[1]

    def batch_objective(policies: FloatArray, *, with_gradient: bool = True) -> tuple[FloatArray, FloatArray | None]:
        return _leader_value_and_gradient_batch(
            problem.eta, reward, pure_follower_rewards, policies, with_gradient=with_gradient
        )

    values = _prefilter_batch(policies, batch_objective, config.policy)
    refine_count = min(config.policy.batch_refine_starts, start_count)

    results: list[LeaderOptimizationResult] = []
    for index in range(n_problems):
        candidates = np.argsort(values[index])[-refine_count:][::-1]
        omega = omegas[index]
        scalar_rewards = np.einsum("abd,d->ab", phi, omega, optimize=True)

        def local_objective(candidate_policy: FloatArray, current_omega: FloatArray = omega,
                current_rewards: FloatArray = scalar_rewards) -> tuple[float, FloatArray]:  # fmt: skip
            return _leader_value_and_gradient_precomputed(
                problem, current_omega, reward, candidate_policy, current_rewards
            )

        successful: list[tuple[FloatArray, float, FloatArray, float]] = []
        failures: list[str] = []
        for start_index in candidates:
            policy, value, gradient, residual, failure = _local_solve(
                local_objective, policies[index, start_index], config.policy
            )
            if failure is None:
                successful.append((policy, value, gradient, residual))
            else:
                failures.append(f"prefiltered start {start_index}: {failure}")
        if not successful:
            results.append(optimize_leader_policy(problem, omega, reward, config, seed=seeds[index]))
            continue
        policy, value, gradient, residual = max(successful, key=lambda item: item[1])
        psi = np.einsum("a,abd->bd", policy, phi, optimize=True)
        follower_policy = stable_softmax(problem.eta * (psi @ omega))
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
            LeaderOptimizationResult(
                policy=policy, follower_policy=follower_policy, value=optimization.objective, optimization=optimization
            )
        )
    return tuple(results)
