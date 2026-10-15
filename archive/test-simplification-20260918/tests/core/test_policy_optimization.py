from __future__ import annotations

import numpy as np

from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.exploration.policy import (
    _information_gain_batch_precomputed,
    information_gain_and_gradient,
    select_information_policy,
    select_information_policy_batch,
)
from mairl.exploration.simplex import PolicyOptimizationConfig, simplex_grid
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, solve_mle
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)


def test_batched_grid_information_matches_scalar_evaluation(small_problem) -> None:
    rng = np.random.default_rng(7)
    policies = simplex_grid(3, resolution=20, max_points=1_000)
    omega = np.array([0.3, -0.5])
    inverse_information = np.linalg.inv(np.array([[2.0, 0.2], [0.2, 1.3]]))
    pure_rewards = np.tensordot(small_problem.features, omega, axes=([2], [0]))

    batched = _information_gain_batch_precomputed(
        small_problem.features, small_problem.eta, inverse_information, pure_rewards, policies
    )
    scalar = np.array(
        [information_gain_and_gradient(small_problem, omega, inverse_information, policy)[0] for policy in policies]
    )

    assert np.allclose(batched, scalar, atol=1e-13)
    assert simplex_grid(3, 20, 1_000) is policies
    assert not policies.flags.writeable
    assert np.isfinite(batched[int(rng.integers(batched.size))])


def test_batched_policy_prefilter_matches_full_multistart(small_problem) -> None:
    seed_count = 8
    rng = np.random.default_rng(44)
    omegas = rng.normal(size=(seed_count, 2))
    information = np.repeat(np.array([[[2.0, 0.2], [0.2, 1.3]]]), seed_count, axis=0)
    config = PolicyOptimizationConfig(
        random_starts=4, projected_gradient_tolerance=1e-6, batch_prefilter_iterations=4, batch_refine_starts=1
    )
    scalar_rngs = tuple(np.random.default_rng(90 + index) for index in range(seed_count))
    batch_rngs = tuple(np.random.default_rng(90 + index) for index in range(seed_count))
    scalar = tuple(
        select_information_policy(small_problem, omegas[index], information[index], config, scalar_rngs[index])
        for index in range(seed_count)
    )
    batched = select_information_policy_batch(small_problem, omegas, information, config, batch_rngs)

    assert np.allclose(
        [result.information_gain for result in batched], [result.information_gain for result in scalar], atol=1e-10
    )
    assert all(
        result.optimization.projected_gradient_residual <= config.projected_gradient_tolerance for result in batched
    )


def test_batched_grid_leader_values_match_scalar_evaluation(small_problem) -> None:
    omega = np.array([0.3, -0.5])
    reward = np.array([[0.0, 0.4, 1.0], [0.8, -0.1, 0.2], [0.3, 0.7, -0.2]])
    policy_config = PolicyOptimizationConfig(
        method="grid", grid_resolution=20, grid_max_points=1_000, grid_batch_size=7, refine_grid_solution=False
    )
    result = optimize_leader_policy(small_problem, omega, reward, LeaderOptimizationConfig(policy_config))
    points = simplex_grid(3, 20, 1_000)
    scalar_values = np.array([leader_value_and_gradient(small_problem, omega, reward, policy)[0] for policy in points])

    assert np.isclose(result.value, np.max(scalar_values), atol=1e-14)
    assert np.array_equal(result.policy, points[int(np.argmax(scalar_values))])


def test_batched_leader_policy_matches_sequential(small_problem) -> None:
    n_problems = 5
    reward = np.array([[0.0, 0.4, 1.0], [0.8, -0.1, 0.2], [0.3, 0.7, -0.2]])
    rng = np.random.default_rng(7)
    omegas = rng.normal(scale=0.6, size=(n_problems, 2))
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=6, batch_refine_starts=2)
    )
    seeds = tuple(300 + index for index in range(n_problems))

    sequential = [
        optimize_leader_policy(small_problem, omegas[index], reward, config, seed=seeds[index])
        for index in range(n_problems)
    ]
    batched = optimize_leader_policy_batch(small_problem, omegas, reward, config, seeds)

    assert np.allclose([result.value for result in batched], [result.value for result in sequential], atol=1e-8)
    for batched_result, sequential_result in zip(batched, sequential, strict=True):
        assert np.allclose(batched_result.policy, sequential_result.policy, atol=1e-6)
        assert np.allclose(batched_result.follower_policy, sequential_result.follower_policy, atol=1e-6)


def test_batched_leader_policy_grid_fallback_matches_scalar(small_problem) -> None:
    reward = np.array([[0.0, 0.4, 1.0], [0.8, -0.1, 0.2], [0.3, 0.7, -0.2]])
    rng = np.random.default_rng(8)
    omegas = rng.normal(scale=0.6, size=(3, 2))
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="grid", grid_resolution=20, grid_max_points=1_000, refine_grid_solution=False)
    )
    seeds = (10, 11, 12)

    sequential = [
        optimize_leader_policy(small_problem, omegas[index], reward, config, seed=seeds[index]) for index in range(3)
    ]
    batched = optimize_leader_policy_batch(small_problem, omegas, reward, config, seeds)
    assert np.allclose([result.value for result in batched], [result.value for result in sequential], atol=1e-14)


def test_mle_and_two_policy_optimizers(small_problem) -> None:
    rng = np.random.default_rng(22)
    true_omega = np.array([0.45, -0.65])
    oracle = SoftmaxFollowerOracle(small_problem, true_omega)
    policies = np.repeat(np.eye(3), repeats=150, axis=0)
    actions = np.array([oracle.sample(policy, rng) for policy in policies])
    dataset = InteractionDataset(policies, actions)
    mle = solve_mle(small_problem, dataset, MLEConfig(ridge=1e-4))
    assert np.linalg.norm(mle.omega - true_omega) < 0.3
    assert mle.projected_gradient_residual <= 1e-6

    information = 2.0 * np.eye(2)
    multistart = select_information_policy(
        small_problem, mle.omega, information, PolicyOptimizationConfig(method="multistart", random_starts=4), rng
    )
    grid = select_information_policy(
        small_problem,
        mle.omega,
        information,
        PolicyOptimizationConfig(method="grid", grid_resolution=30, grid_max_points=10_000),
        rng,
    )
    assert multistart.information_gain >= grid.information_gain - 1e-7

    leader_reward = np.array([[0.0, 0.4, 1.0], [0.8, -0.1, 0.2], [0.3, 0.7, -0.2]])
    leader = optimize_leader_policy(
        small_problem,
        mle.omega,
        leader_reward,
        LeaderOptimizationConfig(PolicyOptimizationConfig(method="multistart", random_starts=4)),
        seed=4,
    )
    assert np.isclose(leader.policy.sum(), 1.0)
    assert np.isclose(leader.follower_policy.sum(), 1.0)
