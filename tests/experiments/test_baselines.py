from __future__ import annotations

from dataclasses import replace

import numpy as np

from mairl.core.math import fisher_information
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.exploration.design import WarmupConfig
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.evaluation.metrics import FisherTransferMetricConfig, StackelbergFisherError
from mairl_exp.studies.baselines import RandomBaselineConfig, RandomExplorationBaseline


def _baseline_config(method: str, *, warmup: bool) -> RandomBaselineConfig:
    return RandomBaselineConfig(
        total_samples=35,
        method=method,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=30, design_tolerance=0.2, enabled=warmup),
        seed=12,
    )


def test_random_baselines_use_uniform_simplex_policies_and_exact_budget(small_problem, small_true_omega) -> None:
    true_omega = small_true_omega
    fixed = RandomExplorationBaseline(small_problem, _baseline_config("fixed_random", warmup=False)).run(
        SoftmaxFollowerOracle(small_problem, true_omega)
    )
    changing = RandomExplorationBaseline(small_problem, _baseline_config("random_each_round", warmup=False)).run(
        SoftmaxFollowerOracle(small_problem, true_omega)
    )

    assert fixed.dataset.size == changing.dataset.size == 35
    assert np.allclose(np.sum(fixed.dataset.policies, axis=1), 1.0)
    assert np.all(fixed.dataset.policies >= 0)
    assert np.allclose(fixed.dataset.policies, fixed.dataset.policies[0])
    assert not np.allclose(changing.dataset.policies, changing.dataset.policies[0])
    assert np.array_equal(fixed.sample_counts, np.arange(36))
    assert np.array_equal(changing.sample_counts, np.arange(36))


def test_warmup_toggle_is_shared_and_preserves_total_sample_fairness(small_problem, small_true_omega) -> None:
    true_omega = small_true_omega
    fixed_config = _baseline_config("fixed_random", warmup=True)
    random_config = replace(fixed_config, method="random_each_round")
    fixed = RandomExplorationBaseline(small_problem, fixed_config).run(SoftmaxFollowerOracle(small_problem, true_omega))
    changing = RandomExplorationBaseline(small_problem, random_config).run(
        SoftmaxFollowerOracle(small_problem, true_omega)
    )

    warmup_size = fixed_config.warmup.effective_budget
    assert fixed.dataset.size == changing.dataset.size == fixed_config.total_samples
    assert np.array_equal(fixed.dataset.policies[:warmup_size], changing.dataset.policies[:warmup_size])
    assert np.array_equal(fixed.dataset.follower_actions[:warmup_size], changing.dataset.follower_actions[:warmup_size])
    assert np.allclose(fixed.dataset.policies[warmup_size:], fixed.dataset.policies[warmup_size])
    assert fixed.sample_counts[0] == changing.sample_counts[0] == warmup_size
    assert fixed.sample_counts[-1] == changing.sample_counts[-1] == 35


def test_warmup_toggle_preserves_online_common_random_numbers(small_problem, small_true_omega) -> None:
    """Warm-up must not shift the later random-policy or follower streams."""

    true_omega = small_true_omega
    for method in ("fixed_random", "random_each_round"):
        warm_config = _baseline_config(method, warmup=True)
        cold_config = replace(warm_config, warmup=replace(warm_config.warmup, enabled=False))
        warm = RandomExplorationBaseline(small_problem, warm_config).run(
            SoftmaxFollowerOracle(small_problem, true_omega)
        )
        cold = RandomExplorationBaseline(small_problem, cold_config).run(
            SoftmaxFollowerOracle(small_problem, true_omega)
        )
        first_online = warm_config.warmup.effective_budget

        np.testing.assert_allclose(warm.dataset.policies[first_online:], cold.dataset.policies[first_online:], atol=0.0)
        np.testing.assert_array_equal(
            warm.dataset.follower_actions[first_online:], cold.dataset.follower_actions[first_online:]
        )


def test_stackelberg_fisher_error_matches_the_lemma_term(small_problem, small_true_omega, small_leader_reward) -> None:
    true_omega = small_true_omega
    estimate = np.array([0.2, -0.3])
    leader_reward = small_leader_reward
    leader_config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="grid", grid_resolution=20, grid_max_points=1_000, refine_grid_solution=False)
    )
    metric = StackelbergFisherError(
        small_problem, true_omega, leader_reward, FisherTransferMetricConfig(leader=leader_config)
    )

    computed = metric(estimate)
    estimated_policy = optimize_leader_policy(small_problem, estimate, leader_reward, leader_config).policy
    error = true_omega - estimate
    expected = max(
        np.sqrt(error @ fisher_information(small_problem, metric.true_policy, estimate) @ error),
        np.sqrt(error @ fisher_information(small_problem, estimated_policy, estimate) @ error),
    )

    assert np.isclose(computed, expected, atol=1e-14)
