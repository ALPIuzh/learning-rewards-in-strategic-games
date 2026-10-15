from __future__ import annotations

import numpy as np
import pytest

import mairl.algorithm as algorithm_module
from mairl.algorithm import ActiveIRL, ActiveIRLConfig
from mairl.core.problem import compile_problem
from mairl.exploration.design import WarmupConfig
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig


@pytest.fixture
def captured_information(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Record the information matrix handed to the policy selector each round."""

    captured: list[np.ndarray] = []
    original_select = algorithm_module.select_information_policy_batch

    def capture_information(problem, omega, information, config, rng):
        captured.append(information[0].copy())
        return original_select(problem, omega, information, config, rng)

    monkeypatch.setattr(algorithm_module, "select_information_policy_batch", capture_information)
    return captured


def test_active_irl_retains_warmup_and_returns_last_mle(small_problem, small_oracle) -> None:
    config = ActiveIRLConfig(
        horizon=6,
        lambda_information=small_problem.eta**2,
        lambda_warm=0.01,
        confidence_delta=0.05,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=60, design_tolerance=0.2),
        policy=PolicyOptimizationConfig(random_starts=3),
        seed=9,
    )
    oracle = small_oracle
    result = ActiveIRL(small_problem, config).run(oracle)

    assert result.status == "completed"
    assert result.dataset.size == config.warmup.budget + config.horizon
    assert result.omega_history.shape == (config.horizon + 1, 2)
    assert np.array_equal(result.omega, result.omega_history[-1])
    assert np.array_equal(result.omega, result.final_mle.omega)
    assert len(result.diagnostics) == config.horizon
    assert all(item.total_samples >= config.warmup.budget for item in result.diagnostics)


def test_degenerate_problem_returns_zero_without_sampling() -> None:
    features = np.broadcast_to(np.array([[[0.2, -0.1]]]), (2, 3, 2)).copy()
    problem = compile_problem(features, eta=1.0, radius=1.0)
    config = ActiveIRLConfig(
        horizon=10,
        lambda_information=1.0,
        lambda_warm=0.1,
        confidence_delta=0.05,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=0, design_tolerance=0.2),
    )

    class FailingOracle:
        def sample(self, leader_policy, rng):
            raise AssertionError("degenerate problems must not sample")

    result = ActiveIRL(problem, config).run(FailingOracle())
    assert result.degenerate
    assert result.status == "degenerate_no_identifiable_reward_directions"
    assert np.array_equal(result.omega, np.zeros(2))
    assert result.dataset.size == 0


def test_active_irl_can_disable_warmup_without_changing_its_configured_budget(small_problem, small_oracle) -> None:
    config = ActiveIRLConfig(
        horizon=5,
        lambda_information=small_problem.eta**2,
        lambda_warm=0.01,
        confidence_delta=0.05,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=60, design_tolerance=0.2, enabled=False),
        policy=PolicyOptimizationConfig(random_starts=1),
        seed=4,
    )

    result = ActiveIRL(small_problem, config).run(small_oracle)

    assert config.warmup.budget == 60
    assert config.warmup.effective_budget == 0
    assert result.warmup_design.support_size == 0
    assert result.dataset.size == config.horizon
    assert result.omega_history.shape == (config.horizon + 1, 2)
    assert result.diagnostics[0].total_samples == 1


def test_active_irl_accepts_positive_information_regularization_below_eta_squared(
    small_problem, captured_information, small_oracle
) -> None:
    lambda_information = 0.1 * small_problem.eta**2
    config = ActiveIRLConfig(
        horizon=1,
        lambda_information=lambda_information,
        lambda_warm=0.01,
        confidence_delta=0.05,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=0, design_tolerance=0.2, enabled=False),
        policy=PolicyOptimizationConfig(random_starts=1),
        seed=17,
    )

    ActiveIRL(small_problem, config).run(small_oracle)

    assert len(captured_information) == 1
    assert np.allclose(
        captured_information[0], lambda_information * np.eye(small_problem.feature_dimension), atol=1e-12
    )


def test_lambda_warm_does_not_regularize_online_information_without_warmup(
    small_problem, captured_information, small_oracle
) -> None:

    def run(lambda_warm: float):
        config = ActiveIRLConfig(
            horizon=5,
            lambda_information=small_problem.eta**2,
            lambda_warm=lambda_warm,
            confidence_delta=0.05,
            mle=MLEConfig(ridge=1e-4),
            warmup=WarmupConfig(budget=60, design_tolerance=0.2, enabled=False),
            policy=PolicyOptimizationConfig(random_starts=1),
            seed=12,
        )
        oracle = small_oracle
        return ActiveIRL(small_problem, config).run(oracle)

    small = run(1e-4)
    small_information = tuple(captured_information)
    captured_information.clear()
    large = run(10.0)
    large_information = tuple(captured_information)

    assert len(small_information) == len(large_information) == 5
    for small_matrix, large_matrix in zip(small_information, large_information, strict=True):
        assert np.allclose(small_matrix, large_matrix, atol=1e-12)
    assert np.array_equal(small.dataset.follower_actions, large.dataset.follower_actions)
    assert np.allclose(small.dataset.policies, large.dataset.policies, atol=1e-12)
    assert np.allclose(small.omega_history, large.omega_history, atol=1e-12)
