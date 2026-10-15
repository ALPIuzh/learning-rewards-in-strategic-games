from __future__ import annotations

import numpy as np
import pytest

from mairl import (
    CompilationError,
    ConfigurationError,
    EstimateOutsideBallWarning,
    FixedLeaderOLSConfig,
    FrequencySmoothingWarning,
    QREValidationConfig,
    compile_problem,
    compile_zero_sum_game,
    fit_fixed_leader_ols,
    fit_liao_zero_sum_ols,
    fit_liao_zero_sum_paired_actions,
    solve_fixed_leader_ols,
    solve_liao_zero_sum_ols,
    validate_follower_best_response,
    validate_qre,
)
from mairl.core.math import follower_probabilities, stable_softmax
from mairl.inference.fixed_leader_ols import _ball_constrained_solution


def _config(**overrides) -> FixedLeaderOLSConfig:
    values = {
        "pseudocount": 0.5,
        "smoothing_warning_ratio": 0.1,
        "identifiability": "require_unique",
        "constraint": "ball",
    }
    values.update(overrides)
    return FixedLeaderOLSConfig(**values)


def _general_problem(*, radius: float = 2.0):
    follower_action_features = np.asarray([[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]])
    features = np.broadcast_to(follower_action_features, (3, 3, 2)).copy()
    problem = compile_problem(features, eta=1.4, radius=radius)
    leader_policy = np.asarray([0.7, 0.2, 0.1])
    return problem, leader_policy


def _zero_sum_game(*, radius: float = 2.0):
    row_features = np.asarray([-0.4, 0.1, 0.45])
    column_features = np.asarray([-0.35, 0.2, 0.4])
    leader_features = np.empty((3, 3, 2), dtype=np.float64)
    leader_features[:, :, 0] = row_features[:, None]
    leader_features[:, :, 1] = column_features[None, :]
    game = compile_zero_sum_game(leader_features, -leader_features, eta=1.4, radius=radius)
    return game, row_features, column_features


def _qre(row_features, column_features, theta, eta):
    leader_policy = stable_softmax(eta * theta[0] * row_features)
    follower_policy = stable_softmax(-eta * theta[1] * column_features)
    return leader_policy, follower_policy


def test_general_sum_fixed_leader_recovers_follower_reward() -> None:
    problem, leader_policy = _general_problem()
    omega = np.asarray([0.8, -0.6])
    follower_policy = follower_probabilities(problem, leader_policy, omega)
    response = validate_follower_best_response(problem, omega, leader_policy, follower_policy)

    result = solve_fixed_leader_ols(problem, response, follower_policy, _config())

    np.testing.assert_allclose(result.omega, omega, atol=1e-12)
    assert result.equation_mode == "follower_only"
    assert result.design_matrix.shape == (problem.n_follower_actions - 1, problem.feature_dimension)
    assert result.identifiable
    assert result.residual_norm < 1e-12


def test_general_mode_refuses_a_policy_that_is_not_the_fixed_leader_best_response() -> None:
    problem, leader_policy = _general_problem()
    omega = np.asarray([0.8, -0.6])
    follower_policy = follower_probabilities(problem, leader_policy, omega)
    invalid = follower_policy.copy()
    invalid[0] += 0.02
    invalid /= np.sum(invalid)

    with pytest.raises(ConfigurationError, match="not the entropy-regularized best response"):
        validate_follower_best_response(problem, omega, leader_policy, invalid)


def test_general_mode_counts_only_follower_samples() -> None:
    problem, leader_policy = _general_problem()
    omega = np.asarray([0.8, -0.6])
    follower_policy = follower_probabilities(problem, leader_policy, omega)
    response = validate_follower_best_response(problem, omega, leader_policy, follower_policy)
    rng = np.random.default_rng(5)
    actions = rng.choice(problem.n_follower_actions, size=100_000, p=follower_policy)

    result = fit_fixed_leader_ols(problem, response, actions, _config())

    assert result.follower_sample_count == actions.size
    assert result.leader_frequency_diagnostics is None
    assert result.follower_frequency_diagnostics is not None
    np.testing.assert_allclose(result.leader_policy, leader_policy)
    np.testing.assert_allclose(result.omega, omega, atol=0.02)


def test_general_mode_warns_when_smoothing_is_material() -> None:
    problem, leader_policy = _general_problem(radius=10.0)
    omega = np.asarray([0.8, -0.6])
    follower_policy = follower_probabilities(problem, leader_policy, omega)
    response = validate_follower_best_response(problem, omega, leader_policy, follower_policy)

    with pytest.warns(FrequencySmoothingWarning, match="unobserved actions"):
        result = fit_fixed_leader_ols(
            problem, response, np.zeros(20, dtype=np.int64), _config(pseudocount=0.5, smoothing_warning_ratio=0.01)
        )

    assert result.follower_frequency_diagnostics is not None
    assert result.follower_frequency_diagnostics.zero_count_actions == 2
    assert np.all(result.follower_policy > 0)


def test_best_response_certificate_is_problem_specific() -> None:
    problem, leader_policy = _general_problem()
    omega = np.asarray([0.8, -0.6])
    follower_policy = follower_probabilities(problem, leader_policy, omega)
    response = validate_follower_best_response(problem, omega, leader_policy, follower_policy)
    other_problem, _ = _general_problem()

    with pytest.raises(ConfigurationError, match="different compiled problem"):
        solve_fixed_leader_ols(other_problem, response, follower_policy, _config())


def test_zero_sum_compilation_rejects_general_sum_features() -> None:
    game, _, _ = _zero_sum_game()
    invalid_follower_features = game.min_player_features.copy()
    invalid_follower_features[0, 0, 0] += 0.01
    with pytest.raises(CompilationError, match="not zero-sum"):
        compile_zero_sum_game(game.max_player_features, invalid_follower_features, eta=game.eta, radius=game.radius)


def test_zero_sum_mode_adds_leader_equations_and_restores_identification() -> None:
    game, rows, columns = _zero_sum_game()
    theta = np.asarray([0.8, -0.6])
    leader_policy, follower_policy = _qre(rows, columns, theta, game.eta)
    response = validate_follower_best_response(game.problem, theta, leader_policy, follower_policy)
    expert = validate_qre(game, theta, leader_policy, follower_policy)

    with pytest.raises(ConfigurationError, match="not strongly identifiable"):
        solve_fixed_leader_ols(game.problem, response, follower_policy, _config())
    follower_only = solve_fixed_leader_ols(
        game.problem, response, follower_policy, _config(identifiability="minimum_norm")
    )
    zero_sum = solve_liao_zero_sum_ols(game, expert, leader_policy, follower_policy, _config())

    assert follower_only.rank == 1
    assert not follower_only.identifiable
    np.testing.assert_allclose(follower_only.omega, [0.0, theta[1]], atol=1e-12)
    assert zero_sum.equation_mode == "zero_sum"
    assert zero_sum.design_matrix.shape[0] == (game.n_max_actions + game.n_min_actions - 2)
    np.testing.assert_allclose(zero_sum.omega, theta, atol=1e-12)
    assert zero_sum.identifiable


def test_zero_sum_mode_refuses_a_non_qre_policy_pair() -> None:
    game, rows, columns = _zero_sum_game()
    theta = np.asarray([0.8, -0.6])
    leader_policy, follower_policy = _qre(rows, columns, theta, game.eta)
    invalid_leader = leader_policy.copy()
    invalid_leader[0] += 0.02
    invalid_leader /= np.sum(invalid_leader)

    with pytest.raises(ConfigurationError, match="not an entropy-regularized"):
        validate_qre(game, theta, invalid_leader, follower_policy, QREValidationConfig(equilibrium_tolerance=1e-6))


def test_zero_sum_fixed_expert_and_paired_protocols_count_followers() -> None:
    game, rows, columns = _zero_sum_game()
    theta = np.asarray([0.8, -0.6])
    leader_policy, follower_policy = _qre(rows, columns, theta, game.eta)
    expert = validate_qre(game, theta, leader_policy, follower_policy)
    rng = np.random.default_rng(8)
    leader_actions = rng.choice(game.n_max_actions, size=100_000, p=leader_policy)
    follower_actions = rng.choice(game.n_min_actions, size=100_000, p=follower_policy)

    fixed = fit_liao_zero_sum_ols(game, expert, follower_actions, _config())
    paired = fit_liao_zero_sum_paired_actions(game, expert, leader_actions, follower_actions, _config())

    assert fixed.follower_sample_count == follower_actions.size
    assert fixed.leader_frequency_diagnostics is None
    assert paired.follower_sample_count == follower_actions.size
    assert paired.leader_frequency_diagnostics is not None
    np.testing.assert_allclose(fixed.omega, theta, atol=0.02)
    np.testing.assert_allclose(paired.omega, theta, atol=0.025)


def test_ball_constraint_warns_and_solves_constrained_least_squares() -> None:
    problem, leader_policy = _general_problem(radius=0.5)
    true_omega = np.asarray([0.2, -0.1])
    true_follower = follower_probabilities(problem, leader_policy, true_omega)
    response = validate_follower_best_response(problem, true_omega, leader_policy, true_follower)
    outside_omega = np.asarray([2.0, -1.5])
    estimated_follower = follower_probabilities(problem, leader_policy, outside_omega)

    with pytest.warns(EstimateOutsideBallWarning):
        unconstrained = solve_fixed_leader_ols(
            problem, response, estimated_follower, _config(constraint="unconstrained")
        )
    assert np.linalg.norm(unconstrained.omega) > problem.radius
    assert not unconstrained.constrained

    with pytest.warns(EstimateOutsideBallWarning):
        constrained = solve_fixed_leader_ols(problem, response, estimated_follower, _config(constraint="ball"))
    assert constrained.constrained
    assert np.linalg.norm(constrained.omega) == pytest.approx(problem.radius, abs=1e-9)
    assert constrained.residual_norm >= constrained.unconstrained_residual_norm


def test_ball_constraint_refines_a_near_zero_ill_conditioned_multiplier() -> None:
    matrix = np.asarray([[1.0e-6]])
    target = np.asarray([2.0e-3])
    config = _config(ball_tolerance=1.0e-6, ball_root_tolerance=1.0e-12)

    constrained = _ball_constrained_solution(matrix, target, radius=2.0, config=config)

    assert np.linalg.norm(constrained) == pytest.approx(2.0, abs=1.0e-10)
