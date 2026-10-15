"""Tests for the GuardApplication adapter and problem compilation."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.guard.application import GuardApplication, build_guard_problem, load_guard_game_data
from mairl.exceptions import ConfigurationError
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl_exp.config import ExperimentNumerics, WarmupNumerics


def test_load_guard_game_data_nw_chinatown() -> None:
    att_mat, def_mat, target_utils, _schedules = load_guard_game_data("nw_chinatown_sfg_isg_1")
    assert att_mat.shape == (23, 8)
    assert def_mat.shape == (23, 8)
    assert target_utils is not None
    assert target_utils.shape == (4, 8)


def test_build_guard_problem_target_coverage() -> None:
    problem, true_omega, leader_reward = build_guard_problem(
        "nw_chinatown_sfg_isg_1", feature_mode="target_coverage", eta=1.25
    )
    assert problem.n_leader_actions == 23
    assert problem.n_follower_actions == 8
    assert problem.feature_dimension == 16
    assert problem.identifiable_dimension == 15
    assert true_omega.shape == (16,)
    assert leader_reward.shape == (23, 8)


def test_build_guard_problem_behavioral_chinatown() -> None:
    problem, true_omega, leader_reward = build_guard_problem(
        "nw_chinatown_sfg_isg_1", feature_mode="behavioral", eta=1.25
    )
    assert problem.n_leader_actions == 23
    assert problem.n_follower_actions == 8
    assert problem.feature_dimension == 3
    assert problem.identifiable_dimension == 3
    assert leader_reward.shape == (23, 8)
    np.testing.assert_allclose(true_omega, [0.0, 1.0, -2.0 / 3.0], atol=1e-8)
    assert problem.max_feature_norm <= 1.0 + 1e-12


def test_build_guard_problem_behavioral_etosha() -> None:
    problem, true_omega, leader_reward = build_guard_problem("etosha_sfg_gsg_1", feature_mode="behavioral", eta=1.25)
    assert problem.n_leader_actions == 169
    assert problem.n_follower_actions == 7
    assert problem.feature_dimension == 3
    assert problem.identifiable_dimension == 3
    assert leader_reward.shape == (169, 7)
    np.testing.assert_allclose(true_omega, [0.0, 1.0, -0.8], atol=1e-8)


def test_build_guard_problem_behavioral_reconstructs_attacker_payoffs() -> None:
    """The 3-dim behavioral feature map must reproduce GUARD's own payoffs exactly.

    Features are unit-normalized (phi_used = phi_raw / R) with eta rescaled to eta*R
    (footnote 2 of the paper), so undoing that rescaling on phi_used . true_omega must
    recover GUARD's original attacker_utility_matrix exactly.
    """

    input_eta = 1.25
    for game in ("nw_chinatown_sfg_isg_1", "etosha_sfg_gsg_1", "lobeke_sfg_gsg_1"):
        problem, true_omega, _ = build_guard_problem(game, feature_mode="behavioral", eta=input_eta)
        norm_bound = problem.eta / input_eta
        reconstructed_raw = norm_bound * (problem.features @ true_omega)
        attacker_mat, _, _, _ = load_guard_game_data(game)
        np.testing.assert_allclose(reconstructed_raw, attacker_mat, atol=1e-8)


def test_build_guard_problem_behavioral_requires_target_utilities() -> None:
    with pytest.raises(ConfigurationError):
        build_guard_problem("etosha_nfg_gsg_1", feature_mode="behavioral", eta=1.25)


def test_build_guard_problem_tabular() -> None:
    problem, true_omega, _leader_reward = build_guard_problem(
        "nw_chinatown_sfg_isg_1", feature_mode="tabular", eta=1.25
    )
    assert problem.n_leader_actions == 23
    assert problem.n_follower_actions == 8
    assert problem.feature_dimension == 23 * 8
    assert true_omega.shape == (23 * 8,)


def test_guard_application_build_and_evaluate() -> None:
    app = GuardApplication(game_name="nw_chinatown_sfg_isg_1", feature_mode="target_coverage")
    numerics = ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupNumerics(),
        active_policy=PolicyOptimizationConfig(),
        evaluation_policy=PolicyOptimizationConfig(),
    )
    runtime = app.build(numerics)

    assert runtime.name == "guard_nw_chinatown_sfg_isg_1"
    assert runtime.problem.n_leader_actions == 23

    oracle = runtime.oracle_factory(42)
    follower_action = oracle.sample(np.ones(23) / 23.0, np.random.default_rng(42))
    assert 0 <= follower_action < 8

    # Evaluate mock estimates
    estimates = np.tile(runtime.problem.features[0, 0, :], (2, 1))
    metrics = runtime.evaluate(estimates, seed=42)

    assert "stackelberg_suboptimality" in metrics
    assert "coefficient_l2_error" in metrics
    assert metrics["stackelberg_suboptimality"].shape == (2,)
    assert np.all(metrics["stackelberg_suboptimality"] >= 0.0)
