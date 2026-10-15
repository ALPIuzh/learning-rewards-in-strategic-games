"""Zero-sum augmentation of fixed-leader logit OLS."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.problem import FloatArray, _readonly
from mairl.exceptions import ConfigurationError
from mairl.inference.fixed_leader_ols import (
    FixedLeaderOLSConfig,
    FixedLeaderOLSResult,
    FrequencyDiagnostics,
    _build_follower_system,
    _frequency_policy,
    _solve_ols_system,
    _validate_actions,
)
from mairl.zero_sum.equilibrium import ValidatedQRE
from mairl.zero_sum.problem import CompiledZeroSumGame


def _build_zero_sum_system(game: CompiledZeroSumGame, leader_policy: ArrayLike, follower_policy: ArrayLike,
        config: FixedLeaderOLSConfig) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:  # fmt: skip
    leader, follower, follower_matrix, follower_target = _build_follower_system(
        game.problem, leader_policy, follower_policy, config
    )
    if np.any(leader <= 0):
        raise ConfigurationError("zero-sum QRE log-ratio equations require a full-support leader policy")
    reference = config.leader_reference_action
    if reference >= game.n_max_actions:
        raise ConfigurationError("leader_reference_action is outside the leader action space")

    leader_actions = np.delete(np.arange(game.n_max_actions), reference)
    differences = game.max_player_features[leader_actions] - game.max_player_features[reference][None, :, :]
    leader_matrix = np.einsum("abd,b->ad", differences, follower, optimize=True)
    leader_target = np.log(leader[leader_actions] / leader[reference]) / game.eta
    return (
        leader,
        follower,
        _readonly(np.concatenate((leader_matrix, follower_matrix), axis=0)),
        _readonly(np.concatenate((leader_target, follower_target))),
    )


def solve_liao_zero_sum_ols(game: CompiledZeroSumGame, expert: ValidatedQRE,
        leader_policy: ArrayLike, follower_policy: ArrayLike, config: FixedLeaderOLSConfig,
        *, follower_sample_count: int = 0, leader_frequency_diagnostics: FrequencyDiagnostics | None = None,
        follower_frequency_diagnostics: FrequencyDiagnostics | None = None) -> FixedLeaderOLSResult:  # fmt: skip
    """Solve both players' QRE equations after zero-sum validation."""

    expert.validate_for(game)
    leader, follower, matrix, target = _build_zero_sum_system(game, leader_policy, follower_policy, config)
    return _solve_ols_system(
        game.problem,
        leader,
        follower,
        matrix,
        target,
        config,
        equation_mode="zero_sum",
        follower_sample_count=follower_sample_count,
        leader_frequency_diagnostics=leader_frequency_diagnostics,
        follower_frequency_diagnostics=follower_frequency_diagnostics,
    )


def fit_liao_zero_sum_ols(game: CompiledZeroSumGame, expert: ValidatedQRE, follower_actions: ArrayLike,
        config: FixedLeaderOLSConfig) -> FixedLeaderOLSResult:  # fmt: skip
    """Use both QRE equation blocks with a fixed, known zero-sum expert leader."""

    expert.validate_for(game)
    follower, diagnostics = _frequency_policy(follower_actions, game.n_min_actions, config, name="follower_actions")
    return solve_liao_zero_sum_ols(
        game,
        expert,
        expert.max_policy,
        follower,
        config,
        follower_sample_count=diagnostics.sample_count,
        follower_frequency_diagnostics=diagnostics,
    )


def fit_liao_zero_sum_paired_actions(game: CompiledZeroSumGame, expert: ValidatedQRE, leader_actions: ArrayLike,
        follower_actions: ArrayLike, config: FixedLeaderOLSConfig) -> FixedLeaderOLSResult:  # fmt: skip
    """Estimate both QRE marginals as in the paper while counting follower samples."""

    expert.validate_for(game)
    leader_indices = _validate_actions(leader_actions, game.n_max_actions, "leader_actions")
    follower_indices = _validate_actions(follower_actions, game.n_min_actions, "follower_actions")
    if leader_indices.size != follower_indices.size:
        raise ConfigurationError("paired expert action arrays must have the same length")
    leader, leader_diagnostics = _frequency_policy(leader_indices, game.n_max_actions, config, name="leader_actions")
    follower, follower_diagnostics = _frequency_policy(
        follower_indices, game.n_min_actions, config, name="follower_actions"
    )
    return solve_liao_zero_sum_ols(
        game,
        expert,
        leader,
        follower,
        config,
        follower_sample_count=int(follower_indices.size),
        leader_frequency_diagnostics=leader_diagnostics,
        follower_frequency_diagnostics=follower_diagnostics,
    )
