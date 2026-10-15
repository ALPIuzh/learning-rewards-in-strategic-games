"""Validation of entropy-regularized zero-sum Nash equilibria."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root

from mairl.core.math import stable_softmax, validate_policy
from mairl.core.problem import FloatArray, _readonly
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.zero_sum.problem import CompiledZeroSumGame


@dataclass(frozen=True, slots=True)
class QREValidationConfig:
    """Numerical tolerances for validating a supplied expert equilibrium."""

    equilibrium_tolerance: float = 1e-9
    parameter_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.equilibrium_tolerance <= 0 or self.parameter_tolerance < 0:
            raise ConfigurationError("QRE validation tolerances must be positive/nonnegative")


@dataclass(frozen=True, slots=True)
class ValidatedQRE:
    """A simulation-only certificate that a policy pair is the game's true QRE."""

    game: CompiledZeroSumGame
    max_policy: FloatArray
    min_policy: FloatArray
    max_residual: float
    min_residual: float

    def validate_for(self, game: CompiledZeroSumGame) -> None:
        if self.game is not game:
            raise ConfigurationError("validated QRE belongs to a different compiled game")


def validate_qre(game: CompiledZeroSumGame, true_theta: ArrayLike, max_policy: ArrayLike, min_policy: ArrayLike,
        config: QREValidationConfig | None = None) -> ValidatedQRE:  # fmt: skip
    """Return a certificate only when the supplied pair satisfies both QRE equations."""

    rules = config or QREValidationConfig()
    theta = np.asarray(true_theta, dtype=np.float64)
    if theta.shape != (game.feature_dimension,) or not np.all(np.isfinite(theta)):
        raise ConfigurationError(f"true_theta must be finite with shape ({game.feature_dimension},)")
    theta_norm = float(np.linalg.norm(theta))
    if theta_norm > game.radius + rules.parameter_tolerance:
        raise ConfigurationError(f"true_theta norm {theta_norm:.6g} exceeds radius {game.radius:.6g}")
    mu = validate_policy(max_policy, game.n_max_actions)
    nu = validate_policy(min_policy, game.n_min_actions)
    if np.any(mu <= 0) or np.any(nu <= 0):
        raise ConfigurationError("an entropy-regularized QRE must have full support")

    leader_payoff = np.einsum("abd,d->ab", game.max_player_features, theta, optimize=True)
    follower_payoff = np.einsum("abd,d->ab", game.min_player_features, theta, optimize=True)
    expected_mu = stable_softmax(game.eta * (leader_payoff @ nu))
    expected_nu = stable_softmax(game.eta * (follower_payoff.T @ mu))
    max_residual = float(np.max(np.abs(mu - expected_mu)))
    min_residual = float(np.max(np.abs(nu - expected_nu)))
    if max(max_residual, min_residual) > rules.equilibrium_tolerance:
        raise ConfigurationError(
            "provided policies are not an entropy-regularized zero-sum Nash equilibrium: "
            f"max-player residual={max_residual:.3e}, "
            f"min-player residual={min_residual:.3e}, "
            f"tolerance={rules.equilibrium_tolerance:.3e}"
        )
    return ValidatedQRE(
        game=game,
        max_policy=_readonly(mu),
        min_policy=_readonly(nu),
        max_residual=max_residual,
        min_residual=min_residual,
    )


@dataclass(frozen=True, slots=True)
class QRESolveConfig:
    """Numerical tolerances for the joint QRE fixed-point solve."""

    root_tolerance: float = 1e-10
    max_evaluations: int = 10_000

    def __post_init__(self) -> None:
        if self.root_tolerance <= 0:
            raise ConfigurationError("root_tolerance must be strictly positive")
        if self.max_evaluations < 1:
            raise ConfigurationError("max_evaluations must be positive")


def solve_qre(game: CompiledZeroSumGame, true_theta: ArrayLike,
        config: QRESolveConfig | None = None) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Find the mutual entropy-regularized best-response fixed point under ``true_theta``.

    Unlike a Stackelberg solve, both players' policies are simultaneously softmax
    best responses to each other -- the joint equilibrium that Liao's zero-sum OLS
    estimator requires as its sampling expert.
    """

    rules = config or QRESolveConfig()
    theta = np.asarray(true_theta, dtype=np.float64)
    if theta.shape != (game.feature_dimension,) or not np.all(np.isfinite(theta)):
        raise ConfigurationError(f"true_theta must be finite with shape ({game.feature_dimension},)")

    leader_payoff = np.einsum("abd,d->ab", game.max_player_features, theta, optimize=True)
    follower_payoff = np.einsum("abd,d->ab", game.min_player_features, theta, optimize=True)
    n_leader = game.n_max_actions

    def policies(log_ratios: FloatArray) -> tuple[FloatArray, FloatArray]:
        leader = stable_softmax(np.concatenate((log_ratios[: n_leader - 1], np.zeros(1))))
        follower = stable_softmax(np.concatenate((log_ratios[n_leader - 1 :], np.zeros(1))))
        return leader, follower

    def residual(log_ratios: FloatArray) -> FloatArray:
        leader, follower = policies(log_ratios)
        leader_values = leader_payoff @ follower
        follower_values = follower_payoff.T @ leader
        return np.concatenate(
            (
                log_ratios[: n_leader - 1] - game.eta * (leader_values[:-1] - leader_values[-1]),
                log_ratios[n_leader - 1 :] - game.eta * (follower_values[:-1] - follower_values[-1]),
            )
        )

    dimension = game.n_max_actions + game.n_min_actions - 2
    solution = root(
        residual,
        np.zeros(dimension, dtype=np.float64),
        method="hybr",
        options={"xtol": rules.root_tolerance, "maxfev": rules.max_evaluations},
    )
    residual_norm = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or residual_norm > rules.root_tolerance:
        raise ConvergenceError(
            "zero-sum QRE fixed-point solve failed: "
            f"success={solution.success}, residual={residual_norm:.3e}, "
            f"message={solution.message}"
        )
    return policies(solution.x)
