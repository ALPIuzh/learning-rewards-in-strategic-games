"""Paper-aligned evaluation metrics for research experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.math import fisher_information
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)


@dataclass(frozen=True, slots=True)
class FisherTransferMetricConfig:
    """Numerical choices for the Stackelberg transfer error metric."""

    leader: LeaderOptimizationConfig
    negative_quadratic_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.negative_quadratic_tolerance < 0:
            raise ConfigurationError("negative_quadratic_tolerance must be nonnegative")


@dataclass(frozen=True, slots=True)
class StackelbergSuboptimalityConfig:
    """Numerical choices for last-iterate Stackelberg suboptimality."""

    leader: LeaderOptimizationConfig
    negative_gap_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.negative_gap_tolerance < 0:
            raise ConfigurationError("negative_gap_tolerance must be nonnegative")


class StackelbergSuboptimality:
    r"""Evaluate the true leader-value loss of policies optimized for an estimate."""

    def __init__(self, problem: CompiledProblem, true_omega: ArrayLike, leader_reward: ArrayLike,
            config: StackelbergSuboptimalityConfig) -> None:  # fmt: skip
        coefficients = np.asarray(true_omega, dtype=np.float64)
        if coefficients.shape != (problem.feature_dimension,):
            raise ConfigurationError("true_omega has the wrong shape for the compiled problem")
        if not np.all(np.isfinite(coefficients)):
            raise ConfigurationError("true_omega contains NaN or infinite values")
        self.problem = problem
        self.true_omega = coefficients.copy()
        self.leader_reward = np.asarray(leader_reward, dtype=np.float64)
        self.config = config
        self.true_value = optimize_leader_policy_batch(
            problem, self.true_omega[None, :], self.leader_reward, config.leader, (0,)
        )[0].value

    def evaluate_many(self, estimates: ArrayLike, *, seed: int = 0) -> FloatArray:
        """Return true-value gaps for every estimated coefficient vector."""

        coefficients = np.asarray(estimates, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.problem.feature_dimension:
            raise ConfigurationError("estimates must have shape (checkpoints, feature_dimension)")
        if not np.all(np.isfinite(coefficients)):
            raise ConfigurationError("estimates contain NaN or infinite values")
        policies = optimize_leader_policy_batch(
            self.problem,
            coefficients,
            self.leader_reward,
            self.config.leader,
            tuple(seed + index for index in range(coefficients.shape[0])),
        )
        deployed_values = np.asarray(
            [
                leader_value_and_gradient(self.problem, self.true_omega, self.leader_reward, solution.policy)[0]
                for solution in policies
            ]
        )
        gaps = self.true_value - deployed_values
        if np.min(gaps, initial=0.0) < -self.config.negative_gap_tolerance:
            raise ConvergenceError(f"estimated policy exceeded the computed true optimum by {-float(np.min(gaps)):.3e}")
        return np.maximum(gaps, 0.0)


class StackelbergFisherError:
    r"""Evaluate the estimator term in the Stackelberg reward-transfer lemma.

    This computes

    ``max_{mu in {mu*, mu_hat}} ||omega - omega_hat||_{I_mu(omega_hat)}``.

    It intentionally excludes the separate ``2 R_max chi(...)`` multiplier.
    """

    def __init__(self, problem: CompiledProblem, true_omega: ArrayLike,
            leader_reward: ArrayLike, config: FisherTransferMetricConfig) -> None:  # fmt: skip
        coefficients = np.asarray(true_omega, dtype=np.float64)
        if coefficients.shape != (problem.feature_dimension,):
            raise ConfigurationError("true_omega has the wrong shape for the compiled problem")
        if not np.all(np.isfinite(coefficients)):
            raise ConfigurationError("true_omega contains NaN or infinite values")

        self.problem = problem
        self.true_omega = coefficients.copy()
        self.leader_reward = np.asarray(leader_reward, dtype=np.float64)
        self.config = config
        # Batched with a single problem rather than the sequential optimizer: at large
        # leader action counts, sequential multistart runs full SLSQP on every one of
        # its (possibly vertex-screened) starts, while the batched path cheaply
        # prefilters and only exactly refines batch_refine_starts candidates. This is a
        # one-time __init__ cost, not a per-checkpoint one (see evaluate_many below),
        # but at GUARD's largest games it was the difference between seconds and hours.
        self.true_policy = optimize_leader_policy_batch(
            problem, self.true_omega[None, :], self.leader_reward, config.leader, (0,)
        )[0].policy

    def __call__(self, estimate: ArrayLike, *, seed: int = 0) -> float:
        coefficients = np.asarray(estimate, dtype=np.float64)
        if coefficients.shape != (self.problem.feature_dimension,):
            raise ConfigurationError("estimate has the wrong shape for the compiled problem")
        if not np.all(np.isfinite(coefficients)):
            raise ConfigurationError("estimate contains NaN or infinite values")

        estimated_policy = optimize_leader_policy(
            self.problem, coefficients, self.leader_reward, self.config.leader, seed=seed
        ).policy
        error = self.true_omega - coefficients
        return max(
            self._fisher_norm(error, coefficients, self.true_policy),
            self._fisher_norm(error, coefficients, estimated_policy),
        )

    def evaluate_many(self, estimates: ArrayLike, *, seed: int = 0) -> FloatArray:
        """Batched equivalent of calling ``self(estimate, seed=seed + index)`` per row.

        Uses :func:`optimize_leader_policy_batch` to amortize the leader's non-concave
        policy optimization across every checkpoint in one vectorized pass instead of
        solving each independently, which otherwise dominates evaluation cost for large
        leader action spaces.
        """

        coefficients = np.asarray(estimates, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.problem.feature_dimension:
            raise ConfigurationError("estimates must have shape (checkpoints, feature_dimension)")
        if not np.all(np.isfinite(coefficients)):
            raise ConfigurationError("estimates contain NaN or infinite values")

        n_checkpoints = coefficients.shape[0]
        seeds = tuple(seed + index for index in range(n_checkpoints))
        estimated_policies = optimize_leader_policy_batch(
            self.problem, coefficients, self.leader_reward, self.config.leader, seeds
        )
        errors = self.true_omega[None, :] - coefficients
        results = np.empty(n_checkpoints, dtype=np.float64)
        for index in range(n_checkpoints):
            results[index] = max(
                self._fisher_norm(errors[index], coefficients[index], self.true_policy),
                self._fisher_norm(errors[index], coefficients[index], estimated_policies[index].policy),
            )
        return results

    def _fisher_norm(self, error: FloatArray, estimate: FloatArray, policy: FloatArray) -> float:
        information = fisher_information(self.problem, policy, estimate)
        quadratic = float(error @ information @ error)
        if quadratic < -self.config.negative_quadratic_tolerance:
            raise ConvergenceError(f"Fisher error quadratic is unexpectedly negative: {quadratic:.3e}")
        return float(np.sqrt(max(0.0, quadratic)))
