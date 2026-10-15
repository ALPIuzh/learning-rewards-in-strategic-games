"""Zero-sum flagship application: the Tier-3 Lobeke anti-poaching game.

Compares Active IRL against Liao's zero-sum OLS, the follower-only ("unaware of
zero-sum structure") OLS baseline, and the Stackelberg random baselines, all on
the same GUARD-generated Lobeke game used for ``geographic/map_lobeke.py``.
Evaluation reuses :class:`mairl_exp.evaluation.metrics.StackelbergFisherError` and the same
``stackelberg_suboptimality``/``fisher_transfer_error``/``coefficient_l2_error``
metric vocabulary as :mod:`experiments.guard.application`'s ``GuardApplication``,
so every method in the comparison -- OLS-based or exploration-based -- is scored
against the leader's actual Stackelberg deployment value, not the joint QRE value
that only the OLS estimators' sampling scheme relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.guard.geographic.map_lobeke import build_game, build_lobeke_zero_sum_game
from experiments.guard.paper.rewards import leader_reward_normalization_metadata, unit_interval_leader_reward
from experiments.zero_sum_comparison.application import ZeroSumApplicationRuntime
from mairl import QRESolveConfig, QREValidationConfig, solve_qre, validate_follower_best_response, validate_qre
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy_batch
from mairl_exp.config import ExperimentNumerics
from mairl_exp.evaluation.metrics import FisherTransferMetricConfig, StackelbergFisherError

NEGATIVE_TARGET_SUBOPTIMALITY_TOLERANCE = 1.0e-9


def _validated_stackelberg_suboptimality(optimum_value: float, deployed_values: FloatArray) -> FloatArray:
    """Clip numerical noise only after validating the computed source optimum."""

    raw = optimum_value - np.asarray(deployed_values, dtype=np.float64)
    minimum = float(np.min(raw))
    if minimum < -NEGATIVE_TARGET_SUBOPTIMALITY_TOLERANCE:
        raise ConvergenceError(f"estimated deployment exceeded the computed source optimum by {-minimum:.3e}")
    return np.maximum(0.0, raw)


def _self_concordance_distances(features: FloatArray, coefficient_errors: FloatArray) -> FloatArray:
    """Evaluate the simulation-only ``max_mu d_mu`` certificate quantity."""

    utility_errors = np.einsum(
        "nd,abd->nab",
        np.asarray(coefficient_errors, dtype=np.float64),
        np.asarray(features, dtype=np.float64),
        optimize=True,
    )
    return np.max(np.ptp(utility_errors, axis=2), axis=1)


@dataclass(frozen=True, slots=True)
class LobekeZeroSumApplication:
    """Build the Tier-3 Lobeke anti-poaching game as a zero-sum MAIRL application."""

    num_rows: int = 5
    num_columns: int = 5
    num_timesteps: int = 10
    eta: float = 1.25
    radius_multiplier: float = 1.5
    behavioral_fit_tolerance: float = 1e-6
    qre_tolerance: float = 1e-10
    qre_max_evaluations: int = 10_000

    def __post_init__(self) -> None:
        if self.qre_tolerance <= 0:
            raise ConfigurationError("qre_tolerance must be strictly positive")
        if self.qre_max_evaluations < 1:
            raise ConfigurationError("qre_max_evaluations must be positive")

    def build(self, numerics: object) -> ZeroSumApplicationRuntime:
        """Compile the Lobeke zero-sum game, solve its joint QRE expert, and wire evaluation."""

        if not isinstance(numerics, ExperimentNumerics):
            raise TypeError("LobekeZeroSumApplication requires ExperimentNumerics")

        gsg = build_game(num_rows=self.num_rows, num_columns=self.num_columns, num_timesteps=self.num_timesteps)
        game, true_omega, defender_mat, _target_nodes, coverage = build_lobeke_zero_sum_game(
            gsg,
            eta=self.eta,
            radius_multiplier=self.radius_multiplier,
            behavioral_fit_tolerance=self.behavioral_fit_tolerance,
        )
        evaluation_leader_reward = unit_interval_leader_reward(defender_mat)

        qre_solve_config = QRESolveConfig(root_tolerance=self.qre_tolerance, max_evaluations=self.qre_max_evaluations)
        leader_policy, follower_policy = solve_qre(game, true_omega, qre_solve_config)
        expert = validate_qre(
            game,
            true_omega,
            leader_policy,
            follower_policy,
            QREValidationConfig(equilibrium_tolerance=self.qre_tolerance),
        )
        fixed_response = validate_follower_best_response(game.problem, true_omega, leader_policy, follower_policy)

        leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
        fisher_metric = StackelbergFisherError(
            game.problem, true_omega, evaluation_leader_reward, FisherTransferMetricConfig(leader=leader_config)
        )
        optimal_leader_sol = optimize_leader_policy_batch(
            game.problem, true_omega[None, :], evaluation_leader_reward, leader_config, (0,)
        )[0]

        def oracle_factory(_: int) -> SoftmaxFollowerOracle:
            return SoftmaxFollowerOracle(game.problem, true_omega)

        def evaluate_estimates(estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
            n_checkpoints = estimates.shape[0]
            coeff_l2_errors = np.linalg.norm(estimates - true_omega[None, :], axis=1)
            self_concordance_distances = _self_concordance_distances(
                game.problem.features, estimates - true_omega[None, :]
            )

            est_solutions = optimize_leader_policy_batch(
                game.problem,
                estimates,
                evaluation_leader_reward,
                leader_config,
                tuple(seed for _ in range(n_checkpoints)),
            )
            deployed_values = np.empty(n_checkpoints, dtype=np.float64)
            for i, est_sol in enumerate(est_solutions):
                deployed_value, _ = leader_value_and_gradient(
                    game.problem, true_omega, evaluation_leader_reward, est_sol.policy
                )
                deployed_values[i] = deployed_value
            stackelberg_subopt = _validated_stackelberg_suboptimality(optimal_leader_sol.value, deployed_values)

            return {
                "stackelberg_suboptimality": stackelberg_subopt,
                "fisher_transfer_error": fisher_metric.evaluate_many(estimates, seed=seed),
                "coefficient_l2_error": coeff_l2_errors,
                "self_concordance_distance": self_concordance_distances,
            }

        return ZeroSumApplicationRuntime(
            name="guard_lobeke_paper",
            problem=game.problem,
            oracle_factory=oracle_factory,
            evaluate_estimates=evaluate_estimates,
            metadata={
                "game_name": "lobeke",
                "num_rows": self.num_rows,
                "num_columns": self.num_columns,
                "num_timesteps": self.num_timesteps,
                "eta": self.eta,
                "n_leader_actions": game.n_max_actions,
                "n_follower_actions": game.n_min_actions,
                "feature_dimension": game.feature_dimension,
                "identifiable_dimension": game.problem.identifiable_dimension,
                "true_omega": true_omega.tolist(),
                "qre_expert_leader_policy": leader_policy.tolist(),
                "qre_expert_follower_policy": follower_policy.tolist(),
                "fixed_mle_expert_definition": "source_stackelberg_optimum",
                "fixed_mle_leader_policy": optimal_leader_sol.policy.tolist(),
                "qre_max_residual": expert.max_residual,
                "qre_min_residual": expert.min_residual,
                "minimum_expert_follower_probability": float(np.min(follower_policy)),
                "negative_target_suboptimality_tolerance": (NEGATIVE_TARGET_SUBOPTIMALITY_TOLERANCE),
                "self_concordance_certificate_threshold": 1.0 / game.problem.eta,
                "defender_reward": evaluation_leader_reward.tolist(),
                "raw_defender_reward": defender_mat.tolist(),
                "leader_reward_normalization": leader_reward_normalization_metadata(defender_mat),
                "coverage_indicator": coverage.tolist(),
            },
            game=game,
            expert=expert,
            fixed_response=fixed_response,
            true_omega=true_omega,
            fixed_mle_leader_policy=optimal_leader_sol.policy,
            fixed_mle_expert_definition="source_stackelberg_optimum",
        )
