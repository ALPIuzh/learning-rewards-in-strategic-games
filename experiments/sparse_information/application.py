"""An analytically transparent sparse-information Stackelberg game."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mairl.core.math import fisher_information
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import CompiledProblem, FloatArray, compile_problem
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics


def make_sparse_information_problem(*, leader_actions: int = 16,
        eta: float = 2.0, radius: float = 2.0, true_omega: float = 1.0,
        safe_payoff: float = 0.75) -> tuple[CompiledProblem, FloatArray, FloatArray]:  # fmt: skip
    r"""Construct a game with one informative leader action.

    With ``x = mu[0]``, the follower selects action one with probability
    ``sigmoid(eta * true_omega * x)``. Its scalar Fisher information is
    ``eta**2 * x**2 * p * (1 - p)``. All leader actions except zero are
    informationally identical decoys.
    """

    if leader_actions < 2:
        raise ConfigurationError("leader_actions must be at least two")
    if not eta > 0 or not radius > 0:
        raise ConfigurationError("eta and radius must be strictly positive")
    if not np.isfinite(true_omega) or abs(true_omega) > radius:
        raise ConfigurationError("true_omega must be finite and lie inside the radius")
    if not np.isfinite(safe_payoff):
        raise ConfigurationError("safe_payoff must be finite")

    features = np.zeros((leader_actions, 2, 1), dtype=np.float64)
    features[0, 1, 0] = 1.0
    omega = np.asarray([true_omega], dtype=np.float64)

    # Action zero earns one only when the follower takes action one. Every decoy
    # guarantees safe_payoff, independently of the follower.
    leader_reward = np.full((leader_actions, 2), safe_payoff, dtype=np.float64)
    leader_reward[0] = (0.0, 1.0)
    return compile_problem(features, eta=eta, radius=radius), omega, leader_reward


@dataclass(frozen=True, slots=True)
class SparseInformationApplication:
    """External experiment adapter for the sparse-information construction."""

    leader_actions: int = 16
    eta: float = 2.0
    radius: float = 2.0
    true_omega: float = 1.0
    safe_payoff: float = 0.75

    def build(self, numerics: object) -> ApplicationRuntime:
        """Compile the game and expose seeded oracles and evaluation metrics."""

        if not isinstance(numerics, ExperimentNumerics):
            raise TypeError("SparseInformationApplication requires ExperimentNumerics")
        problem, true_omega, leader_reward = make_sparse_information_problem(
            leader_actions=self.leader_actions,
            eta=self.eta,
            radius=self.radius,
            true_omega=self.true_omega,
            safe_payoff=self.safe_payoff,
        )
        leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
        true_solution = optimize_leader_policy(problem, true_omega, leader_reward, leader_config, seed=0)

        def oracle_factory(_: int) -> SoftmaxFollowerOracle:
            return SoftmaxFollowerOracle(problem, true_omega)

        def evaluate_estimates(estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
            fisher_errors = np.empty(estimates.shape[0], dtype=np.float64)
            suboptimalities = np.empty(estimates.shape[0], dtype=np.float64)
            for index, estimate in enumerate(estimates):
                estimated_solution = optimize_leader_policy(
                    problem, estimate, leader_reward, leader_config, seed=seed + index
                )
                error = true_omega - estimate
                true_policy_information = fisher_information(problem, true_solution.policy, estimate)
                estimated_policy_information = fisher_information(problem, estimated_solution.policy, estimate)
                fisher_errors[index] = np.sqrt(
                    max(
                        0.0,
                        float(error @ true_policy_information @ error),
                        float(error @ estimated_policy_information @ error),
                    )
                )
                deployed_value, _ = leader_value_and_gradient(
                    problem, true_omega, leader_reward, estimated_solution.policy
                )
                suboptimalities[index] = max(0.0, true_solution.value - deployed_value)
            return {
                "fisher_transfer_error": fisher_errors,
                "stackelberg_suboptimality": suboptimalities,
                "coefficient_l2_error": np.linalg.norm(estimates - true_omega[None, :], axis=1),
            }

        information_ratio = self.leader_actions * (self.leader_actions + 1) / 2
        return ApplicationRuntime(
            name="sparse-information",
            problem=problem,
            oracle_factory=oracle_factory,
            evaluate_estimates=evaluate_estimates,
            metadata={
                "leader_actions": self.leader_actions,
                "follower_actions": 2,
                "feature_dimension": 1,
                "informative_leader_action": 0,
                "eta": self.eta,
                "radius": self.radius,
                "true_omega": self.true_omega,
                "safe_payoff": self.safe_payoff,
                "initial_active_to_random_fisher_ratio": information_ratio,
            },
        )
