"""Zero-sum version of the nontrivial toy game, kept outside the libraries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.toy.application import make_toy_problem
from mairl import (
    CompiledZeroSumGame,
    QRESolveConfig,
    QREValidationConfig,
    ValidatedFollowerBestResponse,
    ValidatedQRE,
    compile_zero_sum_game,
    solve_qre,
    validate_follower_best_response,
    validate_qre,
)
from mairl.core.math import fisher_information
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics


@dataclass(frozen=True, slots=True)
class ZeroSumApplicationRuntime(ApplicationRuntime):
    """Generic experiment runtime plus zero-sum information used only by this study."""

    game: CompiledZeroSumGame
    expert: ValidatedQRE
    fixed_response: ValidatedFollowerBestResponse
    true_omega: FloatArray
    fixed_mle_leader_policy: FloatArray
    fixed_mle_expert_definition: str


@dataclass(frozen=True, slots=True)
class ZeroSumToyApplication:
    """Build a zero-sum game from the existing nontrivial toy feature geometry."""

    problem_seed: int = 17
    eta: float = 1.25
    radius: float = 2.0
    true_omega: tuple[float, ...] = (0.55, -0.65, 0.35)
    qre_tolerance: float = 1e-10
    qre_max_evaluations: int = 10_000

    def __post_init__(self) -> None:
        if self.qre_tolerance <= 0:
            raise ConfigurationError("qre_tolerance must be strictly positive")
        if self.qre_max_evaluations < 1:
            raise ConfigurationError("qre_max_evaluations must be positive")

    def build(self, numerics: object) -> ZeroSumApplicationRuntime:
        """Compile the game and validate its fixed expert QRE."""

        if not isinstance(numerics, ExperimentNumerics):
            raise TypeError("ZeroSumToyApplication requires ExperimentNumerics")
        follower_problem, true_omega, _ = make_toy_problem(
            self.problem_seed, eta=self.eta, radius=self.radius, true_omega=self.true_omega
        )
        game = compile_zero_sum_game(
            -follower_problem.features, follower_problem.features, eta=self.eta, radius=self.radius
        )
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
        fixed_leader_information = fisher_information(game.problem, leader_policy, true_omega)

        follower_expected_features = np.einsum("a,abd->bd", leader_policy, game.min_player_features, optimize=True)
        follower_matrix = follower_expected_features[:-1] - follower_expected_features[-1]
        leader_differences = game.max_player_features[:-1] - game.max_player_features[-1][None, :, :]
        leader_matrix = np.einsum("abd,b->ad", leader_differences, follower_policy, optimize=True)
        stacked_matrix = np.vstack((leader_matrix, follower_matrix))

        def oracle_factory(_: int) -> SoftmaxFollowerOracle:
            return SoftmaxFollowerOracle(game.problem, true_omega)

        def evaluate_estimates(estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
            del seed
            errors = estimates - true_omega[None, :]
            fixed_leader_quadratics = np.einsum("nd,de,ne->n", errors, fixed_leader_information, errors, optimize=True)
            qre_leader_errors = np.empty(estimates.shape[0], dtype=np.float64)
            for index, estimate in enumerate(estimates):
                estimated_leader, _ = solve_qre(game, estimate, qre_solve_config)
                qre_leader_errors[index] = np.linalg.norm(estimated_leader - leader_policy, ord=1)
            return {
                "fixed_leader_fisher_error": np.sqrt(np.maximum(0.0, fixed_leader_quadratics)),
                "coefficient_l2_error": np.linalg.norm(errors, axis=1),
                "regularized_qre_leader_l1_error": qre_leader_errors,
            }

        return ZeroSumApplicationRuntime(
            name="zero-sum-toy",
            problem=game.problem,
            oracle_factory=oracle_factory,
            evaluate_estimates=evaluate_estimates,
            metadata={
                "problem_seed": self.problem_seed,
                "eta": self.eta,
                "leader_entropy_coefficient": 1.0 / self.eta,
                "follower_entropy_coefficient": 1.0 / self.eta,
                "radius": self.radius,
                "true_omega": true_omega.tolist(),
                "fixed_expert_leader_policy": leader_policy.tolist(),
                "fixed_expert_follower_policy": follower_policy.tolist(),
                "minimum_expert_follower_probability": float(np.min(follower_policy)),
                "fixed_leader_fisher_eigenvalues": np.linalg.eigvalsh(fixed_leader_information).tolist(),
                "follower_only_rank": int(np.linalg.matrix_rank(follower_matrix)),
                "follower_only_condition_number": float(np.linalg.cond(follower_matrix)),
                "zero_sum_rank": int(np.linalg.matrix_rank(stacked_matrix)),
                "zero_sum_condition_number": float(np.linalg.cond(stacked_matrix)),
            },
            game=game,
            expert=expert,
            fixed_response=fixed_response,
            true_omega=true_omega,
            fixed_mle_leader_policy=leader_policy,
            fixed_mle_expert_definition="joint_qre",
        )
