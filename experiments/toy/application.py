"""A minimal application implemented outside the MAIRL source packages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import CompiledProblem, FloatArray, compile_problem
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.evaluation.metrics import (
    FisherTransferMetricConfig,
    StackelbergFisherError,
    StackelbergSuboptimality,
    StackelbergSuboptimalityConfig,
)


def make_toy_problem(seed: int, *, eta: float = 1.25, radius: float = 2.0, true_omega: ArrayLike | None = None,
        leader_reward: ArrayLike | None = None) -> tuple[CompiledProblem, FloatArray, FloatArray]:  # fmt: skip
    """Create the synthetic game used by the example experiment."""

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(3, 4, 3))
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features /= np.maximum(1.0, 1.1 * norms)
    omega = np.asarray([0.55, -0.65, 0.35] if true_omega is None else true_omega, dtype=np.float64)
    reward = np.asarray(
        (
            [[0.1, 0.9, 0.3, -0.2], [0.7, -0.1, 0.5, 0.2], [-0.2, 0.4, 0.8, 0.6]]
            if leader_reward is None
            else leader_reward
        ),
        dtype=np.float64,
    )
    if omega.shape != (3,) or not np.all(np.isfinite(omega)):
        raise ConfigurationError("toy true_omega must be finite with shape (3,)")
    if reward.shape != (3, 4) or not np.all(np.isfinite(reward)):
        raise ConfigurationError("toy leader_reward must be finite with shape (3, 4)")
    return compile_problem(features, eta=eta, radius=radius), omega, reward


@dataclass(frozen=True, slots=True)
class ToyApplication:
    """Thin adapter from the synthetic game to the generic experiment contract."""

    problem_seed: int = 17
    eta: float = 1.25
    radius: float = 2.0
    true_omega: tuple[float, ...] = (0.55, -0.65, 0.35)
    leader_reward: tuple[tuple[float, ...], ...] = ((0.1, 0.9, 0.3, -0.2), (0.7, -0.1, 0.5, 0.2), (-0.2, 0.4, 0.8, 0.6))

    def build(self, numerics: object) -> ApplicationRuntime:
        """Compile the problem and expose only factories and evaluation metrics."""

        if not isinstance(numerics, ExperimentNumerics):
            raise TypeError("ToyApplication requires ExperimentNumerics")
        problem, true_omega, leader_reward = make_toy_problem(
            self.problem_seed,
            eta=self.eta,
            radius=self.radius,
            true_omega=self.true_omega,
            leader_reward=self.leader_reward,
        )
        fisher_metric = StackelbergFisherError(
            problem,
            true_omega,
            leader_reward,
            FisherTransferMetricConfig(leader=LeaderOptimizationConfig(numerics.evaluation_policy)),
        )
        suboptimality_metric = StackelbergSuboptimality(
            problem,
            true_omega,
            leader_reward,
            StackelbergSuboptimalityConfig(leader=LeaderOptimizationConfig(numerics.evaluation_policy)),
        )

        def oracle_factory(_: int) -> SoftmaxFollowerOracle:
            return SoftmaxFollowerOracle(problem, true_omega)

        def evaluate_estimates(estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
            return {
                "fisher_transfer_error": fisher_metric.evaluate_many(estimates, seed=seed),
                "stackelberg_suboptimality": suboptimality_metric.evaluate_many(estimates, seed=seed),
                "coefficient_l2_error": np.linalg.norm(estimates - true_omega[None, :], axis=1),
            }

        return ApplicationRuntime(
            name="toy",
            problem=problem,
            oracle_factory=oracle_factory,
            evaluate_estimates=evaluate_estimates,
            metadata={
                "problem_seed": self.problem_seed,
                "eta": self.eta,
                "radius": self.radius,
                "true_omega": true_omega.tolist(),
                "leader_reward": leader_reward.tolist(),
            },
        )
