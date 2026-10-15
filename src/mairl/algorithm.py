"""End-to-end active Stackelberg IRL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mairl.core.math import cumulative_fisher_information
from mairl.core.oracle import FollowerOracle, sample_follower_action
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig, WarmupDesign
from mairl.exploration.policy import PolicySelectionResult, select_information_policy_batch
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.exploration.warmup import PreparedWarmup, prepare_warmup, run_warmup
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, MLEResult, solve_mle


@dataclass(frozen=True, slots=True)
class ActiveIRLConfig:
    """Complete, serializable experiment configuration."""

    horizon: int
    lambda_information: float
    lambda_warm: float
    confidence_delta: float
    mle: MLEConfig
    warmup: WarmupConfig
    policy: PolicyOptimizationConfig = field(default_factory=PolicyOptimizationConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ConfigurationError("horizon must be nonnegative")
        if not self.lambda_information > 0:
            raise ConfigurationError("lambda_information must be strictly positive")
        if not self.lambda_warm > 0:
            raise ConfigurationError("lambda_warm must be strictly positive")
        if not 0 < self.confidence_delta < 1:
            raise ConfigurationError("confidence_delta must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class IterationDiagnostics:
    round_index: int
    total_samples: int
    selected_policy: FloatArray
    observed_follower_action: int
    information_gain: float
    information_min_eigenvalue: float
    information_max_eigenvalue: float
    policy_projected_gradient_residual: float
    policy_failed_starts: int
    policy_failed_start_messages: tuple[str, ...]
    mle_log_likelihood: float
    mle_penalized_objective: float
    mle_projected_gradient_residual: float
    mle_iterations: int
    omega_norm: float


@dataclass(frozen=True, slots=True)
class ActiveIRLResult:
    omega: FloatArray
    status: str
    degenerate: bool
    dataset: InteractionDataset
    warmup_design: WarmupDesign
    warmup_mle: MLEResult
    final_mle: MLEResult
    omega_history: FloatArray
    diagnostics: tuple[IterationDiagnostics, ...]


class ActiveIRL:
    """Compiled, reusable active-IRL runner.

    A runner is immutable with respect to model and configuration. Each :meth:`run`
    creates fresh random and dataset state, which makes repeated hyperparameter
    experiments reproducible and independent.
    """

    def __init__(self, problem: CompiledProblem, config: ActiveIRLConfig) -> None:
        self.problem = problem
        self.config = config

    def prepare(self) -> PreparedWarmup:
        """Precompute deterministic warm-up work reusable by compatible trials."""

        return prepare_warmup(self.problem, self.config.warmup)

    def run(self, oracle: FollowerOracle, *, prepared_warmup: PreparedWarmup | None = None,
            on_round: Callable[[IterationDiagnostics], None] | None = None) -> ActiveIRLResult:  # fmt: skip
        """Run Algorithm 2 using the last MLE as its output.

        The control flow intentionally mirrors the paper pseudocode; component-specific
        numerical details are delegated to the named operations.
        """

        prepared = prepared_warmup or self.prepare()
        prepared.validate_for(self.problem, self.config.warmup)
        if self.problem.is_degenerate:
            return self._degenerate_result(prepared.design)

        warmup = run_warmup(
            self.problem,
            oracle,
            self.config.warmup,
            self.config.mle,
            experiment_rng(self.config.seed, RandomPurpose.WARMUP_FOLLOWER),
            prepared=prepared,
        )
        dataset = warmup.dataset
        mle = warmup.mle
        omega_history = [mle.omega.copy()]
        diagnostics: list[IterationDiagnostics] = []

        for round_index in range(1, self.config.horizon + 1):
            sample_index = dataset.size
            # Online information includes warm-up observations; lambda_warm is only an offline certificate.
            information = self.config.lambda_information * np.eye(
                self.problem.feature_dimension
            ) + cumulative_fisher_information(self.problem, dataset.policies, mle.omega)
            policy = select_information_policy_batch(
                self.problem,
                mle.omega[None, :],
                information[None, :, :],
                self.config.policy,
                (experiment_rng(self.config.seed, RandomPurpose.ACTIVE_POLICY, sample_index),),
            )[0]
            follower_action = sample_follower_action(
                oracle,
                self.problem,
                policy.policy,
                experiment_rng(self.config.seed, RandomPurpose.ONLINE_FOLLOWER, sample_index),
            )
            dataset = dataset.append(policy.policy, follower_action)
            mle = solve_mle(self.problem, dataset, self.config.mle, initial=mle.omega)
            omega_history.append(mle.omega.copy())
            diagnostics.append(
                self._iteration_diagnostics(round_index, dataset, mle, information, policy, follower_action)
            )
            if on_round is not None:
                on_round(diagnostics[-1])

        return ActiveIRLResult(
            omega=mle.omega,
            status="completed",
            degenerate=False,
            dataset=dataset,
            warmup_design=warmup.design,
            warmup_mle=warmup.mle,
            final_mle=mle,
            omega_history=np.stack(omega_history),
            diagnostics=tuple(diagnostics),
        )

    def _iteration_diagnostics(self, round_index: int, dataset: InteractionDataset, mle: MLEResult,
            information: FloatArray, policy: PolicySelectionResult,
            follower_action: int) -> IterationDiagnostics:  # fmt: skip
        eigenvalues = np.linalg.eigvalsh(information)
        return IterationDiagnostics(
            round_index=round_index,
            total_samples=dataset.size,
            selected_policy=policy.policy.copy(),
            observed_follower_action=follower_action,
            information_gain=policy.information_gain,
            information_min_eigenvalue=float(eigenvalues[0]),
            information_max_eigenvalue=float(eigenvalues[-1]),
            policy_projected_gradient_residual=(policy.optimization.projected_gradient_residual),
            policy_failed_starts=len(policy.optimization.failed_start_messages),
            policy_failed_start_messages=policy.optimization.failed_start_messages,
            mle_log_likelihood=mle.log_likelihood,
            mle_penalized_objective=mle.penalized_objective,
            mle_projected_gradient_residual=mle.projected_gradient_residual,
            mle_iterations=mle.iterations,
            omega_norm=float(np.linalg.norm(mle.omega)),
        )

    def _degenerate_result(self, design: WarmupDesign) -> ActiveIRLResult:
        dataset = InteractionDataset.empty(self.problem.n_leader_actions)
        zero_mle = solve_mle(self.problem, dataset, self.config.mle)
        return ActiveIRLResult(
            omega=zero_mle.omega,
            status="degenerate_no_identifiable_reward_directions",
            degenerate=True,
            dataset=dataset,
            warmup_design=design,
            warmup_mle=zero_mle,
            final_mle=zero_mle,
            omega_history=zero_mle.omega[None, :],
            diagnostics=(),
        )
