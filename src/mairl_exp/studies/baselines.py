"""Random-exploration baselines for active reward estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mairl.core.oracle import FollowerOracle, sample_follower_action
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig, WarmupDesign
from mairl.exploration.warmup import PreparedWarmup, WarmupResult, prepare_warmup, run_warmup
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, MLEResult, solve_mle

RandomBaselineMethod = Literal["fixed_random", "random_each_round"]


@dataclass(frozen=True, slots=True)
class RandomBaselineConfig:
    """Configuration shared by the two uniform-simplex exploration baselines."""

    total_samples: int
    method: RandomBaselineMethod
    mle: MLEConfig
    warmup: WarmupConfig
    seed: int = 0

    def __post_init__(self) -> None:
        if self.total_samples < 0:
            raise ConfigurationError("total_samples must be nonnegative")
        if self.method not in ("fixed_random", "random_each_round"):
            raise ConfigurationError("method must be 'fixed_random' or 'random_each_round'")
        if self.total_samples < self.warmup.effective_budget:
            raise ConfigurationError(
                "total_samples must be at least the enabled warm-up budget; "
                f"got total={self.total_samples}, warm-up={self.warmup.effective_budget}"
            )


@dataclass(frozen=True, slots=True)
class RandomBaselineResult:
    omega: FloatArray
    status: str
    degenerate: bool
    dataset: InteractionDataset
    warmup_design: WarmupDesign
    warmup_mle: MLEResult
    final_mle: MLEResult
    omega_history: FloatArray
    sample_counts: np.ndarray


class RandomExplorationBaseline:
    """Estimate rewards while selecting leader policies independently of the MLE."""

    def __init__(self, problem: CompiledProblem, config: RandomBaselineConfig) -> None:
        self.problem = problem
        self.config = config

    def prepare(self) -> PreparedWarmup:
        return prepare_warmup(self.problem, self.config.warmup)

    def run(self, oracle: FollowerOracle, *, prepared_warmup: PreparedWarmup | None = None,
            on_sample: Callable[[int], None] | None = None) -> RandomBaselineResult:  # fmt: skip
        """Run warm-up if enabled, then use the configured random policy rule."""

        prepared = prepared_warmup or self.prepare()
        warmup = run_warmup(
            self.problem,
            oracle,
            self.config.warmup,
            self.config.mle,
            experiment_rng(self.config.seed, RandomPurpose.WARMUP_FOLLOWER),
            prepared=prepared,
        )
        if self.problem.is_degenerate:
            return self._result(warmup, warmup.dataset, warmup.mle, [], [])

        dataset = warmup.dataset
        mle = warmup.mle
        omega_history: list[FloatArray] = []
        sample_counts: list[int] = []
        fixed_policy = (
            experiment_rng(self.config.seed, RandomPurpose.FIXED_RANDOM_POLICY).dirichlet(
                np.ones(self.problem.n_leader_actions)
            )
            if self.config.method == "fixed_random"
            else None
        )

        while dataset.size < self.config.total_samples:
            sample_index = dataset.size
            policy = fixed_policy
            if policy is None:
                policy = experiment_rng(self.config.seed, RandomPurpose.ONLINE_RANDOM_POLICY, sample_index).dirichlet(
                    np.ones(self.problem.n_leader_actions)
                )
            action = sample_follower_action(
                oracle,
                self.problem,
                policy,
                experiment_rng(self.config.seed, RandomPurpose.ONLINE_FOLLOWER, sample_index),
            )
            dataset = dataset.append(policy, action)
            mle = solve_mle(self.problem, dataset, self.config.mle, initial=mle.omega)
            omega_history.append(mle.omega.copy())
            sample_counts.append(dataset.size)
            if on_sample is not None:
                on_sample(dataset.size)

        return self._result(warmup, dataset, mle, omega_history, sample_counts)

    def _result(self, warmup: WarmupResult, dataset: InteractionDataset, mle: MLEResult,
            online_history: list[FloatArray], online_sample_counts: list[int]) -> RandomBaselineResult:  # fmt: skip
        history = [warmup.mle.omega.copy(), *online_history]
        counts = [warmup.dataset.size, *online_sample_counts]
        degenerate = self.problem.is_degenerate
        return RandomBaselineResult(
            omega=mle.omega,
            status=("degenerate_no_identifiable_reward_directions" if degenerate else "completed"),
            degenerate=degenerate,
            dataset=dataset,
            warmup_design=warmup.design,
            warmup_mle=warmup.mle,
            final_mle=mle,
            omega_history=np.stack(history),
            sample_counts=np.asarray(counts, dtype=np.int64),
        )
