"""The complete offline WARM procedure from design through initial MLE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mairl.core.oracle import FollowerOracle, sample_follower_action
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig, WarmupDesign, compute_warmup_design
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, MLEResult, solve_mle


@dataclass(frozen=True, slots=True)
class WarmupResult:
    design: WarmupDesign
    dataset: InteractionDataset
    mle: MLEResult


@dataclass(frozen=True, slots=True)
class PreparedWarmup:
    """Problem-specific warm-up design reusable across compatible trials."""

    problem: CompiledProblem
    config: WarmupConfig
    design: WarmupDesign

    def validate_for(self, problem: CompiledProblem, config: WarmupConfig) -> None:
        if self.problem is not problem:
            raise ConfigurationError("prepared warm-up belongs to a different compiled problem")
        if self.config != config:
            raise ConfigurationError("prepared warm-up configuration does not match this trial")


def prepare_warmup(problem: CompiledProblem, config: WarmupConfig) -> PreparedWarmup:
    """Compute the deterministic design once, without sampling the application."""

    return PreparedWarmup(problem=problem, config=config, design=compute_warmup_design(problem, config))


def run_warmup(problem: CompiledProblem, oracle: FollowerOracle, config: WarmupConfig, mle_config: MLEConfig,
        rng: np.random.Generator, *, prepared: PreparedWarmup | None = None) -> WarmupResult:  # fmt: skip
    """Compute the allocation, collect one sample per allocation, and fit the MLE."""

    if prepared is None:
        prepared = prepare_warmup(problem, config)
    else:
        prepared.validate_for(problem, config)

    design = prepared.design
    policies, actions = _collect_allocated_samples(problem, oracle, design, rng)
    dataset = InteractionDataset.from_rows(policies, actions, problem.n_leader_actions)
    if dataset.size != config.effective_budget:
        raise RuntimeError("internal error: warm-up sample count differs from allocation")
    return WarmupResult(design=design, dataset=dataset, mle=solve_mle(problem, dataset, mle_config))


def _collect_allocated_samples(problem: CompiledProblem, oracle: FollowerOracle, design: WarmupDesign,
        rng: np.random.Generator) -> tuple[list[FloatArray], list[int]]:  # fmt: skip
    policies: list[FloatArray] = []
    actions: list[int] = []
    for action_index, count in zip(design.support_actions, design.allocation_counts, strict=True):
        policy = np.zeros(problem.n_leader_actions, dtype=np.float64)
        policy[int(action_index)] = 1.0
        for _ in range(int(count)):
            policies.append(policy)
            actions.append(sample_follower_action(oracle, problem, policy, rng))
    return policies, actions
