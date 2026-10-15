"""Fixed deterministic downstream-task suites for the Lobeke diagnostics.

Keeping task definitions here prevents a plotting or tuning script from silently
choosing a favorable collection of downstream rewards. Tuning and final
diagnostics use disjoint random seeds, but uncertainty remains conditional on
the resulting fixed task suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from experiments.guard.paper.figures.data import draw_diverse_structured_tasks
from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy


@dataclass(frozen=True, slots=True)
class StructuredTaskSet:
    """A deterministic collection of normalized Lobeke leader rewards."""

    name: str
    count: int
    seed: int
    candidate_pool: int = 500
    scale: float = 1.0


TUNING_TASKS = StructuredTaskSet(name="tuning", count=10, seed=9_100)
TRANSFER_EVALUATION_TASKS = StructuredTaskSet(name="fresh_primary", count=10, seed=202_609_700)
BEHAVIORAL_TRANSFER_TASKS = StructuredTaskSet(name="fresh_behavioral", count=180, seed=202_609_701, candidate_pool=180)
ILLUSTRATION_BEHAVIOR_QUANTILES = (0.0, 0.5, 1.0)
NEGATIVE_GAP_TOLERANCE = 1.0e-9


def policy_total_variation(left: FloatArray, right: FloatArray) -> float:
    """Return total variation between two leader policies."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("leader policies must be one-dimensional with equal shape")
    if (
        np.min(left) < -1.0e-10
        or np.min(right) < -1.0e-10
        or not np.isclose(np.sum(left), 1.0, atol=1.0e-8)
        or not np.isclose(np.sum(right), 1.0, atol=1.0e-8)
    ):
        raise ValueError("leader policies must lie on the probability simplex")
    return 0.5 * float(np.sum(np.abs(left - right)))


def source_policy_gap(*, oracle_value: float, source_policy_value: float,
        tolerance: float = NEGATIVE_GAP_TOLERANCE) -> float:  # fmt: skip
    """Return downstream regret of deploying the true source-optimal policy."""

    raw = float(oracle_value - source_policy_value)
    if raw < -tolerance:
        raise ConvergenceError(f"source-policy value exceeded the computed downstream optimum by {-raw:.3e}")
    return max(0.0, raw)


def quantile_bucket_assignments(values: FloatArray, *, bucket_count: int = 3,
        tie_relative_tolerance: float = 1.0e-10) -> tuple[FloatArray, np.ndarray]:  # fmt: skip
    """Assign values to balanced quantile buckets without splitting numerical ties."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < bucket_count or not np.all(np.isfinite(values)):
        raise ValueError("bucket values must be a sufficiently long finite vector")
    if bucket_count < 2:
        raise ValueError("at least two buckets are required")
    if tie_relative_tolerance < 0.0:
        raise ValueError("tie tolerance must be nonnegative")

    ordered = np.sort(values)
    adjacent_scale = np.maximum(1.0, np.maximum(np.abs(ordered[:-1]), np.abs(ordered[1:])))
    boundaries = np.flatnonzero(np.diff(ordered) > tie_relative_tolerance * adjacent_scale) + 1
    if boundaries.size < bucket_count - 1:
        raise ValueError("transfer-distance metric has too few distinct values for the buckets")

    targets = values.size * np.arange(1, bucket_count) / bucket_count
    cuts = np.asarray(
        min(
            combinations(boundaries.tolist(), bucket_count - 1),
            key=lambda choice: float(np.sum((np.asarray(choice) - targets) ** 2)),
        ),
        dtype=np.int64,
    )
    internal_edges = np.asarray([(ordered[cut - 1] + ordered[cut]) / 2.0 for cut in cuts], dtype=np.float64)
    edges = np.concatenate(([ordered[0]], internal_edges, [ordered[-1]]))
    assignments = np.searchsorted(internal_edges, values, side="right")
    return edges, assignments.astype(np.int64)


@dataclass(frozen=True, slots=True)
class BehavioralTaskSelection:
    """Three fixed-suite tasks spanning oracle-policy distance from the source."""

    tasks: FloatArray
    suite_indices: tuple[int, ...]
    source_policy_total_variation: tuple[float, ...]
    pairwise_policy_total_variation: FloatArray
    suite_source_policy_total_variation: tuple[float, ...]


def normalized_structured_tasks(runtime: Any, task_set: StructuredTaskSet) -> FloatArray:
    """Generate fixed tasks with every leader reward canonically in ``[0, 1]``."""

    coverage = runtime.metadata.get("coverage_indicator")
    if coverage is None:
        raise ConfigurationError("coverage_indicator is missing from the application runtime")
    tasks, _ = draw_diverse_structured_tasks(
        np.asarray(coverage, dtype=np.float64),
        num_tasks=task_set.count,
        seed=task_set.seed,
        scale=task_set.scale,
        candidate_pool=task_set.candidate_pool,
    )
    if (
        not np.all(np.isfinite(tasks))
        or not np.allclose(np.min(tasks, axis=(1, 2)), 0.0)
        or not np.allclose(np.max(tasks, axis=(1, 2)), 1.0)
    ):
        raise ConfigurationError(f"{task_set.name} contains an invalid normalized reward")
    return tasks


def _quantile_representative_indices(distances: FloatArray, quantiles: tuple[float, ...]) -> tuple[int, ...]:
    """Select deterministic task indices nearest to empirical distance quantiles."""

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or not np.all(np.isfinite(distances)):
        raise ConfigurationError("behavioral distances must be a finite vector")
    if len(quantiles) > distances.size or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ConfigurationError("behavioral quantiles must lie in [0,1] and fit the task suite")

    # Rounding makes solver noise below publication precision irrelevant. Ties
    # are broken by the predeclared suite index, never by an algorithm's score.
    rounded = np.round(distances, decimals=12)
    targets = np.quantile(rounded, quantiles)
    available = set(range(distances.size))
    selected: list[int] = []
    for target in targets:
        index = min(available, key=lambda item: (abs(rounded[item] - target), item))
        selected.append(index)
        available.remove(index)
    return tuple(selected)


def select_behaviorally_stratified_tasks(runtime: Any,
        task_set: StructuredTaskSet, leader_config: LeaderOptimizationConfig,
        *, quantiles: tuple[float, ...] = ILLUSTRATION_BEHAVIOR_QUANTILES) -> BehavioralTaskSelection:  # fmt: skip
    """Choose fixed-suite illustrations by oracle-policy TV from the source.

    Selection depends only on the true source and downstream optimal leader
    policies. It never inspects any learned estimate or method performance.
    """

    tasks = normalized_structured_tasks(runtime, task_set)
    source_reward = runtime.metadata.get("defender_reward")
    if source_reward is None:
        raise ConfigurationError("defender_reward is missing from the application runtime")
    source_policy = optimize_leader_policy(
        runtime.problem,
        runtime.true_omega,
        np.asarray(source_reward, dtype=np.float64),
        leader_config,
        seed=task_set.seed,
    ).policy
    task_policies = np.asarray(
        [
            optimize_leader_policy(
                runtime.problem, runtime.true_omega, reward, leader_config, seed=task_set.seed + index
            ).policy
            for index, reward in enumerate(tasks)
        ]
    )
    source_distances = np.asarray([policy_total_variation(policy, source_policy) for policy in task_policies])
    indices = _quantile_representative_indices(source_distances, quantiles)
    selected_policies = task_policies[list(indices)]
    pairwise_distances = 0.5 * np.abs(selected_policies[:, None, :] - selected_policies[None, :, :]).sum(axis=2)
    return BehavioralTaskSelection(
        tasks=tasks[list(indices)].copy(),
        suite_indices=indices,
        source_policy_total_variation=tuple(float(source_distances[index]) for index in indices),
        pairwise_policy_total_variation=pairwise_distances,
        suite_source_policy_total_variation=tuple(float(value) for value in source_distances),
    )
