"""Task-averaged transfer diagnostics for retained Lobeke estimate histories."""

from __future__ import annotations

import numpy as np

from experiments.guard.paper.figures.data import SeedTrace
from mairl.core.math import cumulative_fisher_information_batch
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConvergenceError
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy_batch

NEGATIVE_SUBOPTIMALITY_TOLERANCE = 1.0e-9


def _evaluation_seeds(traces: list[SeedTrace], *, task_seed: int, task_index: int) -> tuple[int, ...]:
    """Use identical optimization randomness across methods for paired traces."""

    return tuple(
        task_seed * 1_000_000 + task_index * 100_000 + trace.seed * 100 + checkpoint_index
        for trace in traces
        for checkpoint_index in range(trace.checkpoints.size)
    )


def task_transfer_metrics(problem: CompiledProblem, true_omega: FloatArray,
        leader_reward: FloatArray, leader_config: LeaderOptimizationConfig, traces: list[SeedTrace],
        *, task_seed: int, task_index: int) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Return task-specific SubOpt and transfer-lemma Fisher error.

    Both arrays have shape ``(seeds, checkpoints)``.  The Fisher quantity is the
    maximum of the coefficient-error norm induced by the Fisher information at
    the task's true optimal policy and at the policy optimized from the estimate.
    """

    checkpoints = traces[0].checkpoints
    if any(not np.array_equal(trace.checkpoints, checkpoints) for trace in traces):
        raise ValueError("inconsistent checkpoints across paired traces")

    true_solution = optimize_leader_policy_batch(
        problem,
        np.asarray(true_omega, dtype=np.float64)[None, :],
        leader_reward,
        leader_config,
        (task_seed + task_index,),
    )[0]
    estimates = np.concatenate([trace.omega_history for trace in traces], axis=0)
    solutions = optimize_leader_policy_batch(
        problem,
        estimates,
        leader_reward,
        leader_config,
        _evaluation_seeds(traces, task_seed=task_seed, task_index=task_index),
    )
    estimated_policies = np.asarray([solution.policy for solution in solutions], dtype=np.float64)

    deployed_values = np.asarray(
        [leader_value_and_gradient(problem, true_omega, leader_reward, policy)[0] for policy in estimated_policies],
        dtype=np.float64,
    )
    raw_suboptimality = true_solution.value - deployed_values
    minimum_suboptimality = float(np.min(raw_suboptimality))
    if minimum_suboptimality < -NEGATIVE_SUBOPTIMALITY_TOLERANCE:
        raise ConvergenceError(
            f"estimated deployment exceeded the computed true optimum by {-minimum_suboptimality:.3e}"
        )
    suboptimality = np.maximum(0.0, raw_suboptimality)

    errors = np.asarray(true_omega, dtype=np.float64)[None, :] - estimates
    true_policies = np.broadcast_to(true_solution.policy, estimated_policies.shape)
    true_information = cumulative_fisher_information_batch(problem, true_policies[:, None, :], estimates)
    deployed_information = cumulative_fisher_information_batch(problem, estimated_policies[:, None, :], estimates)
    true_quadratics = np.einsum("nd,nde,ne->n", errors, true_information, errors, optimize=True)
    deployed_quadratics = np.einsum("nd,nde,ne->n", errors, deployed_information, errors, optimize=True)
    minimum_quadratic = float(min(np.min(true_quadratics), np.min(deployed_quadratics)))
    if minimum_quadratic < -1.0e-10:
        raise ConvergenceError(f"transfer Fisher quadratic is unexpectedly negative: {minimum_quadratic:.3e}")
    fisher_error = np.sqrt(np.maximum(0.0, np.maximum(true_quadratics, deployed_quadratics)))

    shape = (len(traces), checkpoints.size)
    return suboptimality.reshape(shape), fisher_error.reshape(shape)


def averaged_transfer_metrics(problem: CompiledProblem, true_omega: FloatArray,
        leader_rewards: FloatArray, leader_config: LeaderOptimizationConfig,
        traces: list[SeedTrace], *, task_seed: int) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Average task-specific metrics within every seed and checkpoint."""

    task_suboptimality: list[FloatArray] = []
    task_fisher_error: list[FloatArray] = []
    for task_index, leader_reward in enumerate(leader_rewards):
        suboptimality, fisher_error = task_transfer_metrics(
            problem, true_omega, leader_reward, leader_config, traces, task_seed=task_seed, task_index=task_index
        )
        task_suboptimality.append(suboptimality)
        task_fisher_error.append(fisher_error)
    return (np.mean(np.stack(task_suboptimality, axis=0), axis=0), np.mean(np.stack(task_fisher_error, axis=0), axis=0))
