from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.guard.paper.figures.common import _array_sha256, _trace_context_sha256, _valid_cached_transfer_arrays
from experiments.guard.paper.figures.corrected_traces import comparison_config
from experiments.guard.paper.figures.data import (
    SeedTrace,
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_in_unit_reward,
)
from experiments.guard.paper.figures.task_sets import (
    TRANSFER_EVALUATION_TASKS,
    TUNING_TASKS,
    StructuredTaskSet,
    normalized_structured_tasks,
)
from experiments.guard.paper.figures.transfer_diagnostics import averaged_transfer_metrics, task_transfer_metrics
from experiments.guard.paper.rewards import unit_interval_leader_reward
from mairl.exceptions import ConvergenceError
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.evaluation.metrics import FisherTransferMetricConfig, StackelbergFisherError


def _leader_config() -> LeaderOptimizationConfig:
    return LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="grid", grid_resolution=12, grid_max_points=1_000, refine_grid_solution=False)
    )


def test_composed_run_config_ignores_only_scheduling_fields(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    base = {
        "application": {"eta": 1.25},
        "numerics": {"mle": {"ridge": 0.0}},
        "experiment": {"name": "mle-run", "methods": ["active"], "warmup_modes": [True], "total_samples": 1600},
    }
    other = json.loads(json.dumps(base))
    other["experiment"].update({"name": "ols-run", "methods": ["zero_sum_ols"], "warmup_modes": [False]})
    (first / "semantic_config.json").write_text(json.dumps(base))
    (second / "semantic_config.json").write_text(json.dumps(other))

    assert comparison_config(first) == comparison_config(second)
    other["application"]["eta"] = 2.0
    (second / "semantic_config.json").write_text(json.dumps(other))
    assert comparison_config(first) != comparison_config(second)


def test_transfer_cache_identity_includes_shape_dtype_and_optimizer_seeds() -> None:
    vector = np.asarray([1, 2], dtype=np.int64)
    assert _array_sha256(vector) != _array_sha256(vector.reshape(1, 2))
    assert _array_sha256(vector) != _array_sha256(vector.astype(np.float64))

    trace = SeedTrace(seed=11, checkpoints=np.asarray([40, 80]), metrics={}, omega_history=np.zeros((2, 2)))
    reference = _trace_context_sha256([trace], task_seed=9_200, task_index=0)
    assert reference != _trace_context_sha256(
        [SeedTrace(12, trace.checkpoints, {}, trace.omega_history)], task_seed=9_200, task_index=0
    )
    assert reference != _trace_context_sha256([trace], task_seed=9_201, task_index=0)
    assert reference != _trace_context_sha256([trace], task_seed=9_200, task_index=1)
    assert reference != _trace_context_sha256(
        [SeedTrace(11, np.asarray([40, 120]), {}, trace.omega_history)], task_seed=9_200, task_index=0
    )


def test_transfer_cache_rejects_invalid_derived_arrays() -> None:
    valid = np.zeros((2, 3), dtype=np.float64)
    assert _valid_cached_transfer_arrays(valid, valid, expected_shape=(2, 3))
    assert not _valid_cached_transfer_arrays(valid[:, :2], valid, expected_shape=(2, 3))
    assert not _valid_cached_transfer_arrays(np.full((2, 3), np.nan), valid, expected_shape=(2, 3))
    assert not _valid_cached_transfer_arrays(np.zeros((2, 3), dtype=np.int64), valid, expected_shape=(2, 3))


def test_task_transfer_fisher_matches_metric_definition(small_problem, small_leader_reward, small_true_omega) -> None:
    estimates = np.asarray([[0.1, -0.2], [0.35, -0.6]])
    reward = small_leader_reward
    trace = SeedTrace(seed=17, checkpoints=np.asarray([10, 20]), metrics={}, omega_history=estimates)
    config = _leader_config()

    suboptimality, fisher_error = task_transfer_metrics(
        small_problem, small_true_omega, reward, config, [trace], task_seed=123, task_index=0
    )
    metric = StackelbergFisherError(small_problem, small_true_omega, reward, FisherTransferMetricConfig(leader=config))

    assert suboptimality.shape == fisher_error.shape == (1, 2)
    np.testing.assert_allclose(fisher_error[0], np.asarray([metric(estimate) for estimate in estimates]), atol=1e-14)
    assert np.all(suboptimality >= 0.0)


def test_averaged_transfer_metrics_average_taskwise_scalars(small_problem, small_true_omega) -> None:
    trace = SeedTrace(
        seed=17, checkpoints=np.asarray([10, 20]), metrics={}, omega_history=np.asarray([[0.1, -0.2], [0.35, -0.6]])
    )
    tasks = np.asarray(
        [[[0.0, 0.4, 1.0], [0.8, -0.1, 0.2], [0.3, 0.7, -0.2]], [[0.3, -0.1, 0.5], [-0.2, 0.8, 0.4], [0.7, 0.1, -0.3]]]
    )
    config = _leader_config()
    individual = [
        task_transfer_metrics(small_problem, small_true_omega, reward, config, [trace], task_seed=123, task_index=index)
        for index, reward in enumerate(tasks)
    ]
    averaged = averaged_transfer_metrics(small_problem, small_true_omega, tasks, config, [trace], task_seed=123)

    np.testing.assert_allclose(averaged[0], np.mean([metrics[0] for metrics in individual], axis=0), atol=1e-14)
    np.testing.assert_allclose(averaged[1], np.mean([metrics[1] for metrics in individual], axis=0), atol=1e-14)


def test_transfer_tasks_are_normalized_and_disjoint_from_tuning() -> None:
    coverage = np.asarray([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]])
    runtime = SimpleNamespace(metadata={"coverage_indicator": coverage})
    task_set = StructuredTaskSet(name="test", count=4, seed=7, candidate_pool=20)
    tasks = normalized_structured_tasks(runtime, task_set)

    assert tasks.shape == (4, 2, 3)
    np.testing.assert_allclose(np.min(tasks, axis=(1, 2)), np.zeros(4))
    np.testing.assert_allclose(np.max(tasks, axis=(1, 2)), np.ones(4))
    assert TRANSFER_EVALUATION_TASKS.seed != TUNING_TASKS.seed


def test_leader_reward_normalization_is_affine_invariant_and_has_rmax_one() -> None:
    reward = np.asarray([[-3.0, 1.0, 5.0], [3.0, -1.0, 0.0]])
    expected = (reward + 3.0) / 8.0

    normalized = unit_interval_leader_reward(reward)
    transformed = unit_interval_leader_reward(7.0 * reward + 11.0)

    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_allclose(transformed, expected)
    assert np.max(np.abs(normalized)) == 1.0


def test_source_subopt_unit_conversion_is_exact_and_not_applied_twice() -> None:
    trace = SeedTrace(
        seed=1,
        checkpoints=np.asarray([10]),
        metrics={"stackelberg_suboptimality": np.asarray([0.25])},
        omega_history=np.zeros((1, 2)),
    )
    traces = {("active", False): [trace]}

    historical = source_suboptimality_in_unit_reward(traces, raw_reward_range=0.5)
    canonical = source_suboptimality_in_unit_reward(traces, raw_reward_range=0.5, already_unit_reward=True)

    np.testing.assert_array_equal(
        historical[("active", False)][0].metrics["stackelberg_suboptimality"], np.asarray([0.5])
    )
    np.testing.assert_array_equal(
        canonical[("active", False)][0].metrics["stackelberg_suboptimality"], np.asarray([0.25])
    )
    np.testing.assert_array_equal(trace.metrics["stackelberg_suboptimality"], np.asarray([0.25]))


def test_run_reward_provenance_rejects_mixed_metric_units(tmp_path) -> None:
    marker = {"definition": "(r - min(r)) / (max(r) - min(r))", "normalized_r_max": 1.0}
    entries = [{"job_id": "old"}, {"job_id": "unit"}]
    (tmp_path / "manifest.json").write_text(json.dumps({"expected_jobs": entries}))
    for job_id, summary in (("old", {}), ("unit", {"leader_reward_normalization": marker})):
        job_dir = tmp_path / "jobs" / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="mixes historical and unit-reward"):
        run_source_suboptimality_is_unit_reward(tmp_path)


def test_task_transfer_rejects_a_false_true_optimum(small_problem, monkeypatch, small_true_omega) -> None:
    import experiments.guard.paper.figures.transfer_diagnostics as diagnostics

    policy = np.full(small_problem.n_leader_actions, 1.0 / small_problem.n_leader_actions)
    monkeypatch.setattr(
        diagnostics, "optimize_leader_policy_batch", lambda *args, **kwargs: [SimpleNamespace(policy=policy, value=0.0)]
    )
    monkeypatch.setattr(
        diagnostics,
        "leader_value_and_gradient",
        lambda *args, **kwargs: (1.0, np.zeros(small_problem.n_leader_actions)),
    )
    trace = SeedTrace(seed=1, checkpoints=np.asarray([10]), metrics={}, omega_history=np.asarray([[0.1, -0.2]]))
    reward = np.zeros((small_problem.n_leader_actions, small_problem.n_follower_actions))

    with pytest.raises(ConvergenceError, match="exceeded the computed true optimum"):
        task_transfer_metrics(
            small_problem, small_true_omega, reward, _leader_config(), [trace], task_seed=123, task_index=0
        )
