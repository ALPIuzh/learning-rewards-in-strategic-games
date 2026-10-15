from __future__ import annotations

import numpy as np
import pytest

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.export_behavioral_transfer import (
    _at_checkpoints,
    _bucket_diagnostics,
)
from experiments.guard.paper.figures.task_sets import (
    policy_total_variation,
    quantile_bucket_assignments,
    source_policy_gap,
)
from investigations.behavioral_transfer.explain_dense import paired_contrast
from investigations.behavioral_transfer.oracle_stability import representative_task_indices
from investigations.behavioral_transfer.pilot import (
    METHODS,
    _distribution_summary,
    align_common_checkpoints,
    select_bucket_tasks,
)
from investigations.behavioral_transfer.ridge_refit_audit import _summary as ridge_summary
from investigations.behavioral_transfer.small_k_geometry import finite_difference_hessian
from investigations.behavioral_transfer.small_k_tuning import _best_configurations
from mairl.exceptions import ConvergenceError


def test_behavioral_distance_metrics_have_their_intended_semantics() -> None:
    source = np.asarray([0.7, 0.2, 0.1])
    same = source.copy()
    disjoint = np.asarray([0.0, 0.0, 1.0])

    assert policy_total_variation(source, same) == pytest.approx(0.0)
    assert policy_total_variation(source, disjoint) == pytest.approx(0.9)
    assert source_policy_gap(oracle_value=1.0, source_policy_value=0.7) == pytest.approx(0.3)
    assert source_policy_gap(oracle_value=1.0, source_policy_value=1.0 + 1.0e-12) == 0.0
    with pytest.raises(ConvergenceError, match="exceeded"):
        source_policy_gap(oracle_value=1.0, source_policy_value=1.1)


def test_bucket_selection_is_balanced_deterministic_and_performance_free() -> None:
    values = np.arange(12, dtype=np.float64)
    edges, assignments = quantile_bucket_assignments(values)
    first = select_bucket_tasks(assignments, tasks_per_bucket=2, seed=17)
    second = select_bucket_tasks(assignments, tasks_per_bucket=2, seed=17)

    assert np.all(np.diff(edges) > 0.0)
    assert np.array_equal(first, second)
    assert first.shape == (3, 2)
    expected_buckets = np.repeat(np.arange(3)[:, None], 2, axis=1)
    assert np.array_equal(assignments[first], expected_buckets)


def test_quantile_buckets_never_split_numerically_tied_distances() -> None:
    values = np.asarray([0.1, 0.1, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0 + 1.0e-13, 1.0])

    _, assignments = quantile_bucket_assignments(values)

    assert np.array_equal(np.bincount(assignments), np.asarray([4, 4, 4]))
    assert np.unique(assignments[values > 0.9]).size == 1


def test_distribution_summary_exposes_extremes_and_concentration() -> None:
    values = np.asarray([1.0, 1.0, 1.0, 7.0])

    summary = _distribution_summary(values, seed=3)

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["quantiles"]["minimum"] == pytest.approx(1.0)
    assert summary["quantiles"]["maximum"] == pytest.approx(7.0)
    assert summary["largest_task_share_of_sum"] == pytest.approx(0.7)
    assert summary["worst_ten_percent_share_of_sum"] == pytest.approx(0.7)


def test_oracle_stability_probes_are_metric_based_and_deterministic() -> None:
    manifest = {
        "candidate_metrics": {
            "tv": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "gap": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        },
        "buckets": {
            "tv": {"assignments": [0, 0, 0, 1, 1, 1, 2, 2, 2]},
            "gap": {"assignments": [2, 2, 2, 1, 1, 1, 0, 0, 0]},
        },
    }

    selected = representative_task_indices(manifest)

    assert selected == list(range(9))


def test_paired_contrast_preserves_pairs_and_reports_direction() -> None:
    contrast = paired_contrast(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert contrast["mean"] == pytest.approx(2.5)
    assert contrast["positive_pair_count"] == 4
    assert contrast["pair_count"] == 4
    assert contrast["student_t_95"][0] < 2.5 < contrast["student_t_95"][1]
    assert contrast["paired_differences"] == [1.0, 2.0, 3.0, 4.0]


def test_ridge_audit_summary_keeps_seed_dispersion() -> None:
    summary = ridge_summary(np.asarray([1.0, 2.0, 3.0]))

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["standard_error"] == pytest.approx(1.0 / np.sqrt(3.0))


def test_small_k_selection_keeps_criterion_specific_optima() -> None:
    records = [
        {
            "protocol": "active_no_warmup",
            "ridge": 0.1,
            "lambda_information_multiplier": 1.0,
            "source_metrics": {
                "stackelberg_suboptimality": {"mean": 2.0},
                "fisher_transfer_error": {"mean": 1.0},
                "coefficient_l2_error": {"mean": 3.0},
            },
        },
        {
            "protocol": "active_no_warmup",
            "ridge": 1.0,
            "lambda_information_multiplier": 3.0,
            "source_metrics": {
                "stackelberg_suboptimality": {"mean": 1.0},
                "fisher_transfer_error": {"mean": 3.0},
                "coefficient_l2_error": {"mean": 2.0},
            },
        },
    ]

    selected = _best_configurations(records)

    assert selected["source_suboptimality"]["ridge"] == pytest.approx(1.0)
    assert selected["source_fisher_error"]["ridge"] == pytest.approx(0.1)


def test_small_k_geometry_recovers_a_quadratic_hessian() -> None:
    expected = np.asarray([[2.0, 0.5], [0.5, 4.0]])

    observed = finite_difference_hessian(
        lambda error: 0.5 * float(error @ expected @ error),
        dimension=2,
        step=0.1,
    )

    assert observed == pytest.approx(expected)


def test_protocols_are_aligned_to_common_positive_sample_counts() -> None:
    traces = {}
    for key in METHODS:
        checkpoints = (
            np.asarray([0, 100, 200]) if key == ("active", False) else np.asarray([100, 200])
        )
        traces[key] = [
            SeedTrace(
                seed=1,
                checkpoints=checkpoints,
                metrics={"example": checkpoints.astype(np.float64)},
                omega_history=np.repeat(checkpoints[:, None], 3, axis=1),
            )
        ]

    aligned, checkpoints = align_common_checkpoints(traces)

    assert np.array_equal(checkpoints, np.asarray([100, 200]))
    for key in METHODS:
        trace = aligned[key][0]
        assert np.array_equal(trace.checkpoints, checkpoints)
        assert np.array_equal(trace.metrics["example"], checkpoints)
        assert np.array_equal(trace.omega_history[:, 0], checkpoints)


def test_dense_export_selects_only_explicit_common_checkpoints() -> None:
    trace = SeedTrace(
        seed=7,
        checkpoints=np.asarray([0, 40, 200, 400, 800]),
        metrics={"example": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])},
        omega_history=np.arange(10, dtype=np.float64).reshape(5, 2),
    )

    selected = _at_checkpoints([trace], (200, 400, 800))[0]

    assert np.array_equal(selected.checkpoints, np.asarray([200, 400, 800]))
    assert np.array_equal(selected.metrics["example"], np.asarray([2.0, 3.0, 4.0]))
    assert np.array_equal(selected.omega_history, trace.omega_history[2:])
    with pytest.raises(ValueError, match="missing"):
        _at_checkpoints([trace], (200, 600))


def test_dense_bucket_diagnostics_retain_extremes_without_task_selection() -> None:
    assignments = {
        "tv": np.asarray([0, 0, 1, 1, 2, 2]),
        "gap": np.asarray([0, 1, 0, 2, 1, 2]),
    }
    geometry = {
        "tv": np.asarray([0.1, 0.2, 0.4, 0.5, 0.8, 1.0]),
        "gap": np.asarray([0.01, 0.03, 0.02, 0.08, 0.04, 0.10]),
    }
    task_values = np.arange(1, 13, dtype=np.float64).reshape(6, 2, 1)

    diagnostics = _bucket_diagnostics(
        assignments=assignments,
        geometry=geometry,
        method_values={"method": task_values},
    )

    high_tv = diagnostics["tv"]["method"]["high"]
    assert high_tv["count"] == 2
    assert high_tv["representative_task_indices"] == {
        "minimum": 4,
        "median": 5,
        "maximum": 5,
    }
    assert high_tv["final_task_suboptimality_quantiles"][0] == pytest.approx(9.5)
    assert high_tv["final_task_suboptimality_quantiles"][-1] == pytest.approx(11.5)
