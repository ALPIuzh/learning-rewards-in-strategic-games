from __future__ import annotations

import json

import numpy as np

from experiments.zero_sum_comparison.application import ZeroSumToyApplication
from experiments.zero_sum_comparison.directional_transfer import select_directional_tasks
from experiments.zero_sum_comparison.study import (
    ActiveWarmupTuningStudy,
    ZeroSumComparisonJob,
    ZeroSumEstimatorComparisonStudy,
)
from mairl import FixedLeaderOLSConfig


def test_zero_sum_recipes_protect_selection_confirmation_and_final_seeds(composed_study) -> None:
    selection, refinement, confirmation, comparison = composed_study(
        "runs/zero_sum/joint_selection",
        "runs/zero_sum/joint_refinement",
        "runs/zero_sum/joint_confirmation",
        "runs/zero_sum/comparison",
    )

    def seeds(study):
        return set(range(study.seed_start, study.seed_start + study.seed_count))

    assert seeds(refinement) <= seeds(selection)
    assert seeds(selection).isdisjoint(seeds(confirmation))
    assert (seeds(selection) | seeds(confirmation)).isdisjoint(seeds(comparison))
    assert seeds(refinement) == set(
        range(confirmation.selection_seed_start, confirmation.selection_seed_start + confirmation.selection_seed_count)
    )
    candidate = (confirmation.candidate_warmup_budget, confirmation.candidate_multiplier)
    assert candidate in refinement.candidate_pairs
    assert candidate in confirmation.candidate_pairs
    assert (comparison.warmup_budget, comparison.lambda_information_multiplier) in confirmation.candidate_pairs
    for study in (selection, refinement, confirmation):
        assert len(set(study.candidate_pairs)) == len(study.candidate_pairs)
        assert len(study.jobs()) == len(study.candidate_pairs) * study.seed_count
        assert seeds(comparison) == set(
            range(
                study.final_evaluation_seed_start, study.final_evaluation_seed_start + study.final_evaluation_seed_count
            )
        )
    assert sum(warmup == selection.total_samples for warmup, _ in selection.candidate_pairs) == 1
    assert selection.tuning_metric == "fixed_leader_fisher_error"
    assert selection.guardrail_metric == "regularized_qre_leader_l1_error"


def test_zero_sum_application_has_valid_informative_expert(experiment_numerics) -> None:
    runtime = ZeroSumToyApplication().build(experiment_numerics)

    assert runtime.metadata["follower_only_rank"] == runtime.problem.feature_dimension
    assert runtime.metadata["zero_sum_rank"] == runtime.problem.feature_dimension
    assert runtime.metadata["zero_sum_condition_number"] < (runtime.metadata["follower_only_condition_number"])
    assert runtime.expert.max_residual < 1e-10
    assert runtime.expert.min_residual < 1e-10
    assert runtime.fixed_response.residual < 1e-10
    metrics = runtime.evaluate(runtime.true_omega[None, :], seed=0)
    assert metrics["fixed_leader_fisher_error"][0] == 0.0
    assert metrics["regularized_qre_leader_l1_error"][0] < 1e-12


def test_directional_tasks_are_selected_without_estimator_outcomes(experiment_numerics) -> None:
    runtime = ZeroSumToyApplication().build(experiment_numerics)
    selection = select_directional_tasks(runtime, candidate_pool=16, minimum_policy_l1_distance=0.1)

    first_index, second_index = selection.candidate_indices
    assert 0 <= first_index < second_index < 16
    assert selection.tasks.shape == (2, 3, 4)
    assert selection.policies.shape == (2, 3)
    assert selection.fisher_matrices.shape == (2, 3, 3)
    assert selection.controlled_error_direction.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(selection.controlled_error_direction), 1.0)
    assert runtime.true_omega @ selection.controlled_error_direction <= 0
    np.testing.assert_allclose(selection.policy_l1_distance, 2.0, rtol=0.0, atol=1e-12)
    assert selection.generalized_contrast_ratio > 100.0
    assert not np.allclose(selection.fisher_matrices[0], selection.fisher_matrices[1])


def test_tuning_uses_disjoint_splits_and_writes_frozen_selection(run_study) -> None:
    study = ActiveWarmupTuningStudy(
        name="zero-sum-tuning-test",
        selection_seed_start=10,
        selection_seed_count=2,
        confirmation_seed_start=20,
        confirmation_seed_count=2,
        total_samples=24,
        warmup_budgets=(0, 24),
        evaluation_interval=12,
        lambda_information_multiplier=1.0,
        lambda_warm=0.01,
        confidence_delta=0.05,
        confirmation_standard_errors=100.0,
        minimum_seeds_per_split=2,
    )
    run_dir = run_study(
        application=ZeroSumToyApplication(),
        study=study,
        semantic_config={"study": "zero-sum-tuning-test"},
        resolved_config_yaml="test: tuning\n",
        seed_batch_size=2,
    )
    selection = json.loads((run_dir / "warmup_selection.json").read_text())

    assert selection["status"] == "selected"
    assert selection["test_seeds_used"] is False
    assert selection["selection_and_confirmation_seeds_disjoint"] is True
    assert selection["selected_warmup_budget"] in study.warmup_budgets
    assert all(
        item["seeds"] == 2
        for split in ("selection_statistics", "confirmation_statistics")
        for item in selection[split].values()
    )


def test_equal_budget_comparison_runs_both_ols_fits_on_identical_samples(experiment_numerics) -> None:
    runtime = ZeroSumToyApplication().build(experiment_numerics)
    study = ZeroSumEstimatorComparisonStudy(
        ols=FixedLeaderOLSConfig(pseudocount=0.5, smoothing_warning_ratio=0.1),
        name="zero-sum-comparison-test",
        seed_start=30,
        seed_count=2,
        total_samples=6,
        warmup_budget=4,
        evaluation_interval=2,
        lambda_information_multiplier=1.0,
        lambda_warm=0.01,
        confidence_delta=0.05,
    )
    executor = study.executor(runtime, experiment_numerics)
    follower_job = ZeroSumComparisonJob("follower_ols", 30)
    zero_sum_job = ZeroSumComparisonJob("zero_sum_ols", 30)
    follower_result = executor.run(follower_job)
    zero_sum_result = executor.run(zero_sum_job)

    np.testing.assert_array_equal(
        follower_result.arrays["follower_actions"], zero_sum_result.arrays["follower_actions"]
    )
    np.testing.assert_array_equal(follower_result.arrays["evaluation_sample_counts"], np.asarray([2, 4, 6]))
    assert follower_result.summary["samples"] == 6
    assert zero_sum_result.summary["samples"] == 6
    assert follower_result.summary["equation_mode"] == "follower_only"
    assert zero_sum_result.summary["equation_mode"] == "zero_sum"
