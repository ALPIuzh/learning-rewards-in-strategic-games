from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.guard.paper.lambda_tuning import ActiveLambdaTuningStudy
from experiments.toy.application import ToyApplication
from mairl.exceptions import ConfigurationError
from mairl_exp.execution.runner import run_experiment
from mairl_exp.studies.confirmation import FrozenJointCandidateConfirmationStudy
from mairl_exp.studies.discovery import JointWarmupLambdaDiscoveryStudy


def test_lambda_tuning_protects_splits_and_writes_per_mode_selection(experiment_numerics,
        repository_root, tmp_path: Path) -> None:  # fmt: skip
    study = ActiveLambdaTuningStudy(
        name="lambda-tuning-test",
        selection_seed_start=30,
        selection_seed_count=2,
        confirmation_seed_start=40,
        confirmation_seed_count=2,
        final_evaluation_seed_start=100,
        final_evaluation_seed_count=2,
        total_samples=14,
        warmup_budget=12,
        lambda_information_multipliers=(1.0, 2.0),
        lambda_warm=0.01,
        confidence_delta=0.05,
        minimum_seeds_per_split=2,
        tuning_metric="coefficient_l2_error",
        guardrail_standard_errors=100.0,
        confirmation_standard_errors=100.0,
        reject_upper_boundary=False,
    )
    run_dir = run_experiment(
        application=ToyApplication(problem_seed=3),
        study=study,
        numerics=experiment_numerics,
        semantic_config={"study": "lambda-tuning-test"},
        resolved_config_yaml="test: lambda-tuning\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        seed_batch_size=2,
        command=("pytest",),
    )
    selection = json.loads((run_dir / "lambda_selection.json").read_text())

    assert study.evaluation_interval == study.total_samples
    assert selection["status"] == "selected"
    assert selection["tuned_parameter"] == "lambda_information_multiplier"
    assert selection["lambda_warm_tuned"] is False
    assert selection["selection_and_confirmation_seeds_disjoint"] is True
    assert selection["candidates_paired_by_seed_within_each_split"] is True
    assert selection["final_evaluation_seeds_protected"] is True
    assert set(selection["decisions"]) == {"warmup", "no_warmup"}
    assert all(
        decision["selected_multiplier"] in study.lambda_information_multipliers
        for decision in selection["decisions"].values()
    )


def test_joint_warmup_lambda_discovery_writes_guarded_shortlist(experiment_numerics,
        repository_root, tmp_path: Path) -> None:  # fmt: skip
    study = JointWarmupLambdaDiscoveryStudy(
        name="joint-discovery-test",
        seed_start=50,
        seed_count=2,
        final_evaluation_seed_start=100,
        final_evaluation_seed_count=2,
        total_samples=14,
        warmup_budgets=(0, 12),
        lambda_information_multipliers=(1.0, 2.0),
        lambda_warm=0.01,
        confidence_delta=0.05,
        reference_warmup_budget=12,
        reference_multiplier=1.0,
        candidate_pairs=((0, 2.0), (12, 1.0)),
        tuning_metric="coefficient_l2_error",
        minimum_seed_count=2,
        guardrail_standard_errors=100.0,
        shortlist_standard_errors=100.0,
    )
    run_dir = run_experiment(
        application=ToyApplication(problem_seed=3),
        study=study,
        numerics=experiment_numerics,
        semantic_config={"study": "joint-discovery-test"},
        resolved_config_yaml="test: joint-discovery\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        seed_batch_size=2,
        command=("pytest",),
    )
    discovery = json.loads((run_dir / "joint_discovery.json").read_text())

    assert discovery["status"] == "discovery_complete"
    assert discovery["final_selection_made"] is False
    assert discovery["lambda_warm_tuned"] is False
    assert discovery["candidates_paired_by_seed"] is True
    assert discovery["final_evaluation_seeds_protected"] is True
    assert discovery["shortlist"]
    assert len(discovery["statistics"]) == 2


def test_fixed_warmup_budget_is_not_reported_as_a_grid_boundary() -> None:
    study = JointWarmupLambdaDiscoveryStudy(
        name="fixed-budget-discovery",
        seed_start=50,
        seed_count=2,
        final_evaluation_seed_start=100,
        final_evaluation_seed_count=2,
        total_samples=14,
        warmup_budgets=(0,),
        lambda_information_multipliers=(1.0, 2.0, 4.0),
        lambda_warm=0.01,
        confidence_delta=0.05,
        reference_warmup_budget=0,
        reference_multiplier=1.0,
        minimum_seed_count=2,
    )
    statistics = {
        "warmup=0,lambda=2": {
            "stackelberg_suboptimality": {"mean": 0.1, "median": 0.1, "standard_error": 0.01, "seeds": 2}
        }
    }

    summary = study._candidate_summary("warmup=0,lambda=2", statistics)

    assert summary["on_grid_boundary"] is False


def test_frozen_joint_candidate_confirmation_never_reselects(experiment_numerics,
        repository_root, tmp_path: Path) -> None:  # fmt: skip
    study = FrozenJointCandidateConfirmationStudy(
        name="joint-confirmation-test",
        seed_start=50,
        seed_count=2,
        final_evaluation_seed_start=100,
        final_evaluation_seed_count=2,
        total_samples=14,
        warmup_budgets=(0, 12),
        lambda_information_multipliers=(1.0, 2.0),
        lambda_warm=0.01,
        confidence_delta=0.05,
        reference_warmup_budget=0,
        reference_multiplier=1.0,
        candidate_warmup_budget=12,
        candidate_multiplier=2.0,
        candidate_pairs=((0, 1.0), (12, 2.0)),
        selection_run_id="selection-test",
        selection_seed_start=10,
        selection_seed_count=2,
        tuning_metric="coefficient_l2_error",
        minimum_seed_count=2,
        confirmation_standard_errors=100.0,
    )
    run_dir = run_experiment(
        application=ToyApplication(problem_seed=3),
        study=study,
        numerics=experiment_numerics,
        semantic_config={"study": "joint-confirmation-test"},
        resolved_config_yaml="test: joint-confirmation\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        seed_batch_size=2,
        command=("pytest",),
    )
    confirmation = json.loads((run_dir / "confirmation.json").read_text())

    assert confirmation["status"] == "confirmed"
    assert confirmation["candidate_frozen_before_confirmation"] is True
    assert confirmation["reselection_performed"] is False
    assert confirmation["selection_run_id"] == "selection-test"
    assert confirmation["selection_and_confirmation_seeds_disjoint"] is True


def test_frozen_joint_confirmation_rejects_selection_seed_reuse() -> None:
    with pytest.raises(ConfigurationError, match="selection and confirmation seeds overlap"):
        FrozenJointCandidateConfirmationStudy(
            name="overlapping-confirmation-test",
            seed_start=50,
            seed_count=2,
            final_evaluation_seed_start=100,
            final_evaluation_seed_count=2,
            total_samples=14,
            warmup_budgets=(0, 12),
            lambda_information_multipliers=(1.0, 2.0),
            lambda_warm=0.01,
            confidence_delta=0.05,
            reference_warmup_budget=0,
            reference_multiplier=1.0,
            candidate_warmup_budget=12,
            candidate_multiplier=2.0,
            candidate_pairs=((0, 1.0), (12, 2.0)),
            selection_run_id="selection-test",
            selection_seed_start=50,
            selection_seed_count=2,
            tuning_metric="coefficient_l2_error",
            minimum_seed_count=2,
        )


def test_lambda_tuning_rejects_final_evaluation_seed_leakage() -> None:
    with pytest.raises(ConfigurationError, match="protected final evaluation"):
        ActiveLambdaTuningStudy(
            name="leaking-lambda-tuning",
            selection_seed_start=100,
            selection_seed_count=2,
            confirmation_seed_start=200,
            confirmation_seed_count=2,
            final_evaluation_seed_start=100,
            final_evaluation_seed_count=50,
            total_samples=14,
            warmup_budget=12,
            lambda_information_multipliers=(1.0, 2.0),
            lambda_warm=0.01,
            confidence_delta=0.05,
            minimum_seeds_per_split=2,
        )


def test_lambda_selection_excludes_source_winner_that_breaks_fisher_guardrail() -> None:
    study = ActiveLambdaTuningStudy(
        name="guarded-lambda-selection",
        selection_seed_start=10,
        selection_seed_count=2,
        confirmation_seed_start=20,
        confirmation_seed_count=2,
        final_evaluation_seed_start=100,
        final_evaluation_seed_count=50,
        total_samples=14,
        warmup_budget=12,
        lambda_information_multipliers=(1.0, 2.0, 4.0),
        lambda_warm=0.01,
        confidence_delta=0.05,
        minimum_seeds_per_split=2,
        warmup_modes=(True,),
    )

    def split(primary: tuple[tuple[float, float], ...], fisher: tuple[tuple[float, float], ...]):
        return {
            multiplier: {
                "stackelberg_suboptimality": dict(zip((10, 11), primary[index], strict=True)),
                "fisher_transfer_error": dict(zip((10, 11), fisher[index], strict=True)),
            }
            for index, multiplier in enumerate(study.lambda_information_multipliers)
        }

    observations = {
        "selection": split(((0.30, 0.30), (0.19, 0.21), (0.09, 0.11)), ((1.00, 1.00), (0.99, 1.01), (2.00, 2.00))),
        "confirmation": split(((0.31, 0.31), (0.20, 0.22), (0.10, 0.12)), ((1.00, 1.00), (0.99, 1.01), (2.00, 2.00))),
    }
    statistics = study._statistics({"warmup": observations})["warmup"]
    decision = study._select_mode(True, statistics, observations)

    assert decision["status"] == "selected"
    assert decision["raw_selection_winner"] == 2.0
    assert decision["selected_multiplier"] == 2.0
    assert decision["fisher_guardrail_eligible"] == [1.0, 2.0]
