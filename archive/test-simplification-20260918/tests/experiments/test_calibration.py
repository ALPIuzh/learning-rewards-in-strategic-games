from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.toy.application import ToyApplication
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl_exp.execution.runner import run_experiment
from mairl_exp.studies.calibration import (
    CalibrationGuardrails,
    CalibrationJob,
    NumericalCalibrationExecutor,
    NumericalCalibrationStudy,
    NumericalCandidate,
    _numeric_boundary_parameters,
)


def test_numerical_calibration_candidates_are_independent_resumable_jobs(experiment_numerics,
        repository_root, tmp_path: Path) -> None:  # fmt: skip
    application = ToyApplication(problem_seed=7)
    candidates = (
        NumericalCandidate(
            name="candidate-a", mle=numerics.mle, warmup=numerics.warmup, active_policy=numerics.active_policy
        ),
        NumericalCandidate(
            name="candidate-b", mle=numerics.mle, warmup=numerics.warmup, active_policy=numerics.active_policy
        ),
    )
    study = NumericalCalibrationStudy(
        candidates=candidates,
        guardrails=CalibrationGuardrails(
            reference_mle=MLEConfig(ridge=1e-4, solver="slsqp"),
            reference_policy=PolicyOptimizationConfig(
                method="grid", grid_resolution=10, grid_max_points=1_000, refine_grid_solution=False
            ),
            max_mle_reference_gap=1e-5,
            max_policy_reference_gap=1e-6,
            max_mle_residual=1e-6,
            max_policy_residual=1e-6,
            minimum_seeds_per_split=1,
        ),
        name="calibration-test",
        calibration_seed_start=11,
        calibration_seed_count=1,
        validation_seed_start=21,
        validation_seed_count=1,
        total_samples=4,
        warmup_budget=0,
        methods=("active",),
        warmup_modes=(False,),
        evaluation_interval=2,
    )
    run_dir = run_experiment(
        application=application,
        study=study,
        numerics=experiment_numerics,
        semantic_config={"study": "calibration-test"},
        resolved_config_yaml="test: calibration\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        command=("pytest",),
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["progress"]["complete"] == 4
    assert [job["spec"]["candidate"] for job in manifest["expected_jobs"]] == [
        "candidate-a",
        "candidate-a",
        "candidate-b",
        "candidate-b",
    ]
    assert all(job["spec"]["warmup_enabled"] is False for job in manifest["expected_jobs"])
    for expected in manifest["expected_jobs"]:
        summary = json.loads((run_dir / "jobs" / expected["job_id"] / "summary.json").read_text())
        assert summary["numerical_candidate"]["name"] == expected["spec"]["candidate"]
        assert summary["guardrails"]["passed"]
        assert summary["final_metrics"]["fisher_transfer_error"] >= 0
    selection = json.loads((run_dir / "calibration_selection.json").read_text())
    assert selection["status"] == "selected"
    assert not selection["application_metrics_used_for_selection"]
    assert not selection["test_seeds_used"]


def test_calibration_rejects_scientific_sweeps_and_numeric_boundaries(experiment_numerics) -> None:

    def candidate(name: str, *, starts: int, ridge: float = 1e-4) -> NumericalCandidate:
        return NumericalCandidate(
            name=name,
            mle=MLEConfig(ridge=ridge),
            warmup=numerics.warmup,
            active_policy=PolicyOptimizationConfig(random_starts=starts),
        )

    candidates = (candidate("low", starts=1), candidate("middle", starts=4), candidate("high", starts=8))
    assert "active_policy.random_starts" in _numeric_boundary_parameters("low", candidates)
    assert "active_policy.random_starts" not in _numeric_boundary_parameters("middle", candidates)

    with pytest.raises(ConfigurationError, match="scientific hyperparameter"):
        NumericalCalibrationStudy(
            candidates=(candidate("ridge-a", starts=1, ridge=1e-4), candidate("ridge-b", starts=1, ridge=1e-3)),
            guardrails=CalibrationGuardrails(
                reference_mle=MLEConfig(ridge=1e-4, solver="slsqp"),
                reference_policy=PolicyOptimizationConfig(method="grid", grid_resolution=10, grid_max_points=1_000),
                max_mle_reference_gap=1e-6,
                max_policy_reference_gap=1e-6,
                max_mle_residual=1e-6,
                max_policy_residual=1e-6,
            ),
        )


def test_calibration_records_candidate_nonconvergence(experiment_numerics, monkeypatch) -> None:
    application = ToyApplication(problem_seed=7)
    candidate = NumericalCandidate(
        name="bad-candidate", mle=numerics.mle, warmup=numerics.warmup, active_policy=numerics.active_policy
    )
    study = NumericalCalibrationStudy(
        candidates=(candidate,),
        guardrails=CalibrationGuardrails(
            reference_mle=MLEConfig(ridge=1e-4, solver="slsqp"),
            reference_policy=PolicyOptimizationConfig(method="grid", grid_resolution=10, grid_max_points=1_000),
            max_mle_reference_gap=1e-6,
            max_policy_reference_gap=1e-6,
            max_mle_residual=1e-6,
            max_policy_residual=1e-6,
            minimum_seeds_per_split=1,
        ),
        calibration_seed_count=1,
        validation_seed_count=1,
        total_samples=2,
        warmup_budget=0,
        methods=("active",),
        warmup_modes=(False,),
    )
    executor = NumericalCalibrationExecutor(study, application.build(experiment_numerics), experiment_numerics)

    def fail(*_args, **_kwargs):
        raise ConvergenceError("deliberate candidate failure")

    monkeypatch.setattr("mairl_exp.studies.comparison.BaselineComparisonExecutor.run", fail)
    result = executor.run(CalibrationJob("bad-candidate", "calibration", "active", False, 1000))

    assert result.arrays == {}
    assert result.summary["candidate_execution"]["status"] == "rejected_nonconvergence"
    assert not result.summary["guardrails"]["passed"]
