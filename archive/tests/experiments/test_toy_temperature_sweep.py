from __future__ import annotations

import numpy as np
import pytest

from experiments.toy.application import ToyApplication
from experiments.toy.temperature_sweep import (
    ToyTemperatureJob,
    ToyTemperatureSweepStudy,
)
from mairl.exceptions import ConfigurationError


def test_temperature_study_has_exact_paired_grid() -> None:
    study = ToyTemperatureSweepStudy()
    jobs = study.jobs()
    assert len(jobs) == 5 * 3 * 100
    assert jobs[0] == ToyTemperatureJob(0.5, "active", 83_000)
    assert jobs[-1] == ToyTemperatureJob(3.0, "random_each_round", 83_099)
    for eta in study.eta_values:
        for method in study.methods:
            assert [job.seed for job in jobs if job.eta == eta and job.method == method] == list(
                range(83_000, 83_100)
            )


@pytest.mark.parametrize(
    "eta_values",
    [(), (0.0, 1.25), (1.25, 1.25), (1.25, 0.75), (0.5, 2.0)],
)
def test_temperature_study_rejects_ambiguous_grids(eta_values: tuple[float, ...]) -> None:
    with pytest.raises(ConfigurationError):
        ToyTemperatureSweepStudy(eta_values=eta_values)


def test_temperature_executor_changes_only_eta_and_preserves_lambda_rule(
    experiment_numerics,
) -> None:
    numerics = experiment_numerics
    runtime = ToyApplication(problem_seed=3).build(numerics)
    study = ToyTemperatureSweepStudy(
        seed_start=10,
        seed_count=2,
        eta_values=(0.5, 1.25, 2.0),
        total_samples=4,
        warmup_budget=2,
        evaluation_interval=2,
        methods=("fixed_random",),
    )
    executor = study.executor(runtime, numerics)

    for eta in study.eta_values:
        delegate = executor._delegate(eta)
        assert delegate.application.problem.eta == eta
        assert delegate.study.lambda_information_multiplier == 1.0
        np.testing.assert_allclose(
            delegate.application.problem.features,
            runtime.problem.features,
        )
        assert delegate.application.metadata["true_omega"] == runtime.metadata["true_omega"]
        assert delegate.application.metadata["leader_reward"] == runtime.metadata["leader_reward"]


def test_temperature_jobs_vectorize_without_changing_seeded_results(
    experiment_numerics,
) -> None:
    numerics = experiment_numerics
    runtime = ToyApplication(problem_seed=3).build(numerics)
    study = ToyTemperatureSweepStudy(
        seed_start=10,
        seed_count=2,
        eta_values=(1.25,),
        total_samples=14,
        warmup_budget=12,
        evaluation_interval=2,
        methods=("fixed_random",),
    )
    jobs = study.jobs()
    batched = study.executor(runtime, numerics).run_many(jobs)
    scalar_executor = study.executor(runtime, numerics)
    scalar = tuple(scalar_executor.run(job) for job in jobs)

    for batched_result, scalar_result in zip(batched, scalar, strict=True):
        np.testing.assert_array_equal(
            batched_result.arrays["follower_actions"],
            scalar_result.arrays["follower_actions"],
        )
        checkpoints = batched_result.arrays["evaluation_sample_counts"]
        batch_positions = np.searchsorted(
            batched_result.arrays["trace_sample_counts"],
            checkpoints,
        )
        scalar_positions = np.searchsorted(
            scalar_result.arrays["trace_sample_counts"],
            checkpoints,
        )
        np.testing.assert_array_equal(
            batched_result.arrays["trace_sample_counts"][batch_positions],
            checkpoints,
        )
        np.testing.assert_array_equal(
            scalar_result.arrays["trace_sample_counts"][scalar_positions],
            checkpoints,
        )
        np.testing.assert_allclose(
            batched_result.arrays["omega_history"][batch_positions],
            scalar_result.arrays["omega_history"][scalar_positions],
            atol=2 * numerics.mle.projected_gradient_tolerance,
        )
