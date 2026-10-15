from __future__ import annotations

import numpy as np
import pytest

from experiments.toy.application import ToyApplication
from mairl_exp.studies import BaselineComparisonStudy, StudyJob, StudyJobResult
from mairl_exp.warmup_sweep import WarmupSweepJob, WarmupSweepStudy


@pytest.mark.parametrize("method", ("active", "fixed_random", "random_each_round"))
@pytest.mark.parametrize("warmup_enabled", (False, True))
def test_vectorized_seed_execution_matches_scalar_jobs(
    experiment_numerics,
    method: str,
    warmup_enabled: bool,
) -> None:
    application = ToyApplication(problem_seed=3).build(experiment_numerics)
    numerics = experiment_numerics
    study = BaselineComparisonStudy(
        seed_start=10,
        seed_count=3,
        total_samples=16,
        warmup_budget=12,
        evaluation_interval=4,
        methods=(method,),
        warmup_modes=(warmup_enabled,),
    )
    executor = study.executor(application, numerics)
    jobs = tuple(StudyJob(method, warmup_enabled, seed) for seed in range(10, 13))

    batched = executor.run_many(jobs)
    scalar = tuple(executor.run(job) for job in jobs)

    for batch_result, scalar_result in zip(batched, scalar, strict=True):
        checkpoints = batch_result.arrays["evaluation_sample_counts"]
        assert np.array_equal(checkpoints, scalar_result.arrays["evaluation_sample_counts"])

        def estimates_at_checkpoints(
            result: StudyJobResult,
            evaluation_counts: np.ndarray,
        ) -> np.ndarray:
            positions = np.searchsorted(
                result.arrays["trace_sample_counts"],
                evaluation_counts,
            )
            assert np.array_equal(
                result.arrays["trace_sample_counts"][positions],
                evaluation_counts,
            )
            return result.arrays["omega_history"][positions]

        assert np.allclose(
            estimates_at_checkpoints(batch_result, checkpoints),
            estimates_at_checkpoints(scalar_result, checkpoints),
            atol=2 * numerics.mle.projected_gradient_tolerance,
        )
        assert np.allclose(
            batch_result.arrays["policies"],
            scalar_result.arrays["policies"],
            atol=1e-8,
        )
        assert np.array_equal(
            batch_result.arrays["follower_actions"],
            scalar_result.arrays["follower_actions"],
        )


def test_toy_application_exposes_true_stackelberg_suboptimality(
    experiment_numerics,
) -> None:
    runtime = ToyApplication(problem_seed=3).build(experiment_numerics)
    true_omega = np.asarray(runtime.metadata["true_omega"], dtype=np.float64)
    metrics = runtime.evaluate(np.vstack((np.zeros_like(true_omega), true_omega)), seed=17)

    assert set(metrics) == {
        "coefficient_l2_error",
        "fisher_transfer_error",
        "stackelberg_suboptimality",
    }
    assert metrics["stackelberg_suboptimality"][0] >= 0.0
    assert metrics["stackelberg_suboptimality"][1] == pytest.approx(0.0, abs=1e-9)


def test_vectorized_online_execution_ignores_lambda_warm_without_warmup(
    experiment_numerics,
) -> None:
    application = ToyApplication(problem_seed=3).build(experiment_numerics)
    numerics = experiment_numerics
    jobs = tuple(StudyJob("active", False, seed) for seed in range(10, 13))

    def run(lambda_warm: float):
        study = BaselineComparisonStudy(
            seed_start=10,
            seed_count=3,
            total_samples=6,
            warmup_budget=0,
            evaluation_interval=2,
            lambda_warm=lambda_warm,
            methods=("active",),
            warmup_modes=(False,),
        )
        return study.executor(application, numerics).run_many(jobs)

    small = run(1e-4)
    large = run(10.0)
    for small_result, large_result in zip(small, large, strict=True):
        assert np.array_equal(
            small_result.arrays["follower_actions"],
            large_result.arrays["follower_actions"],
        )
        assert np.allclose(
            small_result.arrays["policies"],
            large_result.arrays["policies"],
            atol=1e-12,
        )
        assert np.allclose(
            small_result.arrays["omega_history"],
            large_result.arrays["omega_history"],
            atol=1e-12,
        )


def test_warmup_sweep_endpoint_is_method_independent(
    experiment_numerics,
) -> None:
    application = ToyApplication(problem_seed=3).build(experiment_numerics)
    numerics = experiment_numerics
    study = WarmupSweepStudy(
        seed_start=10,
        seed_count=1,
        total_samples=24,
        warmup_budgets=(24,),
        evaluation_interval=12,
    )
    executor = study.executor(application, numerics)
    jobs = tuple(
        WarmupSweepJob(method, 24, 10) for method in ("active", "fixed_random", "random_each_round")
    )
    results = tuple(executor.run(job) for job in jobs)

    for result in results[1:]:
        np.testing.assert_array_equal(
            result.arrays["policies"],
            results[0].arrays["policies"],
        )
        np.testing.assert_array_equal(
            result.arrays["follower_actions"],
            results[0].arrays["follower_actions"],
        )
        np.testing.assert_allclose(
            result.arrays["omega_history"],
            results[0].arrays["omega_history"],
        )
    assert all(result.summary["job"]["warmup_budget"] == 24 for result in results)
