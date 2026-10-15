from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.sparse_information.application import SparseInformationApplication, make_sparse_information_problem
from experiments.sparse_information.export_sweep import _expected_specs, _verified_result_path
from experiments.sparse_information.study import SparseInformationJob, SparseInformationSweepStudy
from mairl.core.math import fisher_information
from mairl.exceptions import ConfigurationError
from mairl.exploration.policy import select_information_policy
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.config import ExperimentNumerics, WarmupNumerics
from mairl_exp.execution.artifacts import sha256_file


def _numerics() -> ExperimentNumerics:
    policy = PolicyOptimizationConfig(method="multistart", random_starts=0, projected_gradient_tolerance=1e-6)
    return ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4), warmup=WarmupNumerics(), active_policy=policy, evaluation_policy=policy
    )


def test_sparse_information_game_has_exact_initial_information_ratio() -> None:
    leader_actions = 16
    problem, _, _ = make_sparse_information_problem(leader_actions=leader_actions)
    zero = np.zeros(1)
    informative = np.eye(leader_actions)[0]
    decoy = np.eye(leader_actions)[1]

    informative_fisher = fisher_information(problem, informative, zero)[0, 0]
    decoy_fisher = fisher_information(problem, decoy, zero)[0, 0]
    expected_random_fisher = informative_fisher * 2 / (leader_actions * (leader_actions + 1))

    assert informative_fisher == 1.0
    assert decoy_fisher == 0.0
    assert informative_fisher / expected_random_fisher == 136.0


@pytest.mark.parametrize("leader_actions", [2, 4, 8, 16, 32])
def test_swept_sparse_instances_have_analytical_ratio(leader_actions: int) -> None:
    runtime = SparseInformationApplication(leader_actions=leader_actions).build(_numerics())
    assert runtime.problem.n_leader_actions == leader_actions
    assert runtime.metadata["initial_active_to_random_fisher_ratio"] == (leader_actions * (leader_actions + 1) / 2)


def test_sparse_sweep_has_exact_paired_job_grid() -> None:
    study = SparseInformationSweepStudy()
    jobs = study.jobs()
    assert len(jobs) == 5 * 3 * 100
    assert jobs[0] == SparseInformationJob(2, "active", 72000)
    assert jobs[-1] == SparseInformationJob(32, "random_each_round", 72099)
    for m in (2, 4, 8, 16, 32):
        for method in study.methods:
            assert [job.seed for job in jobs if job.leader_actions == m and job.method == method] == list(
                range(72000, 72100)
            )
    assert all(not job.warmup_enabled for job in jobs)
    assert [job.as_dict() for job in jobs] == _expected_specs()


def test_sparse_export_verifies_job_status_and_checksum(tmp_path) -> None:
    job_id = "example"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    result_path = job_dir / "result.npz"
    np.savez_compressed(result_path, values=np.arange(3))
    digest = sha256_file(result_path)
    (job_dir / "status.json").write_text(json.dumps({"status": "complete", "result_sha256": digest}))

    assert _verified_result_path(tmp_path, {"job_id": job_id}) == (job_id, result_path, digest)
    (job_dir / "status.json").write_text(json.dumps({"status": "complete", "result_sha256": "bad"}))
    with pytest.raises(ValueError, match="checksum"):
        _verified_result_path(tmp_path, {"job_id": job_id})


def test_sparse_sweep_rejects_changed_grid_and_invalid_job() -> None:
    with pytest.raises(ConfigurationError, match="exactly"):
        SparseInformationSweepStudy(leader_action_counts=(2, 4, 8, 16, 32, 64))
    study = SparseInformationSweepStudy(seed_count=1, total_samples=2, evaluation_interval=1)
    executor = study.executor(SparseInformationApplication().build(_numerics()), _numerics())
    with pytest.raises(ConfigurationError, match="not part"):
        executor.batch_key(SparseInformationJob(16, "active", 999))


def test_swept_jobs_have_finite_shapes_and_exact_sample_accounting() -> None:
    study = SparseInformationSweepStudy(
        seed_start=7, seed_count=1, total_samples=2, evaluation_interval=1, methods=("random_each_round",)
    )
    executor = study.executor(SparseInformationApplication().build(_numerics()), _numerics())
    results = [executor.run(job) for job in study.jobs()]
    for job, result in zip(study.jobs(), results, strict=True):
        assert result.summary["samples"] == 2
        assert result.arrays["policies"].shape == (2, job.leader_actions)
        assert result.arrays["follower_actions"].shape == (2,)
        np.testing.assert_array_equal(result.arrays["evaluation_sample_counts"], [0, 1, 2])
        assert all(np.all(np.isfinite(array)) for array in result.arrays.values())


def test_active_selection_finds_the_only_informative_action() -> None:
    problem, _, _ = make_sparse_information_problem()
    selection = select_information_policy(
        problem, np.zeros(1), 16.01 * np.eye(1), _numerics().active_policy, np.random.default_rng(4)
    )

    np.testing.assert_allclose(selection.policy, np.eye(16)[0], atol=1e-8)


def test_reward_estimate_changes_the_deployed_leader_policy() -> None:
    problem, true_omega, leader_reward = make_sparse_information_problem()
    config = LeaderOptimizationConfig(_numerics().evaluation_policy)
    true_solution = optimize_leader_policy(problem, true_omega, leader_reward, config)
    initial_solution = optimize_leader_policy(problem, np.zeros(1), leader_reward, config)

    assert true_solution.policy[0] > 1 - 1e-8
    assert initial_solution.policy[0] < 1e-8
    assert true_solution.value > initial_solution.value


def test_sparse_information_application_exposes_all_metrics() -> None:
    runtime = SparseInformationApplication().build(_numerics())
    metrics = runtime.evaluate(np.asarray([[0.0], [1.0]]), seed=10)

    assert set(metrics) == {"coefficient_l2_error", "fisher_transfer_error", "stackelberg_suboptimality"}
    assert all(values.shape == (2,) for values in metrics.values())
    assert metrics["stackelberg_suboptimality"][0] > 0
    assert metrics["stackelberg_suboptimality"][1] < 1e-12
