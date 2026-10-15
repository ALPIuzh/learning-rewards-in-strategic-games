from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.toy.application import ToyApplication
from mairl_exp.execution.artifacts import GitProvenance, experiment_identity, sha256_file
from mairl_exp.execution.jobs import StudyJob
from mairl_exp.execution.runner import run_experiment
from mairl_exp.studies.comparison import BaselineComparisonStudy


def test_experiment_identity_changes_with_config_and_git() -> None:
    clean = GitProvenance(
        commit="abc", branch="main", dirty=False, status=(), patch=b"", untracked=(), fingerprint="code-a"
    )
    changed = GitProvenance(
        commit="def", branch="main", dirty=False, status=(), patch=b"", untracked=(), fingerprint="code-b"
    )

    run_a, config_a = experiment_identity({"ridge": 1e-4}, clean)
    run_b, config_b = experiment_identity({"ridge": 1e-3}, clean)
    run_c, config_c = experiment_identity({"ridge": 1e-4}, changed)

    assert len({run_a, run_b, run_c}) == 3
    assert config_a == config_c
    assert config_a != config_b


def test_interrupted_experiment_resumes_completed_jobs(experiment_numerics,
        baseline_study, repository_root, tmp_path: Path) -> None:  # fmt: skip
    application = ToyApplication(problem_seed=3)
    study = baseline_study
    semantic = {"application": {"seed": 3}, "study": {"name": study.name}}
    arguments = {
        "application": application,
        "study": study,
        "numerics": experiment_numerics,
        "semantic_config": semantic,
        "resolved_config_yaml": "test: true\n",
        "output_root": tmp_path,
        "original_cwd": repository_root,
        "require_clean_git": False,
        "resume": True,
        "command": ("pytest",),
    }

    with pytest.raises(KeyboardInterrupt):
        run_experiment(**arguments, interrupt_after_completed_jobs=1)

    run_dirs = list((tmp_path / study.name).iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    interrupted = json.loads((run_dir / "manifest.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["progress"]["complete"] == 1
    first_job_id = interrupted["expected_jobs"][0]["job_id"]
    first_result = run_dir / "jobs" / first_job_id / "result.npz"
    first_digest = sha256_file(first_result)

    resumed = run_experiment(**arguments)
    assert resumed == run_dir
    completed = json.loads((run_dir / "manifest.json").read_text())
    assert completed["status"] == "complete"
    assert completed["progress"]["complete"] == 2
    assert completed["progress"]["failed"] == 0
    assert sha256_file(first_result) == first_digest

    with np.load(first_result, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["evaluation_sample_counts"], np.array([0, 2, 4]))
        assert "metric__fisher_transfer_error" in arrays
        assert "warmup__design_weights" in arrays
        assert "warmup__allocation_counts" in arrays
        assert "warmup__coverage_eigenvalues" in arrays


def test_interrupted_vectorized_run_retains_completed_seed_batch(experiment_numerics,
        repository_root, tmp_path: Path) -> None:  # fmt: skip
    application = ToyApplication(problem_seed=3)
    study = BaselineComparisonStudy(
        name="batch-resume-test",
        seed_start=20,
        seed_count=3,
        total_samples=4,
        warmup_budget=0,
        evaluation_interval=2,
        methods=("random_each_round",),
        warmup_modes=(False,),
    )
    arguments = {
        "application": application,
        "study": study,
        "numerics": experiment_numerics,
        "semantic_config": {"study": "batch-resume-test"},
        "resolved_config_yaml": "test: batch-resume\n",
        "output_root": tmp_path,
        "original_cwd": repository_root,
        "require_clean_git": False,
        "resume": True,
        "seed_batch_size": 2,
        "command": ("pytest",),
    }

    with pytest.raises(KeyboardInterrupt):
        run_experiment(**arguments, interrupt_after_completed_jobs=1)

    run_dir = next((tmp_path / study.name).iterdir())
    interrupted = json.loads((run_dir / "manifest.json").read_text())
    assert interrupted["progress"]["complete"] == 2
    assert interrupted["execution_batch_size"] == 2
    completed_ids = []
    for expected in interrupted["expected_jobs"]:
        status_path = run_dir / "jobs" / expected["job_id"] / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())["status"] == "complete":
            completed_ids.append(expected["job_id"])
    digests = {job_id: sha256_file(run_dir / "jobs" / job_id / "result.npz") for job_id in completed_ids}

    assert run_experiment(**arguments) == run_dir
    completed = json.loads((run_dir / "manifest.json").read_text())
    assert completed["progress"]["complete"] == 3
    assert all(sha256_file(run_dir / "jobs" / job_id / "result.npz") == digest for job_id, digest in digests.items())


@pytest.mark.parametrize("mode", ["scalar", "batch", "wrong_count"])
def test_failed_execution_marks_pending_jobs_for_retry(experiment_numerics, repository_root, tmp_path, mode):
    jobs = (StudyJob("active", False, 1), StudyJob("active", False, 2))

    def fail(*args, **kwargs):
        raise RuntimeError("execution failed")

    executor = SimpleNamespace(run=fail)
    if mode != "scalar":
        executor.batch_key = lambda job: job.method
        executor.run_many = (lambda jobs, **kwargs: ()) if mode == "wrong_count" else fail
    study = SimpleNamespace(name="failure-test", jobs=lambda: jobs, executor=lambda runtime, numerics: executor)
    with pytest.raises(RuntimeError, match=r"wrong result count|execution failed"):
        run_experiment(
            application=ToyApplication(problem_seed=3),
            study=study,
            numerics=experiment_numerics,
            semantic_config={"mode": mode},
            resolved_config_yaml="test: failure\n",
            output_root=tmp_path,
            original_cwd=repository_root,
            require_clean_git=False,
            resume=True,
        )
    run_dir = next((tmp_path / study.name).iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["progress"]["failed"] == (1 if mode == "scalar" else 2)
    assert manifest["progress"]["complete"] == 0
    assert not list(run_dir.rglob("result.npz"))


@pytest.mark.parametrize("format", ["bytes", "npz"])
def test_failed_atomic_write_preserves_previous_result(tmp_path, monkeypatch, format) -> None:
    from mairl_exp.execution import artifacts

    path = tmp_path / "result"
    path.write_bytes(b"previous result")

    def fail_replace(*_):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        if format == "bytes":
            artifacts.atomic_write_bytes(path, b"new result")
        else:
            artifacts.atomic_write_npz(path, {"values": np.arange(3)})
    assert path.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [path]
