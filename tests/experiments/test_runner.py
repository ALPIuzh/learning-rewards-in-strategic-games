from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.toy.application import ToyApplication
from mairl_exp.execution.artifacts import GitProvenance, experiment_identity, sha256_file
from mairl_exp.execution.jobs import StudyJob
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


def test_interrupted_experiment_resumes_completed_jobs(baseline_study, tmp_path, run_study) -> None:
    application = ToyApplication(problem_seed=3)
    study = baseline_study
    semantic = {"application": {"seed": 3}, "study": {"name": study.name}}
    arguments = {
        "application": application,
        "study": study,
        "semantic_config": semantic,
        "resolved_config_yaml": "test: true\n",
    }

    with pytest.raises(KeyboardInterrupt):
        run_study(**arguments, interrupt_after_completed_jobs=1)

    run_dirs = list((tmp_path / study.name).iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    interrupted = json.loads((run_dir / "manifest.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["progress"]["complete"] == 1
    first_job_id = interrupted["expected_jobs"][0]["job_id"]
    first_result = run_dir / "jobs" / first_job_id / "result.npz"
    first_digest = sha256_file(first_result)

    resumed = run_study(**arguments)
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


def test_interrupted_vectorized_run_retains_completed_seed_batch(tmp_path, run_study) -> None:
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
        "semantic_config": {"study": "batch-resume-test"},
        "resolved_config_yaml": "test: batch-resume\n",
        "seed_batch_size": 2,
    }

    with pytest.raises(KeyboardInterrupt):
        run_study(**arguments, interrupt_after_completed_jobs=1)

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

    assert run_study(**arguments) == run_dir
    completed = json.loads((run_dir / "manifest.json").read_text())
    assert completed["progress"]["complete"] == 3
    assert all(sha256_file(run_dir / "jobs" / job_id / "result.npz") == digest for job_id, digest in digests.items())


@pytest.mark.parametrize("mode", ["scalar", "batch", "wrong_count"])
def test_failed_execution_marks_pending_jobs_for_retry(tmp_path, mode, run_study) -> None:
    jobs = (StudyJob("active", False, 1), StudyJob("active", False, 2))

    def fail(*args, **kwargs):
        raise RuntimeError("execution failed")

    executor = SimpleNamespace(run=fail)
    if mode != "scalar":
        executor.batch_key = lambda job: job.method
        executor.run_many = (lambda jobs, **kwargs: ()) if mode == "wrong_count" else fail
    study = SimpleNamespace(name="failure-test", jobs=lambda: jobs, executor=lambda runtime, numerics: executor)
    with pytest.raises(RuntimeError, match=r"wrong result count|execution failed"):
        run_study(
            application=ToyApplication(problem_seed=3),
            study=study,
            semantic_config={"mode": mode},
            resolved_config_yaml="test: failure\n",
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


def test_progress_tick_preserves_counts_without_reading_other_jobs(tmp_path, repository_root, monkeypatch) -> None:
    from pathlib import Path

    from mairl_exp.execution.artifacts import RunStore, job_identity

    jobs = tuple(StudyJob("active", False, seed) for seed in range(3))
    with RunStore(
        tmp_path,
        study_name="progress",
        semantic_config={},
        resolved_config_yaml="{}",
        jobs=jobs,
        repository=repository_root,
        require_clean_git=False,
        resume=False,
        command=("pytest",),
        execution_batch_size=2,
    ) as store:
        store.start_job(jobs[0])
        store.start_job(jobs[1])
        previous = store.progress()
        other_status = store.jobs_path / job_identity(jobs[1]) / "status.json"
        original_read = Path.read_text

        def read_without_other_job(path, *args, **kwargs):
            assert path != other_status, "a heartbeat must not rescan unrelated jobs"
            return original_read(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", read_without_other_job)
            store.update_job_progress(jobs[0], 4, 8)
            progress = store.progress()
        assert all(
            progress[key] == previous[key]
            for key in ("complete", "failed", "running", "pending", "total", "estimated_remaining_seconds")
        )
        assert progress["current_job"] == jobs[0].label
        status = json.loads((store.jobs_path / job_identity(jobs[0]) / "status.json").read_text())
        assert progress["current_job_progress"] == {
            "label": jobs[0].label,
            "completed_steps": 4,
            "total_steps": 8,
            "heartbeat_at": status["updated_at"],
        }
        assert status["status"] == "running"
        progress["current_job_progress"]["completed_steps"] = -1
        assert store.progress()["current_job_progress"]["completed_steps"] == 4
        store.fail_job(jobs[0], RuntimeError("interrupted"))
        with pytest.raises(RuntimeError, match="non-running"):
            store.update_job_progress(jobs[0], 5, 8)
