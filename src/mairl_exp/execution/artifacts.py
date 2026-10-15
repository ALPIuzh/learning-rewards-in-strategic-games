"""Content-addressed experiment artifacts and Git provenance."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

import numpy as np

from mairl.exceptions import ConfigurationError
from mairl_exp.execution.jobs import ExperimentJob, StudyJobResult

RUN_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: object) -> str:
    """Serialize identity-bearing data deterministically and reject NaNs."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _atomic_stream(path: Path) -> Iterator[BinaryIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    with _atomic_stream(path) as stream:
        stream.write(content)


def atomic_write_text(path: Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with _atomic_stream(path) as stream:
        np.savez_compressed(stream, **arrays)


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(("git", *arguments), cwd=repository, check=check, capture_output=True)


def find_repository(start: Path) -> Path:
    result = _git(start, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise ConfigurationError(f"{start} is not inside a Git repository")
    return Path(result.stdout.decode().strip()).resolve()


@dataclass(frozen=True, slots=True)
class GitProvenance:
    commit: str
    branch: str
    dirty: bool
    status: tuple[str, ...]
    patch: bytes
    untracked: tuple[tuple[str, str], ...]
    fingerprint: str

    def manifest(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "status": list(self.status),
            "untracked": [{"path": path, "sha256": digest} for path, digest in self.untracked],
            "fingerprint": self.fingerprint,
        }


def collect_git_provenance(repository: Path) -> GitProvenance:
    """Fingerprint committed, modified, staged, and untracked repository state."""

    commit = _git(repository, "rev-parse", "HEAD").stdout.decode().strip()
    branch_result = _git(repository, "branch", "--show-current", check=False)
    branch = branch_result.stdout.decode().strip() or "detached"
    status_bytes = _git(repository, "status", "--porcelain=v1", "--untracked-files=all", "-z").stdout
    status_lines = tuple(os.fsdecode(line) for line in status_bytes.split(b"\0") if line)
    patch = _git(repository, "diff", "--binary", "HEAD").stdout

    untracked: list[tuple[str, str]] = []
    untracked_content = hashlib.sha256()
    untracked_paths = _git(repository, "ls-files", "--others", "--exclude-standard", "-z").stdout
    for encoded_relative in untracked_paths.split(b"\0"):
        if not encoded_relative:
            continue
        relative = os.fsdecode(encoded_relative)
        path = repository / relative
        if not path.is_file():
            continue
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
            digest = sha256_bytes(content)
        else:
            digest = sha256_file(path)
        untracked.append((relative, digest))
        untracked_content.update(encoded_relative)
        untracked_content.update(b"\0")
        untracked_content.update(bytes.fromhex(digest))

    identity = hashlib.sha256()
    identity.update(commit.encode())
    identity.update(b"\0")
    identity.update(status_bytes)
    identity.update(b"\0")
    identity.update(patch)
    identity.update(b"\0")
    identity.update(untracked_content.digest())
    return GitProvenance(
        commit=commit,
        branch=branch,
        dirty=bool(status_lines),
        status=status_lines,
        patch=patch,
        untracked=tuple(untracked),
        fingerprint=identity.hexdigest(),
    )


def environment_manifest(repository: Path) -> dict[str, object]:
    packages = {}
    for name in ("mairl", "hydra-core", "omegaconf", "numpy", "scipy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    lockfile = repository / "uv.lock"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(lockfile) if lockfile.exists() else None,
    }


def experiment_identity(semantic_config: object, git: GitProvenance) -> tuple[str, str]:
    config_digest = sha256_bytes(canonical_json(semantic_config).encode())
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "semantic_config_sha256": config_digest,
        "git_fingerprint": git.fingerprint,
    }
    return sha256_bytes(canonical_json(identity).encode())[:16], config_digest


def job_identity(job: ExperimentJob) -> str:
    return sha256_bytes(canonical_json(job.as_dict()).encode())[:16]


class RunStore:
    """Single-writer artifact store that safely resumes completed study jobs."""

    def __init__(self, root: Path, *, study_name: str, semantic_config: object, resolved_config_yaml: str,
            jobs: tuple[ExperimentJob, ...], repository: Path, require_clean_git: bool,
            resume: bool, command: tuple[str, ...], execution_batch_size: int) -> None:  # fmt: skip
        self.repository = repository
        self.git = collect_git_provenance(repository)
        if require_clean_git and self.git.dirty:
            changed = "\n".join(f"  {line}" for line in self.git.status)
            raise ConfigurationError(
                "refusing to launch a versioned experiment from a dirty Git worktree. "
                "Commit the changes, or explicitly set run.require_clean_git=false.\n"
                f"{changed}"
            )
        self.run_id, config_digest = experiment_identity(semantic_config, self.git)
        self.path = root / _safe_name(study_name) / self.run_id
        self.jobs_path = self.path / "jobs"
        self._lock_stream: BinaryIO | None = None
        self._started = perf_counter()
        self._acquire_lock()

        expected_jobs = [{"job_id": job_identity(job), "spec": job.as_dict(), "label": job.label} for job in jobs]
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            if not resume:
                self.close()
                raise ConfigurationError(f"run {self.run_id} already exists; enable resume or change configuration")
            manifest = json.loads(manifest_path.read_text())
            if manifest["semantic_config_sha256"] != config_digest:
                self.close()
                raise ConfigurationError("existing run has a different resolved configuration")
            if manifest["git"]["fingerprint"] != self.git.fingerprint:
                self.close()
                raise ConfigurationError("existing run was produced from different Git state")
            if manifest["expected_jobs"] != expected_jobs:
                self.close()
                raise ConfigurationError("existing run has a different execution plan")
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.jobs_path.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path / "resolved_config.yaml", resolved_config_yaml)
            atomic_write_json(self.path / "semantic_config.json", semantic_config)
            atomic_write_json(self.path / "provenance.json", environment_manifest(repository))
            if self.git.patch:
                atomic_write_bytes(self.path / "git.patch", self.git.patch)
            self._copy_untracked_sources()
            manifest = {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": self.run_id,
                "study_name": study_name,
                "status": "created",
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "semantic_config_sha256": config_digest,
                "git": self.git.manifest(),
                "command": list(command),
                "resume_granularity": "job",
                "execution_batch_size": execution_batch_size,
                "expected_jobs": expected_jobs,
                "progress": {},
            }
            atomic_write_json(manifest_path, manifest)
        self._manifest = manifest
        self._refresh_manifest(status="running")

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, exception_type, exception, _) -> None:
        if exception_type is None:
            self._refresh_manifest(status="complete")
        elif exception_type is KeyboardInterrupt:
            self._refresh_manifest(status="interrupted")
        else:
            self._refresh_manifest(status="failed", error=str(exception))
        self.close()

    def close(self) -> None:
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def is_complete(self, job: ExperimentJob) -> bool:
        status_path = self._job_path(job) / "status.json"
        result_path = self._job_path(job) / "result.npz"
        if not status_path.exists() or not result_path.exists():
            return False
        status = json.loads(status_path.read_text())
        return status.get("status") == "complete" and status.get("result_sha256") == sha256_file(result_path)

    def start_job(self, job: ExperimentJob) -> None:
        job_path = self._job_path(job)
        job_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            job_path / "status.json",
            {
                "job_id": job_identity(job),
                "label": job.label,
                "spec": job.as_dict(),
                "status": "running",
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
            },
        )
        self._refresh_manifest(status="running", current_job=job.label)

    def complete_job(self, job: ExperimentJob, result: StudyJobResult) -> None:
        job_path = self._job_path(job)
        result_path = job_path / "result.npz"
        atomic_write_npz(result_path, result.arrays)
        atomic_write_json(job_path / "summary.json", result.summary)
        atomic_write_json(
            job_path / "status.json",
            {
                "job_id": job_identity(job),
                "label": job.label,
                "spec": job.as_dict(),
                "status": "complete",
                "completed_at": _utc_now(),
                "updated_at": _utc_now(),
                "algorithm_seconds": result.summary["algorithm_seconds"],
                "result_sha256": sha256_file(result_path),
            },
        )
        self._refresh_manifest(status="running", current_job=None)

    def update_job_progress(self, job: ExperimentJob, completed_steps: int, total_steps: int) -> None:
        status_path = self._job_path(job) / "status.json"
        status = json.loads(status_path.read_text())
        if status.get("status") != "running":
            raise RuntimeError(f"cannot update progress for non-running job {job.label!r}")
        status.update(
            {"completed_steps": int(completed_steps), "total_steps": int(total_steps), "updated_at": _utc_now()}
        )
        atomic_write_json(status_path, status)
        # A progress tick changes no job counts or completion times. Those are
        # recomputed on start/complete/fail, so only refresh the live heartbeat here.
        path = self.path / "manifest.json"
        manifest = self._manifest
        manifest["status"] = "running"
        manifest["updated_at"] = status["updated_at"]
        manifest["progress"].update(
            current_job=job.label,
            current_job_progress={
                "label": job.label,
                "completed_steps": int(completed_steps),
                "total_steps": int(total_steps),
                "heartbeat_at": status["updated_at"],
            },
            elapsed_seconds_this_session=perf_counter() - self._started,
        )
        atomic_write_json(path, manifest)

    def fail_job(self, job: ExperimentJob, error: BaseException) -> None:
        atomic_write_json(
            self._job_path(job) / "status.json",
            {
                "job_id": job_identity(job),
                "label": job.label,
                "spec": job.as_dict(),
                "status": "failed",
                "failed_at": _utc_now(),
                "updated_at": _utc_now(),
                "error": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            },
        )
        self._refresh_manifest(status="failed", current_job=None, error=str(error))

    def progress(self) -> dict[str, object]:
        return deepcopy(self._manifest["progress"])

    def _job_path(self, job: ExperimentJob) -> Path:
        return self.jobs_path / job_identity(job)

    def _acquire_lock(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        lock_path = self.path / ".lock"
        self._lock_stream = lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_stream.close()
            self._lock_stream = None
            raise ConfigurationError(f"run directory is already active: {self.path}") from error

    def _copy_untracked_sources(self) -> None:
        for relative, _ in self.git.untracked:
            source = self.repository / relative
            target = self.path / "untracked_sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)

    def _refresh_manifest(self, *, status: str, current_job: str | None = None, error: str | None = None) -> None:
        path = self.path / "manifest.json"
        manifest = self._manifest
        states = {"complete": 0, "failed": 0, "running": 0, "pending": 0}
        completed_seconds = []
        current_progress = None
        for expected in manifest["expected_jobs"]:
            job_status_path = self.jobs_path / expected["job_id"] / "status.json"
            if not job_status_path.exists():
                states["pending"] += 1
                continue
            job_status = json.loads(job_status_path.read_text())
            state = job_status.get("status", "pending")
            if state not in states:
                state = "pending"
            states[state] += 1
            if state == "complete":
                completed_seconds.append(float(job_status.get("algorithm_seconds", 0.0)))
            elif state == "running":
                current_progress = {
                    "label": expected["label"],
                    "completed_steps": job_status.get("completed_steps"),
                    "total_steps": job_status.get("total_steps"),
                    "heartbeat_at": job_status.get("updated_at"),
                }
        remaining = states["pending"] + states["failed"] + states["running"]
        eta_seconds = float(np.mean(completed_seconds)) * remaining if completed_seconds else None
        manifest["status"] = status
        manifest["updated_at"] = _utc_now()
        manifest["progress"] = {
            **states,
            "total": len(manifest["expected_jobs"]),
            "current_job": current_job,
            "current_job_progress": current_progress,
            "elapsed_seconds_this_session": perf_counter() - self._started,
            "estimated_remaining_seconds": eta_seconds,
        }
        if error is not None:
            manifest["last_error"] = error
        elif status == "complete":
            manifest.pop("last_error", None)
        atomic_write_json(path, manifest)


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    safe = safe.strip("-")
    if not safe:
        raise ConfigurationError("study name does not contain a filesystem-safe character")
    return safe
