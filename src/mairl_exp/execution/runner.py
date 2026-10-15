"""Execute composed studies with durable job-level resumption."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from time import perf_counter

from mairl_exp.application import ExperimentApplication
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.artifacts import RunStore, find_repository
from mairl_exp.execution.jobs import ExperimentJob


def run_experiment(*, application: ExperimentApplication, study, numerics: ExperimentNumerics,
        semantic_config: object, resolved_config_yaml: str, output_root: Path, original_cwd: Path,
        require_clean_git: bool, resume: bool, seed_batch_size: int = 64, command: tuple[str, ...] | None = None,
        interrupt_after_completed_jobs: int | None = None) -> Path:  # fmt: skip
    """Execute pending jobs and return the content-addressed run directory.

    ``interrupt_after_completed_jobs`` exists for deterministic interruption tests; the
    public Hydra configuration does not expose it.
    """

    if seed_batch_size < 1:
        raise ValueError("seed_batch_size must be positive")
    repository = find_repository(original_cwd)
    root = output_root if output_root.is_absolute() else original_cwd / output_root
    runtime = application.build(numerics)
    executor = study.executor(runtime, numerics)
    jobs = study.jobs()
    run_many = getattr(executor, "run_many", None)
    batch_key = getattr(executor, "batch_key", None)
    batched = callable(run_many) and callable(batch_key)
    command = command or tuple(sys.argv)

    with RunStore(
        root.resolve(),
        study_name=study.name,
        semantic_config=semantic_config,
        resolved_config_yaml=resolved_config_yaml,
        jobs=jobs,
        repository=repository,
        require_clean_git=require_clean_git,
        resume=resume,
        command=command,
        execution_batch_size=seed_batch_size,
    ) as store:
        print(f"run_id={store.run_id}")
        print(f"run_dir={store.path}")
        completed_this_session = 0
        index = 0
        while index < len(jobs):
            job = jobs[index]
            if store.is_complete(job):
                progress = store.progress()
                print(f"[{index + 1:>3}/{len(jobs)}] resume-skip {job.label} (complete={progress['complete']})")
                index += 1
                continue

            stop = index + 1
            batch: list[ExperimentJob] = [job]
            if batched:
                key = batch_key(job)
                while stop < len(jobs) and batch_key(jobs[stop]) == key and len(batch) < seed_batch_size:
                    if not store.is_complete(jobs[stop]):
                        batch.append(jobs[stop])
                    stop += 1

            label = f"batch ({len(batch)} seeds)" if batched else job.label
            print(f"[{index + 1:>3}-{stop:>3}/{len(jobs)}] running     {label}")
            for current in batch:
                store.start_job(current)
            started = perf_counter()
            try:
                if batched:
                    results = run_many(tuple(batch), on_progress=store.update_job_progress)
                else:
                    results = (executor.run(job, on_progress=partial(store.update_job_progress, job)),)
                if len(results) != len(batch):
                    raise RuntimeError("executor returned the wrong result count")
            except BaseException as error:
                for current in batch:
                    store.fail_job(current, error)
                raise
            for current, result in zip(batch, results, strict=True):
                store.complete_job(current, result)
            completed_this_session += len(batch)
            progress = store.progress()
            eta = progress["estimated_remaining_seconds"]
            eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
            print(
                f"[{index + 1:>3}-{stop:>3}/{len(jobs)}] complete    {label} "
                f"in {perf_counter() - started:.3f}s; ETA {eta_text}"
            )
            index = stop
            if interrupt_after_completed_jobs is not None and completed_this_session >= interrupt_after_completed_jobs:
                raise KeyboardInterrupt("simulated interruption after completed job")

        finalize = getattr(study, "finalize", None)
        if callable(finalize):
            finalize(store.path)

    return store.path
