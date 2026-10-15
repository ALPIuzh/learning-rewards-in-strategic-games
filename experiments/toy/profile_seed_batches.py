"""Measure seed-vectorization batch sizes on the resolved toy experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from mairl_exp.application import ExperimentApplication
from mairl_exp.config import CONFIG_DIRECTORY, ExperimentNumerics
from mairl_exp.execution.jobs import StudyJob, StudyJobResult
from mairl_exp.studies.comparison import BaselineComparisonStudy


@dataclass(frozen=True, slots=True)
class Workload:
    method: str
    warmup_enabled: bool

    @property
    def label(self) -> str:
        mode = "warmup" if self.warmup_enabled else "no-warmup"
        return f"{mode}/{self.method}"


def _parse_workload(value: str) -> Workload:
    try:
        mode, method = value.split("/", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workloads must look like warmup/active or no-warmup/fixed_random") from error
    if mode not in {"warmup", "no-warmup"}:
        raise argparse.ArgumentTypeError("workload mode must be warmup or no-warmup")
    if method not in {"active", "fixed_random", "random_each_round"}:
        raise argparse.ArgumentTypeError(f"unknown method {method!r}")
    return Workload(method, mode == "warmup")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(4, 8, 16, 32, 64, 100))
    parser.add_argument("--workloads", type=_parse_workload, nargs="+", default=(_parse_workload("warmup/active"),))
    parser.add_argument("--config-name", default="runs/toy")
    parser.add_argument("--json", type=Path)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides; the tracked toy experiment is used by default")
    return parser


def _load(args: argparse.Namespace):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIRECTORY)):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    study = instantiate(cfg.experiment, _convert_="object")
    if not isinstance(application, ExperimentApplication):
        raise TypeError("application config did not instantiate ExperimentApplication")
    if not isinstance(numerics, ExperimentNumerics):
        raise TypeError("numerics config did not instantiate ExperimentNumerics")
    if not isinstance(study, BaselineComparisonStudy):
        raise TypeError("seed-batch profiling requires BaselineComparisonStudy")
    return application.build(numerics), numerics, study


def _selected_jobs(study: BaselineComparisonStudy, workloads: tuple[Workload, ...]) -> tuple[StudyJob, ...]:
    selected = {(item.method, item.warmup_enabled) for item in workloads}
    jobs = tuple(job for job in study.jobs() if (job.method, job.warmup_enabled) in selected)
    expected = len(selected) * study.seed_count
    if len(jobs) != expected:
        raise ValueError("a requested workload is absent from the experiment configuration")
    return jobs


def _run_size(batch_size: int, *, runtime, numerics: ExperimentNumerics, study: BaselineComparisonStudy,
        jobs: tuple[StudyJob, ...]) -> tuple[float, dict[str, StudyJobResult]]:  # fmt: skip
    executor = study.executor(runtime, numerics)
    results: dict[str, StudyJobResult] = {}
    started = perf_counter()
    start = 0
    while start < len(jobs):
        key = executor.batch_key(jobs[start])
        stop = start
        batch: list[StudyJob] = []
        while stop < len(jobs) and executor.batch_key(jobs[stop]) == key and len(batch) < batch_size:
            batch.append(jobs[stop])
            stop += 1
        outputs = executor.run_many(tuple(batch))
        results.update((job.label, output) for job, output in zip(batch, outputs, strict=True))
        start = stop
    return perf_counter() - started, results


def _compare(reference: dict[str, StudyJobResult], candidate: dict[str, StudyJobResult]) -> dict[str, float | int]:
    maximum_policy_difference = 0.0
    maximum_estimate_difference = 0.0
    action_mismatches = 0
    for label, expected in reference.items():
        actual = candidate[label]
        maximum_policy_difference = max(
            maximum_policy_difference, float(np.max(np.abs(expected.arrays["policies"] - actual.arrays["policies"])))
        )
        maximum_estimate_difference = max(
            maximum_estimate_difference,
            float(np.max(np.abs(expected.arrays["omega_history"] - actual.arrays["omega_history"]))),
        )
        action_mismatches += int(
            np.count_nonzero(expected.arrays["follower_actions"] != actual.arrays["follower_actions"])
        )
    return {
        "max_policy_abs_difference": maximum_policy_difference,
        "max_estimate_abs_difference": maximum_estimate_difference,
        "follower_action_mismatches": action_mismatches,
    }


def main() -> None:
    args = _parser().parse_args()
    batch_sizes = tuple(args.batch_sizes)
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    workloads = tuple(args.workloads)
    if len({item.label for item in workloads}) != len(workloads):
        raise ValueError("workloads must be unique")

    runtime, numerics, study = _load(args)
    jobs = _selected_jobs(study, workloads)
    rows = []
    reference = None
    reference_size = batch_sizes[0]
    for batch_size in batch_sizes:
        seconds, results = _run_size(batch_size, runtime=runtime, numerics=numerics, study=study, jobs=jobs)
        if reference is None:
            reference = results
            agreement = {
                "max_policy_abs_difference": 0.0,
                "max_estimate_abs_difference": 0.0,
                "follower_action_mismatches": 0,
            }
        else:
            agreement = _compare(reference, results)
        row = {
            "batch_size": batch_size,
            "seconds": seconds,
            "seed_trajectories_per_second": len(jobs) / seconds,
            **agreement,
        }
        rows.append(row)
        print(
            f"batch={batch_size:>3} time={seconds:8.3f}s "
            f"throughput={row['seed_trajectories_per_second']:.3f} seeds/s "
            f"policy_diff={agreement['max_policy_abs_difference']:.3e} "
            f"estimate_diff={agreement['max_estimate_abs_difference']:.3e} "
            f"action_mismatches={agreement['follower_action_mismatches']}",
            flush=True,
        )

    report = {
        "study": {
            "seed_count": study.seed_count,
            "total_samples": study.total_samples,
            "warmup_budget": study.warmup_budget,
            "workloads": [item.label for item in workloads],
            "reference_batch_size": reference_size,
        },
        "measurements": rows,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"json_report={args.json.resolve()}")


if __name__ == "__main__":
    main()
