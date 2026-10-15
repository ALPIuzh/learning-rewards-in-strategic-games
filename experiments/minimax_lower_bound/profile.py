"""Benchmark the worst lower-bound job and project the complete-grid runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.minimax_lower_bound.application import MinimaxLowerBoundApplication
from experiments.minimax_lower_bound.study import MinimaxLowerBoundStudy
from mairl_exp.config import REPOSITORY_ROOT


def projected_work_factor(study: MinimaxLowerBoundStudy, benchmark_weight: int) -> float:
    """Return total coordinate-sample work relative to one benchmark job."""

    total = sum(job.dimension * job.total_samples for job in study.jobs())
    if benchmark_weight <= 0:
        raise ValueError("benchmark weight must be positive")
    return total / benchmark_weight


def profile(output: Path | None = None) -> dict[str, object]:
    """Time the heaviest durable job, including compressed artifact creation."""

    study = instantiate(OmegaConf.load(REPOSITORY_ROOT / "conf/experiment/minimax_lower_bound.yaml"))
    numerics = instantiate(OmegaConf.load(REPOSITORY_ROOT / "conf/numerics/minimax_lower_bound.yaml"))
    benchmark = max(study.jobs(), key=lambda job: job.dimension * job.total_samples)
    runtime = MinimaxLowerBoundApplication().build(numerics)

    started = perf_counter()
    result = study.executor(runtime, numerics).run(benchmark)
    algorithm_seconds = perf_counter() - started
    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "result.npz"
        started = perf_counter()
        np.savez_compressed(artifact, **result.arrays)
        compression_seconds = perf_counter() - started
        artifact_bytes = artifact.stat().st_size

    weight = benchmark.dimension * benchmark.total_samples
    factor = projected_work_factor(study, weight)
    projected_seconds = factor * (algorithm_seconds + compression_seconds)
    payload = {
        "benchmark_job": benchmark.as_dict(),
        "algorithm_seconds": algorithm_seconds,
        "compression_seconds": compression_seconds,
        "artifact_bytes": artifact_bytes,
        "complete_grid_jobs": len(study.jobs()),
        "projected_work_factor": factor,
        "projected_complete_seconds": projected_seconds,
        "under_30_minutes": projected_seconds < 30 * 60,
        "max_mle_residual": float(np.max(result.arrays["diagnostic__mle_residual"])),
        "max_policy_residual": float(np.max(result.arrays["diagnostic__policy_residual"])),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(profile(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
