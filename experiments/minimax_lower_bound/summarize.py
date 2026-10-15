"""Export coordinate-average and coordinate-maximum summaries from a complete run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.minimax_lower_bound.analysis import (
    FinalMetricRecord,
    aggregate_coordinate_risk,
    paired_bootstrap_loglog_slope,
)
from mairl.exceptions import ConfigurationError
from mairl_exp.execution.artifacts import atomic_write_json, sha256_file

METRICS = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")


def summarize_run(run_dir: Path) -> dict[str, object]:
    """Load verified final arrays and aggregate every predeclared metric."""

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ConfigurationError("lower-bound aggregation requires a complete run")
    progress = manifest.get("progress", {})
    if progress.get("complete") != 160 or progress.get("failed") != 0:
        raise ConfigurationError("lower-bound aggregation requires 160 successful jobs")
    expected_jobs = manifest.get("expected_jobs")
    if not isinstance(expected_jobs, list) or not expected_jobs:
        raise ConfigurationError("run manifest has no expected lower-bound jobs")
    cell_counts: dict[tuple[int, int], int] = {}
    for entry in expected_jobs:
        spec = entry["spec"]
        dimension = int(spec["dimension"])
        ratio = int(spec["samples_per_dimension"])
        key = (dimension, ratio)
        cell_counts[key] = cell_counts.get(key, 0) + 1
        if (
            dimension not in {33, 64}
            or ratio not in {4, 8, 16, 32, 64}
            or int(spec["total_samples"]) != dimension * ratio
            or sorted(spec["active_coordinates"]) != list(range(dimension))
            or len(spec["seeds"]) != dimension
        ):
            raise ConfigurationError("lower-bound job grid differs from the frozen protocol")
    if set(cell_counts) != {(dimension, ratio) for dimension in (33, 64) for ratio in (4, 8, 16, 32, 64)} or set(
        cell_counts.values()
    ) != {16}:
        raise ConfigurationError("lower-bound cells do not contain 16 balanced replicates")

    records = {metric: [] for metric in METRICS}
    result_digests: list[str] = []
    for entry in expected_jobs:
        job_id = str(entry["job_id"])
        spec = entry["spec"]
        job_dir = run_dir / "jobs" / job_id
        status = json.loads((job_dir / "status.json").read_text())
        result_path = job_dir / "result.npz"
        if status.get("status") != "complete":
            raise ConfigurationError(f"lower-bound job {job_id} is not complete")
        if status.get("result_sha256") != sha256_file(result_path):
            raise ConfigurationError(f"lower-bound job {job_id} failed its checksum")
        result_digests.append(f"{job_id}:{status['result_sha256']}")
        with np.load(result_path) as arrays:
            coordinates = np.asarray(spec["active_coordinates"], dtype=np.int64)
            if coordinates.shape != (int(spec["dimension"]),):
                raise ConfigurationError(f"lower-bound job {job_id} has invalid coordinates")
            for metric in METRICS:
                values = arrays[f"metric__{metric}"]
                if values.shape != (coordinates.size, 2) or not np.all(np.isfinite(values)):
                    raise ConfigurationError(f"lower-bound job {job_id} has invalid {metric}")
                records[metric].extend(
                    FinalMetricRecord(
                        dimension=int(spec["dimension"]),
                        samples_per_dimension=int(spec["samples_per_dimension"]),
                        active_coordinate=int(coordinate),
                        replicate=int(spec["replicate"]),
                        value=float(values[index, -1]),
                    )
                    for index, coordinate in enumerate(coordinates)
                )
    aggregated = {metric: list(aggregate_coordinate_risk(metric_records)) for metric, metric_records in records.items()}
    subopt_slopes: dict[str, object] = {}
    for dimension in (33, 64):
        tail_cells = [
            cell
            for cell in aggregated["stackelberg_suboptimality"]
            if cell["dimension"] == dimension and cell["samples_per_dimension"] >= 16
        ]
        values = np.stack([np.asarray(cell["coordinate_values"], dtype=np.float64).reshape(-1) for cell in tail_cells])
        subopt_slopes[str(dimension)] = paired_bootstrap_loglog_slope(
            np.asarray([cell["samples_per_dimension"] for cell in tail_cells]),
            values,
            bootstrap_samples=5_000,
            seed=44_031 + dimension,
        )

    return {
        "schema_version": 1,
        "experiment": "Active IRL on the minimax lower-bound instance",
        "source": {
            "run_id": manifest["run_id"],
            "study_name": manifest["study_name"],
            "git_commit": manifest["git"]["commit"],
            "git_fingerprint": manifest["git"]["fingerprint"],
            "semantic_config_sha256": manifest["semantic_config_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "result_set_sha256": hashlib.sha256("\n".join(sorted(result_digests)).encode()).hexdigest(),
            "result_count": len(result_digests),
        },
        "grid": {
            "dimensions": [33, 64],
            "samples_per_dimension": [4, 8, 16, 32, 64],
            "replicates_per_coordinate": 16,
            "warmup_budget": 0,
        },
        "metrics": aggregated,
        "subopt_tail_slopes": subopt_slopes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = summarize_run(args.run_dir.resolve())
    atomic_write_json(args.output.resolve(), payload)
    print(f"summary={args.output.resolve()}")


if __name__ == "__main__":
    main()
