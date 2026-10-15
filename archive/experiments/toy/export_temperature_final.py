"""Validate and compact the protected structured-game temperature study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.toy.final_export import (
    validate_canonical_semantic_config,
    validate_protocol_fields,
    validate_trace_arrays,
)
from mairl_exp.artifacts import atomic_write_json, atomic_write_npz, sha256_file

METHODS = ("active", "fixed_random", "random_each_round")
ETA_VALUES = (0.5, 0.75, 1.25, 2.0, 3.0)
SEEDS = tuple(range(83_000, 83_100))
METRICS = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")
CHECKPOINTS = np.append(np.arange(120, 1001, 25, dtype=np.int64), 1000)


def _expected_specs() -> list[dict[str, object]]:
    return [
        {"eta": eta, "method": method, "seed": seed}
        for eta in ETA_VALUES
        for method in METHODS
        for seed in SEEDS
    ]


def export(run_dir: Path, output_stem: Path) -> tuple[Path, Path]:
    """Fail closed on all raw jobs, then retain complete per-seed metric traces."""

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    semantic = json.loads((run_dir / "semantic_config.json").read_text())
    validate_canonical_semantic_config(semantic)
    validate_protocol_fields(
        semantic["experiment"],
        {
            "name": "structured-toy-temperature-final",
            "seed_start": 83_000,
            "seed_count": 100,
            "eta_values": list(ETA_VALUES),
            "total_samples": 1000,
            "warmup_budget": 120,
            "evaluation_interval": 25,
            "lambda_information_multiplier": 1.0,
            "lambda_warm": 0.01,
            "confidence_delta": 0.05,
            "methods": list(METHODS),
        },
        label="structured-toy temperature experiment",
    )
    progress = manifest.get("progress", {})
    required = {"complete": 1500, "failed": 0, "pending": 0, "running": 0, "total": 1500}
    if manifest.get("status") != "complete" or any(
        progress.get(key) != value for key, value in required.items()
    ):
        raise ValueError("run is not the complete 1,500-job protected temperature sweep")
    expected = manifest["expected_jobs"]
    if [item["spec"] for item in expected] != _expected_specs():
        raise ValueError("temperature grid or paired protected seeds differ from the protocol")

    traces = {
        metric: np.empty((5, 3, 100, CHECKPOINTS.size), dtype=np.float64) for metric in METRICS
    }
    checksums: dict[str, str] = {}
    maximum_mle_residual = 0.0
    for item in expected:
        spec = item["spec"]
        job_id = str(item["job_id"])
        job_dir = run_dir / "jobs" / job_id
        status = json.loads((job_dir / "status.json").read_text())
        result_path = job_dir / "result.npz"
        digest = sha256_file(result_path)
        if status.get("status") != "complete" or status.get("result_sha256") != digest:
            raise ValueError(f"job {job_id} is incomplete or failed its checksum")
        checksums[job_id] = digest
        eta_index = ETA_VALUES.index(spec["eta"])
        method_index = METHODS.index(spec["method"])
        seed_index = spec["seed"] - SEEDS[0]
        with np.load(result_path, allow_pickle=False) as arrays:
            try:
                residual = validate_trace_arrays(
                    arrays,
                    total_samples=1000,
                    checkpoints=CHECKPOINTS,
                )
                if residual is None:
                    raise ValueError("temperature run unexpectedly lacks an online MLE certificate")
                maximum_mle_residual = max(maximum_mle_residual, residual)
            except ValueError as error:
                raise ValueError(f"invalid raw trajectory in {job_id}: {error}") from error
            for metric in METRICS:
                values = arrays[f"metric__{metric}"]
                if values.shape != CHECKPOINTS.shape or not np.all(np.isfinite(values)):
                    raise ValueError(f"bad metric {metric} in {job_id}")
                traces[metric][eta_index, method_index, seed_index] = values
    arrays = {
        "eta_values": np.asarray(ETA_VALUES),
        "methods": np.asarray(METHODS),
        "seeds": np.asarray(SEEDS),
        "evaluation_sample_counts": CHECKPOINTS,
        **{f"final__{metric}": values[..., -1] for metric, values in traces.items()},
    }
    npz_path, json_path = output_stem.with_suffix(".npz"), output_stem.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(npz_path, arrays)
    aggregate = hashlib.sha256(
        "\n".join(f"{job_id}:{checksums[job_id]}" for job_id in sorted(checksums)).encode()
    ).hexdigest()
    atomic_write_json(
        json_path,
        {
            "schema_version": 1,
            "experiment": "structured-toy protected inverse-temperature sensitivity",
            "run_id": manifest["run_id"],
            "semantic_config_sha256": manifest["semantic_config_sha256"],
            "source_commit": manifest["git"]["commit"],
            "source_git_fingerprint": manifest["git"]["fingerprint"],
            "source_manifest_sha256": sha256_file(manifest_path),
            "raw_result_checksum_aggregate_sha256": aggregate,
            "derived_npz_sha256": sha256_file(npz_path),
            "seed_policy": {"paired": True, "seeds": list(SEEDS), "count": len(SEEDS)},
            "eta_values": list(ETA_VALUES),
            "lambda_information_rule": "lambda = eta^2",
            "warmup_budget": 120,
            "total_samples": 1000,
            "methods": list(METHODS),
            "maximum_mle_residual": maximum_mle_residual,
            "compact_export": {
                "aggregation": "per-seed final values",
                "metrics": list(METRICS),
                "raw_traces_retained_locally": True,
            },
        },
    )
    return json_path, npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    json_path, npz_path = export(args.run_dir, args.output_stem)
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
