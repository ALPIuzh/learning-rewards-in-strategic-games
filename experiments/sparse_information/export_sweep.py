"""Validate a completed sweep and export its minimal TikZ input data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

METHODS = ("active", "fixed_random", "random_each_round")
ACTION_COUNTS = (2, 4, 8, 16, 32)
METRICS = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")
SEEDS = tuple(range(72_000, 72_100))
CHECKPOINTS = np.arange(0, 201, 10)


def _expected_specs() -> list[dict[str, object]]:
    return [
        {"leader_actions": m, "method": method, "seed": seed, "warmup_enabled": False}
        for m in ACTION_COUNTS
        for method in METHODS
        for seed in SEEDS
    ]


def _verified_result_path(run_dir: Path, item: dict[str, object]) -> tuple[str, Path, str]:
    job_id = str(item["job_id"])
    job_dir = run_dir / "jobs" / job_id
    status = json.loads((job_dir / "status.json").read_text())
    path = job_dir / "result.npz"
    digest = sha256_file(path)
    if status.get("status") != "complete":
        raise ValueError(f"job {job_id} is not complete")
    if status.get("result_sha256") != digest:
        raise ValueError(f"job {job_id} failed its result checksum")
    return job_id, path, digest


def export(run_dir: Path, output_stem: Path) -> tuple[Path, Path]:
    """Fail closed on malformed raw output, then write compact arrays and provenance."""

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    progress = manifest.get("progress", {})
    required_progress = {"complete": 1500, "failed": 0, "pending": 0, "running": 0, "total": 1500}
    if manifest.get("status") != "complete" or any(
        progress.get(key) != value for key, value in required_progress.items()
    ):
        raise ValueError("run is not the complete 1,500-job final sweep")
    expected = manifest["expected_jobs"]
    specs = [item["spec"] for item in expected]
    if specs != _expected_specs():
        raise ValueError("job grid or paired protected seeds differ from the frozen protocol")

    curves = {metric: np.empty((3, 100, 21)) for metric in METRICS}
    final = {metric: np.empty((5, 3, 100)) for metric in METRICS}
    convergence = np.empty((5, 3, 100, 3))
    raw_checksums: dict[str, str] = {}
    for item in expected:
        spec, job_id = item["spec"], item["job_id"]
        m_index = ACTION_COUNTS.index(spec["leader_actions"])
        method_index = METHODS.index(spec["method"])
        seed_index = spec["seed"] - SEEDS[0]
        job_id, path, digest = _verified_result_path(run_dir, item)
        raw_checksums[job_id] = digest
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["evaluation_sample_counts"], CHECKPOINTS):
                raise ValueError(f"bad sample accounting in {job_id}")
            if data["policies"].shape != (200, spec["leader_actions"]):
                raise ValueError(f"bad policy shape in {job_id}")
            if data["follower_actions"].shape != (200,):
                raise ValueError(f"bad follower sample shape in {job_id}")
            if any(not np.all(np.isfinite(data[name])) for name in data.files):
                raise ValueError(f"non-finite array in {job_id}")
            values = [data[f"metric__{metric}"] for metric in METRICS]
            if any(value.shape != (21,) for value in values):
                raise ValueError(f"bad metric shape in {job_id}")
            for metric, value in zip(METRICS, values, strict=True):
                final[metric][m_index, method_index, seed_index] = value[-1]
                if spec["leader_actions"] == 16:
                    curves[metric][method_index, seed_index] = value
            convergence[m_index, method_index, seed_index] = [
                np.max(data["diagnostic__mle_residual"], initial=0),
                np.max(data["diagnostic__policy_residual"], initial=0),
                np.max(data["diagnostic__information_gain"], initial=0),
            ]

    ratios = np.asarray([m * (m + 1) / 2 for m in ACTION_COUNTS])
    arrays = {
        "action_counts": np.asarray(ACTION_COUNTS),
        "methods": np.asarray(METHODS),
        "seeds": np.asarray(SEEDS),
        "evaluation_sample_counts": CHECKPOINTS,
        "analytical_initial_information_ratio": ratios,
        "convergence_maxima": convergence,
        **{f"m16__{key}": value for key, value in curves.items()},
        **{f"final__{key}": value for key, value in final.items()},
    }
    npz_path, json_path = output_stem.with_suffix(".npz"), output_stem.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(npz_path, arrays)
    summaries = {
        metric: {
            str(m): {
                method: {
                    "mean": float(final[metric][i, j].mean()),
                    "standard_error": float(final[metric][i, j].std(ddof=1) / 10),
                }
                for j, method in enumerate(METHODS)
            }
            for i, m in enumerate(ACTION_COUNTS)
        }
        for metric in METRICS
    }
    aggregate = hashlib.sha256(
        "\n".join(f"{job_id}:{raw_checksums[job_id]}" for job_id in sorted(raw_checksums)).encode()
    ).hexdigest()
    atomic_write_json(
        json_path,
        {
            "schema_version": 1,
            "experiment": "sparse-information protected action-count sweep",
            "run_id": manifest["run_id"],
            "source_manifest_sha256": sha256_file(manifest_path),
            "semantic_config_sha256": manifest["semantic_config_sha256"],
            "source_commit": manifest["git"]["commit"],
            "source_git_fingerprint": manifest["git"]["fingerprint"],
            "raw_result_checksum_aggregate_sha256": aggregate,
            "derived_npz_sha256": sha256_file(npz_path),
            "analytical_ratio_validated": bool(np.array_equal(ratios, np.array([3, 10, 36, 136, 528]))),
            "seed_policy": {"paired": True, "seeds": list(SEEDS), "count": len(SEEDS)},
            "action_counts": list(ACTION_COUNTS),
            "methods": list(METHODS),
            "summaries": summaries,
        },
    )
    return json_path, npz_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.output_stem)


if __name__ == "__main__":
    main()
