from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.guard.paper.figures.ridge_zero_tuning_data import (
    MLE_RIDGE,
    SEEDS,
    TASKS,
    TOTAL_SAMPLES,
    _protocol_slug,
    merge_candidate_artifacts,
)
from mairl.exceptions import ConfigurationError
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

SOURCE_CONFIG_SHA256 = "f" * 64
CANDIDATE = (0, 0.003)


def _write_bundle(path: Path, *, candidate: tuple[int, float] = CANDIDATE, ridge: float = MLE_RIDGE,
        seeds: np.ndarray = SEEDS, omega_offset: float = 0.0) -> Path:  # fmt: skip
    path.mkdir()
    slug = _protocol_slug(candidate)
    artifact = path / f"{slug}.npz"
    atomic_write_npz(
        artifact,
        {
            "schema_version": np.asarray(2, dtype=np.int64),
            "protocol": np.asarray(slug),
            "seeds": np.asarray(seeds, dtype=np.int64),
            "mle_ridge": np.asarray(ridge, dtype=np.float64),
            "omega_history": np.full((seeds.size, 1, 3), omega_offset, dtype=np.float64),
            "trace_sample_counts": np.full((seeds.size, 1), TOTAL_SAMPLES, dtype=np.int64),
        },
    )
    atomic_write_json(
        path / "summary.json",
        {
            "schema_version": 1,
            "sample_count": TOTAL_SAMPLES,
            "mle_ridge": ridge,
            "seeds": seeds.tolist(),
            "task_policy": {
                "name": TASKS.name,
                "seed": TASKS.seed,
                "count": TASKS.count,
                "candidate_pool": TASKS.candidate_pool,
                "normalized_reward_range": [0.0, 1.0],
            },
            "source_run_config_sha256": SOURCE_CONFIG_SHA256,
            "protocols": {
                slug: {
                    "definition": {
                        "method": "active",
                        "warmup_budget": candidate[0],
                        "lambda_information_multiplier": candidate[1],
                    },
                    "artifact": artifact.name,
                    "artifact_sha256": sha256_file(artifact),
                }
            },
        },
    )
    return path


def test_shard_merge_is_input_order_independent(tmp_path: Path) -> None:
    first = _write_bundle(tmp_path / "first")
    second = _write_bundle(tmp_path / "second", candidate=(50, 0.01))
    forward = merge_candidate_artifacts((first, second), source_config_sha256=SOURCE_CONFIG_SHA256)
    reverse = merge_candidate_artifacts((second, first), source_config_sha256=SOURCE_CONFIG_SHA256)
    assert tuple(forward) == tuple(reverse) == (CANDIDATE, (50, 0.01))
    assert {key: item.sha256 for key, item in forward.items()} == {key: item.sha256 for key, item in reverse.items()}


def test_shard_merge_accepts_only_byte_identical_duplicates(tmp_path: Path) -> None:
    first = _write_bundle(tmp_path / "first")
    duplicate = tmp_path / "duplicate"
    shutil.copytree(first, duplicate)
    merged = merge_candidate_artifacts((first, duplicate), source_config_sha256=SOURCE_CONFIG_SHA256)
    assert tuple(merged) == (CANDIDATE,)

    conflicting = _write_bundle(tmp_path / "conflicting", omega_offset=1.0)
    with pytest.raises(ConfigurationError, match="conflicting duplicate"):
        merge_candidate_artifacts((first, conflicting), source_config_sha256=SOURCE_CONFIG_SHA256)


@pytest.mark.parametrize(
    ("ridge", "seeds", "message"),
    ((1.0e-4, SEEDS, "frozen ridge-zero design"), (MLE_RIDGE, SEEDS[:-1], "frozen ridge-zero design")),
)
def test_shard_merge_rejects_design_mismatch(tmp_path: Path, ridge: float, seeds: np.ndarray, message: str) -> None:
    bundle = _write_bundle(tmp_path / "bundle", ridge=ridge, seeds=seeds)
    with pytest.raises(ConfigurationError, match=message):
        merge_candidate_artifacts((bundle,), source_config_sha256=SOURCE_CONFIG_SHA256)


def test_shard_merge_rejects_corrupt_artifact(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    summary_path = bundle / "summary.json"
    summary = json.loads(summary_path.read_text())
    artifact = bundle / next(iter(summary["protocols"].values()))["artifact"]
    artifact.write_bytes(b"not an npz")
    summary["protocols"][_protocol_slug(CANDIDATE)]["artifact_sha256"] = sha256_file(artifact)
    atomic_write_json(summary_path, summary)
    with pytest.raises(ConfigurationError, match="invalid candidate artifact"):
        merge_candidate_artifacts((bundle,), source_config_sha256=SOURCE_CONFIG_SHA256)
