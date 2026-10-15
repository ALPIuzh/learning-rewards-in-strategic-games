"""Merge ridge-zero sensitivity shards into compact publication data.

Each input directory is an independently generated bundle from
``experiments.guard.paper.run_sensitivity``.  Workers must use
different output directories.  This exporter validates that every bundle uses
the same frozen design, accepts only complete candidate cells, rejects
conflicting duplicates, and never imputes a missing cell.

Example::

    uv run python -m experiments.guard.paper.figures.ridge_zero_tuning_data \
      --input-dir artifacts/derived/ridge-zero-shard-00 \
      --input-dir artifacts/derived/ridge-zero-shard-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.task_sets import StructuredTaskSet, normalized_structured_tasks
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

DEFAULT_SOURCE_CONFIG = (
    REPOSITORY_ROOT / "reproducibility" / "data" / "lobeke" / "ridge_zero_tuning" / "source_config.yaml"
)
DEFAULT_INPUT = REPOSITORY_ROOT / "artifacts" / "derived" / "ridge-zero-sensitivity-publication-v1"
DEFAULT_HELDOUT = REPOSITORY_ROOT / "artifacts" / "derived" / "ridge-zero-confirmation-publication-v1"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "data" / "lobeke" / "ridge_zero_tuning" / "provenance.json"

TOTAL_SAMPLES = 1_600
MLE_RIDGE = 0.0
SEEDS = np.arange(94_000, 94_008, dtype=np.int64)
TASKS = StructuredTaskSet(name="active_expert_warmup_audit", count=100, seed=9_400, candidate_pool=2_000)
WARMUP_BUDGETS = (0, 50, 200)
MULTIPLIERS = (0.003, 0.01, 0.03, 0.3, 1.0)
PLANNED_CANDIDATES = tuple((budget, multiplier) for budget in WARMUP_BUDGETS for multiplier in MULTIPLIERS)
FROZEN_PROTOCOLS = ((0, 0.03), (200, 0.003))
HELDOUT_SEEDS = np.arange(95_000, 95_020, dtype=np.int64)
HELDOUT_TASKS = StructuredTaskSet(name="active_expert_warmup_audit", count=100, seed=9_500, candidate_pool=2_000)
HELDOUT_CANDIDATES = ((0, 0.003), (0, 0.03), (200, 0.003), (200, 0.03))
FIXED_EXPERT_SLUG = "fixed_expert_matched_ridge"
COLOR_SCALE = {"min": 0.0, "max": 4.0e-4, "bins": 7}

Candidate = tuple[int, float]


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    candidate: Candidate
    path: Path
    sha256: str
    arrays: dict[str, np.ndarray]


def _protocol_slug(candidate: Candidate) -> str:
    budget, multiplier = candidate
    warmup = f"warm{budget}" if budget else "no_warm"
    multiplier_slug = f"{multiplier:g}".replace(".", "p")
    return f"active_{warmup}_lambda{multiplier_slug}"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError as error:
        raise ConfigurationError(f"publication provenance path is outside the repository: {path}") from error


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _paired_interval(values: np.ndarray) -> dict[str, float | int | bool]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2 or not np.all(np.isfinite(vector)):
        raise ConfigurationError("paired interval requires at least two finite observations")
    mean = float(np.mean(vector))
    standard_error = float(np.std(vector, ddof=1) / np.sqrt(vector.size))
    half_width = float(student_t.ppf(0.975, df=vector.size - 1) * standard_error)
    lower = mean - half_width
    upper = mean + half_width
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "contains_zero": lower <= 0.0 <= upper,
        "paired_seeds": int(vector.size),
    }


def _validate_summary(path: Path, source_config_sha256: str) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise ConfigurationError(f"missing shard summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    expected_task_policy = {
        "name": TASKS.name,
        "seed": TASKS.seed,
        "count": TASKS.count,
        "candidate_pool": TASKS.candidate_pool,
        "normalized_reward_range": [0.0, 1.0],
    }
    checks = {
        "schema_version": summary.get("schema_version") == 1,
        "sample_count": summary.get("sample_count") == TOTAL_SAMPLES,
        "mle_ridge": float(summary.get("mle_ridge", np.nan)) == MLE_RIDGE,
        "seeds": summary.get("seeds") == SEEDS.tolist(),
        "task_policy": summary.get("task_policy") == expected_task_policy,
        "source_config_sha256": (
            summary.get("source_config_sha256", summary.get("source_run_config_sha256")) == source_config_sha256
        ),
    }
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise ConfigurationError(f"shard {path} does not match the frozen ridge-zero design: {', '.join(failures)}")
    protocols = summary.get("protocols")
    if not isinstance(protocols, dict):
        raise ConfigurationError(f"shard {path} has no protocol mapping")
    return summary


def _load_protocol_artifact(path: Path, *, protocol_slug: str, expected_seeds: np.ndarray) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"invalid candidate artifact: {path}") from error
    required = {"schema_version", "protocol", "seeds", "mle_ridge", "omega_history", "trace_sample_counts"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ConfigurationError(f"candidate artifact {path} is missing {missing}")
    if int(arrays["schema_version"].item()) != 2:
        raise ConfigurationError(f"candidate artifact {path} is not schema 2")
    if str(arrays["protocol"].item()) != protocol_slug:
        raise ConfigurationError(f"candidate artifact identity mismatch: {path}")
    if float(arrays["mle_ridge"].item()) != MLE_RIDGE:
        raise ConfigurationError(f"candidate artifact ridge mismatch: {path}")
    if not np.array_equal(arrays["seeds"], expected_seeds):
        raise ConfigurationError(f"candidate artifact seed mismatch: {path}")
    omega_history = np.asarray(arrays["omega_history"], dtype=np.float64)
    sample_counts = np.asarray(arrays["trace_sample_counts"], dtype=np.int64)
    if omega_history.shape[0] != expected_seeds.size or sample_counts.shape[0] != expected_seeds.size:
        raise ConfigurationError(f"candidate artifact is not a complete paired cell: {path}")
    if not np.all(sample_counts == sample_counts[0]) or int(sample_counts[0, -1]) != TOTAL_SAMPLES:
        raise ConfigurationError(f"candidate artifact checkpoint mismatch: {path}")
    if not np.all(np.isfinite(omega_history)):
        raise ConfigurationError(f"candidate artifact contains nonfinite estimates: {path}")
    return arrays


def _load_artifact(path: Path, candidate: Candidate) -> dict[str, np.ndarray]:
    return _load_protocol_artifact(path, protocol_slug=_protocol_slug(candidate), expected_seeds=SEEDS)


def merge_candidate_artifacts(input_dirs: tuple[Path, ...],
        *, source_config_sha256: str) -> dict[Candidate, CandidateArtifact]:  # fmt: skip
    """Validate and merge complete candidate cells from independent bundles."""

    if not input_dirs:
        raise ConfigurationError("at least one ridge-zero shard is required")
    merged: dict[Candidate, CandidateArtifact] = {}
    for input_dir in sorted((path.resolve() for path in input_dirs), key=str):
        summary = _validate_summary(input_dir, source_config_sha256)
        for raw in summary["protocols"].values():
            definition = raw.get("definition", {})
            if definition.get("method") != "active":
                continue
            candidate = (int(definition["warmup_budget"]), float(definition["lambda_information_multiplier"]))
            if candidate not in PLANNED_CANDIDATES:
                raise ConfigurationError(f"unexpected candidate in {input_dir}: {candidate}")
            artifact_path = input_dir / str(raw["artifact"])
            digest = sha256_file(artifact_path)
            if digest != raw.get("artifact_sha256"):
                raise ConfigurationError(f"candidate checksum mismatch: {artifact_path}")
            arrays = _load_artifact(artifact_path, candidate)
            previous = merged.get(candidate)
            if previous is not None:
                if previous.sha256 != digest:
                    raise ConfigurationError(
                        f"conflicting duplicate candidate {candidate}: {previous.path} and {artifact_path}"
                    )
                continue
            merged[candidate] = CandidateArtifact(candidate, artifact_path, digest, arrays)
    return merged


def _transfer_scores(*, runtime: Any, numerics: Any, tasks: np.ndarray, arrays: dict[str, np.ndarray],
        seeds: np.ndarray, task_set: StructuredTaskSet) -> np.ndarray:  # fmt: skip
    checkpoints = np.asarray(arrays["trace_sample_counts"][0], dtype=np.int64)
    traces = [
        SeedTrace(
            seed=int(seed),
            checkpoints=np.asarray([checkpoints[-1]], dtype=np.int64),
            metrics={},
            omega_history=np.asarray(arrays["omega_history"][index, -1:, :], dtype=np.float64),
        )
        for index, seed in enumerate(seeds)
    ]
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    taskwise = []
    for task_index, reward in enumerate(tasks):
        suboptimality, _ = task_transfer_metrics(
            runtime.problem,
            runtime.true_omega,
            reward,
            leader_config,
            traces,
            task_seed=task_set.seed,
            task_index=task_index,
        )
        taskwise.append(suboptimality[:, 0])
    scores = np.mean(np.stack(taskwise), axis=0)
    if scores.shape != (seeds.size,) or not np.all(np.isfinite(scores)) or np.min(scores) < 0.0:
        raise ConfigurationError("invalid transfer scores in candidate artifact")
    return scores


def _heldout_confirmation(*, input_dir: Path, source_config_sha256: str, runtime: Any,
        numerics: Any) -> tuple[dict[str, object], dict[str, np.ndarray]]:  # fmt: skip
    input_dir = input_dir.resolve()
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        raise ConfigurationError(f"missing held-out summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    expected_task_policy = {
        "name": HELDOUT_TASKS.name,
        "seed": HELDOUT_TASKS.seed,
        "count": HELDOUT_TASKS.count,
        "candidate_pool": HELDOUT_TASKS.candidate_pool,
        "normalized_reward_range": [0.0, 1.0],
    }
    valid = (
        summary.get("schema_version") == 1
        and summary.get("sample_count") == TOTAL_SAMPLES
        and float(summary.get("mle_ridge", np.nan)) == MLE_RIDGE
        and summary.get("seeds") == HELDOUT_SEEDS.tolist()
        and summary.get("task_policy") == expected_task_policy
        and summary.get("source_config_sha256", summary.get("source_run_config_sha256")) == source_config_sha256
    )
    if not valid:
        raise ConfigurationError("held-out bundle does not match the frozen confirmation design")

    protocols = summary.get("protocols", {})
    expected_slugs = (*(_protocol_slug(candidate) for candidate in HELDOUT_CANDIDATES), FIXED_EXPERT_SLUG)
    if set(protocols) != set(expected_slugs):
        raise ConfigurationError("held-out bundle has an unexpected protocol set")
    tasks = normalized_structured_tasks(runtime, HELDOUT_TASKS)
    per_seed: dict[str, np.ndarray] = {}
    sources = []
    for slug in expected_slugs:
        raw = protocols[slug]
        artifact_path = input_dir / str(raw["artifact"])
        digest = sha256_file(artifact_path)
        if digest != raw.get("artifact_sha256"):
            raise ConfigurationError(f"held-out artifact checksum mismatch: {artifact_path}")
        arrays = _load_protocol_artifact(artifact_path, protocol_slug=slug, expected_seeds=HELDOUT_SEEDS)
        per_seed[slug] = _transfer_scores(
            runtime=runtime, numerics=numerics, tasks=tasks, arrays=arrays, seeds=HELDOUT_SEEDS, task_set=HELDOUT_TASKS
        )
        sources.append({"protocol": slug, "path": _relative(artifact_path), "sha256": digest})

    no_warm_small = _protocol_slug((0, 0.003))
    no_warm_frozen = _protocol_slug((0, 0.03))
    warm_frozen = _protocol_slug((200, 0.003))
    warm_large = _protocol_slug((200, 0.03))
    comparison_pairs = {
        "no_warm_frozen_minus_alternative": (no_warm_frozen, no_warm_small),
        "warm_frozen_minus_alternative": (warm_frozen, warm_large),
        "warm_minus_no_warm_at_0.003": (warm_frozen, no_warm_small),
        "warm_minus_no_warm_at_0.03": (warm_large, no_warm_frozen),
        "frozen_no_warm_minus_fixed_expert": (no_warm_frozen, FIXED_EXPERT_SLUG),
        "frozen_warm_minus_fixed_expert": (warm_frozen, FIXED_EXPERT_SLUG),
    }
    comparisons = {
        name: {"left": left, "right": right, **_paired_interval(per_seed[left] - per_seed[right])}
        for name, (left, right) in comparison_pairs.items()
    }
    protocol_summaries = [
        {
            "protocol": slug,
            "averaged_transfer_suboptimality": {
                **_paired_interval(per_seed[slug]),
                "per_seed": per_seed[slug].tolist(),
                "tasks": HELDOUT_TASKS.count,
            },
        }
        for slug in expected_slugs
    ]
    payload = {
        "purpose": "independent estimator-matched confirmation; no reselection",
        "sample_count": TOTAL_SAMPLES,
        "mle_ridge": MLE_RIDGE,
        "seed_policy": {
            "start": int(HELDOUT_SEEDS[0]),
            "count": int(HELDOUT_SEEDS.size),
            "paired_across_protocols": True,
            "disjoint_from_sensitivity_and_final_evaluation": True,
        },
        "task_policy": {
            "name": HELDOUT_TASKS.name,
            "count": HELDOUT_TASKS.count,
            "seed": HELDOUT_TASKS.seed,
            "candidate_pool": HELDOUT_TASKS.candidate_pool,
            "normalization": "positive-affine to [0,1] per reward matrix",
        },
        "row_choices_recovered": {
            "no_warm": bool(np.mean(per_seed[no_warm_frozen]) < np.mean(per_seed[no_warm_small])),
            "warm_200": bool(np.mean(per_seed[warm_frozen]) < np.mean(per_seed[warm_large])),
        },
        "protocols": protocol_summaries,
        "paired_comparisons": comparisons,
        "source": {"input_bundle": _relative(input_dir), "candidate_artifacts": sources},
    }
    arrays = {
        "heldout_seeds": HELDOUT_SEEDS,
        "heldout_protocols": np.asarray(expected_slugs),
        "heldout_per_seed_downstream_suboptimality": np.stack([per_seed[slug] for slug in expected_slugs]),
    }
    return payload, arrays


def export(*, input_dirs: tuple[Path, ...], heldout_dir: Path, source_config: Path,
        output: Path, allow_incomplete: bool = False) -> Path:  # fmt: skip
    source_config = source_config.resolve()
    if not source_config.exists():
        raise ConfigurationError(f"missing source configuration: {source_config}")
    source_config_sha256 = sha256_file(source_config)
    artifacts = merge_candidate_artifacts(input_dirs, source_config_sha256=source_config_sha256)
    missing = tuple(candidate for candidate in PLANNED_CANDIDATES if candidate not in artifacts)
    if missing and not allow_incomplete:
        raise ConfigurationError(f"ridge-zero grid is incomplete; missing candidates: {missing}")
    missing_frozen = tuple(candidate for candidate in FROZEN_PROTOCOLS if candidate not in artifacts)
    if missing_frozen:
        raise ConfigurationError(f"frozen reference protocols are missing: {missing_frozen}")

    cfg = OmegaConf.load(source_config)
    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    runtime = application.build(numerics)
    tasks = normalized_structured_tasks(runtime, TASKS)
    scores = {
        candidate: _transfer_scores(
            runtime=runtime,
            numerics=numerics,
            tasks=tasks,
            arrays=artifacts[candidate].arrays,
            seeds=SEEDS,
            task_set=TASKS,
        )
        for candidate in PLANNED_CANDIDATES
        if candidate in artifacts
    }
    heldout, heldout_arrays = _heldout_confirmation(
        input_dir=heldout_dir, source_config_sha256=source_config_sha256, runtime=runtime, numerics=numerics
    )

    candidates = []
    candidate_order = tuple(candidate for candidate in PLANNED_CANDIDATES if candidate in scores)
    for candidate in candidate_order:
        per_seed = scores[candidate]
        reference = FROZEN_PROTOCOLS[0] if candidate[0] == 0 else FROZEN_PROTOCOLS[1]
        summary = _paired_interval(per_seed)
        candidates.append(
            {
                "warmup_budget": candidate[0],
                "lambda_information_multiplier": candidate[1],
                "downstream_suboptimality": {
                    "mean": summary["mean"],
                    "standard_error": summary["standard_error"],
                    "seeds": int(SEEDS.size),
                    "tasks": TASKS.count,
                    "per_seed": per_seed.tolist(),
                },
                "paired_difference_from_frozen": {
                    "reference": {"warmup_budget": reference[0], "lambda_information_multiplier": reference[1]},
                    **_paired_interval(per_seed - scores[reference]),
                },
            }
        )

    data_path = output.with_name("data.npz")
    candidate_budgets = np.asarray([candidate[0] for candidate in candidate_order], dtype=np.int64)
    candidate_multipliers = np.asarray([candidate[1] for candidate in candidate_order], dtype=np.float64)
    per_seed_scores = np.stack([scores[candidate] for candidate in candidate_order])
    arrays = {
        "candidate_warmup_budgets": candidate_budgets,
        "candidate_lambda_information_multipliers": candidate_multipliers,
        "seeds": SEEDS,
        "per_seed_downstream_suboptimality": per_seed_scores,
        **heldout_arrays,
    }
    atomic_write_npz(data_path, arrays)
    atomic_write_json(
        output,
        {
            "schema_version": 3,
            "status": "complete" if not missing else "partial",
            "purpose": "post-freeze estimator-matched sensitivity; not model selection",
            "sample_count": TOTAL_SAMPLES,
            "mle_ridge": MLE_RIDGE,
            "warmup_budgets": list(WARMUP_BUDGETS),
            "lambda_information_multipliers": list(MULTIPLIERS),
            "planned_candidates": len(PLANNED_CANDIDATES),
            "available_candidates": len(candidate_order),
            "missing_candidates": [list(candidate) for candidate in missing],
            "frozen_protocols": [
                {"warmup_budget": budget, "lambda_information_multiplier": multiplier}
                for budget, multiplier in FROZEN_PROTOCOLS
            ],
            "color_scale": COLOR_SCALE,
            "seed_policy": {
                "start": int(SEEDS[0]),
                "count": int(SEEDS.size),
                "paired_across_cells": True,
                "disjoint_from_final_evaluation": True,
            },
            "task_policy": {
                "name": TASKS.name,
                "count": TASKS.count,
                "seed": TASKS.seed,
                "candidate_pool": TASKS.candidate_pool,
                "normalization": "positive-affine to [0,1] per reward matrix",
                "same_tasks_for_every_cell_and_seed": True,
            },
            "score": "per-seed mean downstream SubOpt over fixed tasks, then mean across seeds",
            "uncertainty": {
                "interval": "paired two-sided Student-t 95% CI over seeds",
                "multiplicity_adjusted": False,
                "interpretation": "CI containing zero is unresolved, not evidence of equivalence",
            },
            "heldout_confirmation": heldout,
            "candidates": candidates,
            "source": {
                "source_config": _relative(source_config),
                "source_config_sha256": source_config_sha256,
                "input_bundles": [_relative(path) for path in sorted(input_dirs, key=str)],
                "candidate_artifacts": [
                    {
                        "warmup_budget": candidate[0],
                        "lambda_information_multiplier": candidate[1],
                        "path": _relative(artifacts[candidate].path),
                        "sha256": artifacts[candidate].sha256,
                    }
                    for candidate in candidate_order
                ],
            },
            "data_file": data_path.name,
            "data_sha256": sha256_file(data_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _array_sha256(value)}
                for name, value in arrays.items()
            },
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", dest="input_dirs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-config", type=Path, help="self-contained resolved YAML configuration")
    source.add_argument(
        "--source-run", type=Path, help="deprecated compatibility form; reads resolved_config.yaml from this run"
    )
    parser.add_argument("--heldout-dir", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="write a diagnostic partial export; missing cells are omitted"
    )
    args = parser.parse_args()
    output = export(
        input_dirs=tuple(args.input_dirs or (DEFAULT_INPUT,)),
        heldout_dir=args.heldout_dir,
        source_config=(
            args.source_config
            if args.source_config is not None
            else args.source_run / "resolved_config.yaml"
            if args.source_run is not None
            else DEFAULT_SOURCE_CONFIG
        ),
        output=args.output,
        allow_incomplete=args.allow_incomplete,
    )
    payload = json.loads(output.read_text())
    print(f"wrote {output}: {payload['available_candidates']}/{payload['planned_candidates']} complete cells")


if __name__ == "__main__":
    main()
