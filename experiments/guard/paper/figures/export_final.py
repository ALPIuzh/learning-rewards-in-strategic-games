"""Export the protected Lobeke evaluation as compact, verified figure data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.common import (
    TRANSFER_CACHE_SCHEMA_VERSION,
    _cached_transfer_metrics,
    _implementation_sha256,
    _json_sha256,
    _portable_path,
    _source_record,
)
from experiments.guard.paper.figures.corrected_traces import comparison_config, load_figure_run_traces
from experiments.guard.paper.figures.data import (
    METHOD_SLUGS,
    SeedTrace,
    load_runtime,
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_values_in_unit_reward,
)
from experiments.guard.paper.figures.task_sets import TRANSFER_EVALUATION_TASKS, normalized_structured_tasks
from experiments.guard.paper.rewards import evaluation_reward_normalization_provenance
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "data" / "lobeke" / "final_evaluation_unit_reward_v2"
DEFAULT_TRANSFER_CACHE = REPOSITORY_ROOT / "artifacts" / "derived" / "guard-lobeke-transfer-diagnostics"
CHECKPOINT_STRIDE = 4
TARGET_METRICS = {
    "target_suboptimality": "stackelberg_suboptimality",
    "target_fisher_error": "fisher_transfer_error",
    "coefficient_l2_error": "coefficient_l2_error",
    "self_concordance_distance": "self_concordance_distance",
}


def _subsampled_traces(traces: list[SeedTrace]) -> list[SeedTrace]:
    """Keep a readable convergence grid and always retain the final checkpoint."""

    checkpoint_count = traces[0].checkpoints.size
    positions = np.unique(np.concatenate([np.arange(0, checkpoint_count, CHECKPOINT_STRIDE), [checkpoint_count - 1]]))
    return [
        SeedTrace(
            seed=trace.seed,
            checkpoints=trace.checkpoints[positions],
            metrics={name: values[positions] for name, values in trace.metrics.items()},
            omega_history=trace.omega_history[positions],
        )
        for trace in traces
    ]


def export(run_dirs: Path | Sequence[Path], output_dir: Path,
        transfer_cache: Path = DEFAULT_TRANSFER_CACHE) -> tuple[Path, Path]:  # fmt: skip
    """Write per-seed curves from one run or disjoint clean protocol runs."""

    paths = (run_dirs,) if isinstance(run_dirs, Path) else tuple(Path(path) for path in run_dirs)
    if not paths:
        raise ValueError("at least one Lobeke source run is required")
    manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    source_records = [_source_record(path, manifest) for path, manifest in zip(paths, manifests, strict=True)]
    if any(record["git_dirty"] for record in source_records):
        raise ValueError("final Lobeke sources must come from clean worktrees")
    commits = {str(record["git_commit"]) for record in source_records}
    if len(commits) != 1:
        raise ValueError(f"composed Lobeke runs use different commits: {sorted(commits)}")
    comparison_configs = {_json_sha256(comparison_config(path)) for path in paths}
    if len(comparison_configs) != 1:
        raise ValueError("composed Lobeke runs use different data-generating configurations")
    unit_markers = {run_source_suboptimality_is_unit_reward(path) for path in paths}
    if len(unit_markers) != 1:
        raise ValueError("composed Lobeke runs use different source-reward metric units")
    source_metrics_are_unit_reward = unit_markers.pop()
    traces = load_figure_run_traces(paths)
    if set(traces) != set(METHOD_SLUGS):
        raise ValueError(f"unexpected Lobeke protocols: {sorted(traces)}")

    primary_run_dir = paths[0]
    runtime = load_runtime(primary_run_dir)
    cfg = OmegaConf.load(primary_run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    transfer_tasks = normalized_structured_tasks(runtime, TRANSFER_EVALUATION_TASKS)
    composition_id = _json_sha256([record["run_id"] for record in source_records])[:16]

    arrays: dict[str, np.ndarray] = {}
    common_seeds: tuple[int, ...] | None = None
    array_schema: dict[str, dict[str, object]] = {}
    for key, slug in METHOD_SLUGS.items():
        raw_method_traces = traces[key]
        method_traces = _subsampled_traces(raw_method_traces)
        seeds = tuple(trace.seed for trace in method_traces)
        if common_seeds is None:
            common_seeds = seeds
        elif seeds != common_seeds:
            raise ValueError(f"unpaired final-evaluation seeds for {key}")
        checkpoints = method_traces[0].checkpoints
        if any(not np.array_equal(trace.checkpoints, checkpoints) for trace in method_traces):
            raise ValueError(f"inconsistent checkpoints across seeds for {key}")

        method_arrays = {
            f"{slug}__seeds": np.asarray(seeds, dtype=np.int64),
            f"{slug}__checkpoints": np.asarray(checkpoints, dtype=np.int64),
            f"{slug}__final_omega": np.asarray(
                [trace.omega_history[-1] for trace in raw_method_traces], dtype=np.float64
            ),
        }
        for exported_name, raw_name in TARGET_METRICS.items():
            if raw_name == "stackelberg_suboptimality":
                values = source_suboptimality_values_in_unit_reward(
                    method_traces,
                    raw_reward_range=float(runtime.metadata["leader_reward_normalization"]["raw_range"]),
                    already_unit_reward=source_metrics_are_unit_reward,
                )
            else:
                values = np.asarray([trace.metrics[raw_name] for trace in method_traces], dtype=np.float64)
            method_arrays[f"{slug}__{exported_name}"] = values
        taskwise_suboptimality, taskwise_fisher_error = _cached_transfer_metrics(
            cache_dir=transfer_cache / f"v{TRANSFER_CACHE_SCHEMA_VERSION}" / composition_id,
            slug=slug,
            runtime=runtime,
            traces=method_traces,
            leader_config=leader_config,
            transfer_tasks=transfer_tasks,
        )
        method_arrays[f"{slug}__taskwise_transfer_suboptimality"] = taskwise_suboptimality
        method_arrays[f"{slug}__taskwise_transfer_fisher_error"] = taskwise_fisher_error
        method_arrays[f"{slug}__averaged_transfer_suboptimality"] = np.mean(taskwise_suboptimality, axis=0)
        method_arrays[f"{slug}__averaged_transfer_fisher_error"] = np.mean(taskwise_fisher_error, axis=0)
        print(f"evaluated fixed-suite transfer diagnostics for {slug}", flush=True)
        for name, values in method_arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"derived array {name} contains non-finite values")
            arrays[name] = values
            array_schema[name] = {"shape": list(values.shape), "dtype": str(values.dtype)}

    assert common_seeds is not None
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing Lobeke export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"
    expected_array_count = len(METHOD_SLUGS) * (3 + len(TARGET_METRICS) + 4)
    if len(arrays) != expected_array_count:
        raise ValueError(f"native Lobeke export has {len(arrays)} arrays; expected {expected_array_count}")
    atomic_write_npz(data_path, arrays)
    experiment_config = OmegaConf.to_container(cfg.experiment, resolve=True)
    if not isinstance(experiment_config, dict):
        raise TypeError("resolved Lobeke experiment config is not a mapping")
    provenance = {
        "schema_version": 5,
        "experiment": "Lobeke composed current-protocol final evaluation",
        "source": {
            "composition_id": composition_id,
            "git_commit": next(iter(commits)),
            "result_count": sum(int(record["result_count"]) for record in source_records),
            "runs": source_records,
        },
        "seed_policy": {"paired": True, "seeds": list(common_seeds), "count": len(common_seeds)},
        "methods": {slug: {"method": key[0], "warmup_enabled": key[1]} for key, slug in METHOD_SLUGS.items()},
        "metrics": {
            "target_suboptimality": (
                "target-reward last-iterate SubOpt(hat mu), after positive-affine leader-reward normalization to [0,1]"
            ),
            "target_fisher_error": (
                "maximum Fisher coefficient-error norm at the target-optimal and estimate-optimal target policies"
            ),
            "averaged_transfer_suboptimality": (
                "within-seed arithmetic mean of taskwise SubOpt over the fixed task suite"
            ),
            "averaged_transfer_fisher_error": ("within-seed arithmetic mean of taskwise transfer-lemma Fisher errors"),
            "taskwise_transfer_suboptimality": ("per-task and per-seed SubOpt for every protected transfer reward"),
            "taskwise_transfer_fisher_error": ("per-task and per-seed transfer-lemma Fisher errors"),
            "coefficient_l2_error": "Euclidean coefficient error",
            "self_concordance_distance": (
                "simulation-only max_mu d_mu(true omega, estimated omega); the theorem's "
                "local certificate holds at values no larger than 1 / eta"
            ),
        },
        "transfer_task_policy": {
            "name": TRANSFER_EVALUATION_TASKS.name,
            "count": TRANSFER_EVALUATION_TASKS.count,
            "seed": TRANSFER_EVALUATION_TASKS.seed,
            "candidate_pool": TRANSFER_EVALUATION_TASKS.candidate_pool,
            "scale": TRANSFER_EVALUATION_TASKS.scale,
            "normalization": ("positive-affine normalization to [0,1] per leader-reward matrix; R_max=1"),
            "tuning_task_seed": 9_400,
            "disjoint_from_tuning": TRANSFER_EVALUATION_TASKS.seed != 9_400,
        },
        "checkpoint_policy": {
            "source_interval": 40,
            "display_stride": CHECKPOINT_STRIDE,
            "display_interval": 40 * CHECKPOINT_STRIDE,
            "final_checkpoint_always_retained": True,
        },
        "transfer_cache": {
            "schema_version": TRANSFER_CACHE_SCHEMA_VERSION,
            "implementation_sha256": _implementation_sha256(),
            "context_includes_problem_and_optimizer": True,
            "namespace": f"v{TRANSFER_CACHE_SCHEMA_VERSION}/{composition_id}",
        },
        "self_concordance_certificate": {
            "threshold": 1.0 / runtime.problem.eta,
            "eta": runtime.problem.eta,
            "simulation_only": True,
        },
        "leader_reward_normalization": evaluation_reward_normalization_provenance(
            runtime.metadata["leader_reward_normalization"]
        ),
        "arrays": array_schema,
        "protocol": {
            "active": {
                "warmup_budget": experiment_config["warmup_budget"],
                "warm_lambda_information_multiplier": experiment_config["warmup_lambda_information_multiplier"],
                "no_warm_lambda_information_multiplier": experiment_config["no_warmup_lambda_information_multiplier"],
                "comparison": "separately tuned protocols, not a matched-lambda ablation",
            },
            "fixed_expert_mle": {
                "expert_definition": runtime.fixed_mle_expert_definition,
                "leader_policy": runtime.fixed_mle_leader_policy.tolist(),
                "ridge": experiment_config["fixed_expert_mle_ridge"],
            },
            "ols": {
                "expert_definition": "joint_qre",
                "follower_reference_action": experiment_config["ols"]["follower_reference_action"],
                "leader_reference_action": experiment_config["ols"]["leader_reference_action"],
                "zero_sum_sampling": "paired empirical leader and follower actions",
            },
            "rng": {
                "schema": "purpose-separated streams keyed by seed and absolute sample index",
                "implementation_sha256": sha256_file(REPOSITORY_ROOT / "src/mairl/core/randomness.py"),
            },
        },
        "data_sha256": sha256_file(data_path),
        "generation_command": (
            "python -m experiments.guard.paper.figures.export_final "
            + " ".join(f"--run-dir {_portable_path(path)}" for path in paths)
            + f" --output-dir {_portable_path(output_dir)}"
        ),
    }
    provenance_path = output_dir / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    return data_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transfer-cache", type=Path, default=DEFAULT_TRANSFER_CACHE)
    args = parser.parse_args()
    data_path, provenance_path = export(args.run_dir, args.output_dir, args.transfer_cache)
    print(f"wrote {data_path}")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
