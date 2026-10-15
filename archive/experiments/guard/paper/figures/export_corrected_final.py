"""Export paper data with the paired-action Lobeke OLS correction applied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments._config import REPOSITORY_ROOT
from experiments.guard.paper.figures.corrected_traces import (
    DEFAULT_CORRECTION_DIR,
    load_corrected_run_traces,
)
from experiments.guard.paper.figures.data import (
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_values_in_unit_reward,
)
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.export_final import (
    CHECKPOINT_STRIDE,
    TARGET_METRICS,
    _subsampled_traces,
)
from experiments.guard.paper.figures.task_sets import (
    TRANSFER_EVALUATION_TASKS,
    normalized_structured_tasks,
)
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from experiments.guard.paper.rewards import evaluation_reward_normalization_provenance
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

BASE_EXPORT_DIR = REPOSITORY_ROOT / "reproducibility/data/lobeke/final_evaluation"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility/data/lobeke/final_evaluation_corrected"
CORRECTED_METHODS = {
    ("zero_sum_ols", False): "zero_sum_ols",
    ("follower_ols", False): "follower_ols",
}


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths in tracked provenance commands."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        if "artifacts" in resolved.parts:
            index = resolved.parts.index("artifacts")
            return Path(*resolved.parts[index:]).as_posix()
        return resolved.as_posix()


def _load_verified_base_export(
    base_export_dir: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, np.ndarray]]:
    base_data_path = base_export_dir / "data.npz"
    base_provenance_path = base_export_dir / "provenance.json"
    base_provenance = json.loads(base_provenance_path.read_text())
    if base_provenance.get("data_sha256") != sha256_file(base_data_path):
        raise ValueError("base Lobeke export failed its checksum")
    if base_provenance.get("source", {}).get("run_id") != "a4b1b2a1cdf886dd":
        raise ValueError("unexpected base Lobeke run")
    with np.load(base_data_path, allow_pickle=False) as base_arrays:
        arrays = {name: base_arrays[name].copy() for name in base_arrays.files}
    return base_data_path, base_provenance_path, base_provenance, arrays


def export_corrected(
    run_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    correction_dir: Path = DEFAULT_CORRECTION_DIR,
    base_export_dir: Path = BASE_EXPORT_DIR,
) -> tuple[Path, Path]:
    """Write a new immutable export; never modify the protected base export."""

    base_data_path, base_provenance_path, base_provenance, arrays = _load_verified_base_export(
        base_export_dir
    )

    correction_provenance_path = correction_dir / "provenance.json"
    correction_provenance = json.loads(correction_provenance_path.read_text())
    traces = load_corrected_run_traces(run_dir, correction_dir)
    runtime = load_runtime(run_dir)
    source_metrics_are_unit_reward = run_source_suboptimality_is_unit_reward(run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    transfer_tasks = normalized_structured_tasks(runtime, TRANSFER_EVALUATION_TASKS)

    taskwise_arrays: dict[str, np.ndarray] = {}
    for key, slug in CORRECTED_METHODS.items():
        full_traces = traces[key]
        method_traces = _subsampled_traces(full_traces)
        seeds = np.asarray([trace.seed for trace in method_traces], dtype=np.int64)
        checkpoints = method_traces[0].checkpoints
        if not np.array_equal(seeds, np.arange(30_000, 30_050)):
            raise ValueError(f"unexpected corrected seed set for {slug}")
        if any(not np.array_equal(trace.checkpoints, checkpoints) for trace in method_traces):
            raise ValueError(f"inconsistent corrected checkpoints for {slug}")

        arrays[f"{slug}__seeds"] = seeds
        arrays[f"{slug}__checkpoints"] = checkpoints.copy()
        arrays[f"{slug}__final_omega"] = np.asarray(
            [trace.omega_history[-1] for trace in full_traces], dtype=np.float64
        )
        for exported_name, raw_name in TARGET_METRICS.items():
            if raw_name == "stackelberg_suboptimality":
                values = source_suboptimality_values_in_unit_reward(
                    method_traces,
                    raw_reward_range=float(
                        runtime.metadata["leader_reward_normalization"]["raw_range"]
                    ),
                    already_unit_reward=source_metrics_are_unit_reward,
                )
            else:
                values = np.asarray(
                    [trace.metrics[raw_name] for trace in method_traces], dtype=np.float64
                )
            arrays[f"{slug}__{exported_name}"] = values

        task_suboptimality = []
        task_fisher_error = []
        for task_index, reward in enumerate(transfer_tasks):
            suboptimality, fisher_error = task_transfer_metrics(
                runtime.problem,
                runtime.true_omega,
                reward,
                leader_config,
                method_traces,
                task_seed=TRANSFER_EVALUATION_TASKS.seed,
                task_index=task_index,
            )
            task_suboptimality.append(suboptimality)
            task_fisher_error.append(fisher_error)
            print(
                f"{slug}: corrected transfer task {task_index + 1}/{len(transfer_tasks)}",
                flush=True,
            )
        stacked_suboptimality = np.stack(task_suboptimality, axis=0)
        stacked_fisher_error = np.stack(task_fisher_error, axis=0)
        taskwise_arrays[f"{slug}__taskwise_transfer_suboptimality"] = stacked_suboptimality
        taskwise_arrays[f"{slug}__taskwise_transfer_fisher_error"] = stacked_fisher_error
        arrays[f"{slug}__averaged_transfer_suboptimality"] = np.mean(stacked_suboptimality, axis=0)
        arrays[f"{slug}__averaged_transfer_fisher_error"] = np.mean(stacked_fisher_error, axis=0)

    if len(arrays) != 72:
        raise ValueError(f"corrected Lobeke export has unexpected array count: {len(arrays)}")
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("corrected Lobeke export contains non-finite values")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"
    taskwise_path = output_dir / "corrected_ols_taskwise_transfer.npz"
    atomic_write_npz(data_path, arrays)
    atomic_write_npz(taskwise_path, taskwise_arrays)
    array_schema = {
        name: {"shape": list(values.shape), "dtype": str(values.dtype)}
        for name, values in arrays.items()
    }
    provenance = {
        "schema_version": 3,
        "experiment": "Lobeke protected final evaluation with paired-action OLS correction",
        "source": {
            "run_id": base_provenance["source"]["run_id"],
            "result_count": 450,
            "unaffected_job_count": 350,
            "corrected_job_count": 100,
            "base_export_sha256": sha256_file(base_data_path),
            "base_provenance_sha256": sha256_file(base_provenance_path),
            "correction_data_sha256": correction_provenance["data_sha256"],
            "correction_provenance_sha256": sha256_file(correction_provenance_path),
        },
        "correction": {
            "methods": ["zero_sum_ols", "follower_ols"],
            "zero_sum_sampling": "paired empirical expert leader and follower actions",
            "reference_action": 22,
            "paper_method_label": "Zero-sum OLS",
        },
        "leader_reward_normalization": evaluation_reward_normalization_provenance(
            runtime.metadata["leader_reward_normalization"]
        ),
        "seed_policy": base_provenance["seed_policy"],
        "methods": base_provenance["methods"],
        "metrics": base_provenance["metrics"],
        "transfer_task_policy": base_provenance["transfer_task_policy"],
        "checkpoint_policy": {
            "source_interval": 40,
            "display_stride": CHECKPOINT_STRIDE,
            "display_interval": 40 * CHECKPOINT_STRIDE,
            "final_checkpoint_always_retained": True,
        },
        "arrays": array_schema,
        "data_sha256": sha256_file(data_path),
        "corrected_ols_taskwise_transfer_sha256": sha256_file(taskwise_path),
        "generation_command": (
            "python -m experiments.guard.paper.figures.export_corrected_final "
            f"--run-dir {_portable_path(run_dir)} "
            f"--output-dir {_portable_path(output_dir)} "
            f"--correction-dir {_portable_path(correction_dir)}"
        ),
    }
    provenance_path = output_dir / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    return data_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--base-export-dir", type=Path, default=BASE_EXPORT_DIR)
    args = parser.parse_args()
    data_path, provenance_path = export_corrected(
        args.run_dir, args.output_dir, args.correction_dir, args.base_export_dir
    )
    print(f"wrote {data_path}")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
