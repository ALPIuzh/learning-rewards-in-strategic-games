"""Export the matched Lobeke warm-up/lambda factorial from two native runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from experiments._config import REPOSITORY_ROOT
from experiments.guard.paper.figures.data import (
    MethodKey,
    SeedTrace,
    load_run_traces,
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_values_in_unit_reward,
)
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.export_final import (
    DEFAULT_TRANSFER_CACHE,
    TARGET_METRICS,
    TRANSFER_CACHE_SCHEMA_VERSION,
    _cached_transfer_metrics,
    _result_set_sha256,
)
from experiments.guard.paper.figures.task_sets import (
    TRANSFER_EVALUATION_TASKS,
    normalized_structured_tasks,
)
from experiments.guard.paper.rewards import evaluation_reward_normalization_provenance
from experiments.zero_sum_comparison.application import ZeroSumApplicationRuntime
from mairl.core.problem import FloatArray
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.artifacts import atomic_write_json, atomic_write_npz, sha256_file

DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility/data/lobeke/warmup_factorial_unit_reward"
ACTIVE_KEYS: tuple[MethodKey, MethodKey] = (("active", True), ("active", False))
BASE_SWAPPED_SEEDS = tuple(range(30_000, 30_020))
FULL_FACTORIAL_SEEDS = tuple(range(30_000, 30_050))
ARMS: dict[str, tuple[str, MethodKey]] = {
    "warm_lambda3": ("swapped", ("active", True)),
    "no_warm_lambda3": ("reference", ("active", False)),
    "warm_lambda12": ("reference", ("active", True)),
    "no_warm_lambda12": ("swapped", ("active", False)),
}
EFFECT_METRICS = (
    "target_suboptimality",
    "target_fisher_error",
    "coefficient_l2_error",
    "self_concordance_distance",
    "averaged_transfer_suboptimality",
    "averaged_transfer_fisher_error",
)


def _final_only(traces: list[SeedTrace]) -> list[SeedTrace]:
    return [
        SeedTrace(
            seed=trace.seed,
            checkpoints=trace.checkpoints[-1:].copy(),
            metrics={name: values[-1:].copy() for name, values in trace.metrics.items()},
            omega_history=trace.omega_history[-1:].copy(),
        )
        for trace in traces
    ]


def _paired_interval(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired effects require at least two finite scalar values")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    half_width = float(student_t.ppf(0.975, df=values.size - 1) * standard_error)
    return {
        "mean": mean,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "standard_error": standard_error,
        "count": int(values.size),
    }


def _load_manifest(run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    progress = manifest.get("progress", {})
    if manifest.get("status") != "complete" or progress.get("failed") != 0:
        raise ValueError(f"factorial source run is incomplete: {run_dir}")
    return manifest


def _select_reference_traces(
    traces: dict[MethodKey, list[SeedTrace]],
    expected_seeds: tuple[int, ...],
) -> dict[MethodKey, list[SeedTrace]]:
    selected: dict[MethodKey, list[SeedTrace]] = {}
    for key in ACTIVE_KEYS:
        selected[key] = [trace for trace in traces[key] if trace.seed in expected_seeds]
        if tuple(trace.seed for trace in selected[key]) != expected_seeds:
            raise ValueError(f"reference run is missing paired factorial seeds for {key}")
    return selected


def _config_protocol(run_dir: Path) -> dict[str, object]:
    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    experiment = config.experiment
    return {
        "warmup_budget": int(experiment.warmup_budget),
        "total_samples": int(experiment.total_samples),
        "warm_lambda_multiplier": float(experiment.warmup_lambda_information_multiplier),
        "no_warm_lambda_multiplier": float(experiment.no_warmup_lambda_information_multiplier),
        "evaluation_policy": OmegaConf.to_container(
            config.numerics.evaluation_policy, resolve=True
        ),
    }


def _validate_active_traces(
    traces: dict[MethodKey, list[SeedTrace]],
    expected_seeds: tuple[int, ...],
    *,
    source: str,
) -> None:
    if set(traces) != set(ACTIVE_KEYS):
        raise ValueError(f"{source} contains unexpected protocols: {sorted(traces)}")
    if any(tuple(trace.seed for trace in traces[key]) != expected_seeds for key in ACTIVE_KEYS):
        raise ValueError(f"{source} does not use the expected paired seeds")


def _load_swapped_traces(
    swapped_run: Path,
    swapped_extension_run: Path | None,
) -> tuple[
    dict[MethodKey, list[SeedTrace]],
    tuple[int, ...],
    dict[str, object] | None,
]:
    swapped_traces = load_run_traces(swapped_run)
    _validate_active_traces(
        swapped_traces,
        BASE_SWAPPED_SEEDS,
        source="base swapped run",
    )
    if swapped_extension_run is None:
        return swapped_traces, BASE_SWAPPED_SEEDS, None

    extension_manifest = _load_manifest(swapped_extension_run)
    extension_seeds = tuple(range(30_020, 30_050))
    extension_traces = load_run_traces(swapped_extension_run)
    _validate_active_traces(
        extension_traces,
        extension_seeds,
        source="swapped extension",
    )
    combined = {
        key: sorted([*swapped_traces[key], *extension_traces[key]], key=lambda trace: trace.seed)
        for key in ACTIVE_KEYS
    }
    _validate_active_traces(combined, FULL_FACTORIAL_SEEDS, source="combined swapped runs")
    return combined, FULL_FACTORIAL_SEEDS, extension_manifest


def _validated_protocols(
    reference_run: Path,
    swapped_run: Path,
    swapped_extension_run: Path | None,
) -> dict[str, dict[str, object]]:
    protocols = {
        "reference": _config_protocol(reference_run),
        "swapped": _config_protocol(swapped_run),
    }
    if swapped_extension_run is not None:
        protocols["swapped_extension"] = _config_protocol(swapped_extension_run)

    for name in ("reference", "swapped"):
        protocol = protocols[name]
        if protocol["warmup_budget"] != 200 or protocol["total_samples"] != 1600:
            raise ValueError(f"{name} run has an invalid common protocol: {protocol}")
    multipliers = {
        name: (
            protocol["warm_lambda_multiplier"],
            protocol["no_warm_lambda_multiplier"],
        )
        for name, protocol in protocols.items()
    }
    if multipliers["reference"] != (12.0, 3.0):
        raise ValueError("reference run is not the lambda-(12,3) Active protocol")
    if multipliers["swapped"] != (3.0, 12.0):
        raise ValueError("swapped run is not the lambda-(3,12) Active protocol")
    if protocols["reference"]["evaluation_policy"] != protocols["swapped"]["evaluation_policy"]:
        raise ValueError("factorial runs use different evaluation optimizers")
    if protocols.get("swapped_extension", protocols["swapped"]) != protocols["swapped"]:
        raise ValueError("swapped base and extension runs use different protocols")
    return protocols


def _source_reward_units(
    reference_run: Path,
    swapped_run: Path,
    swapped_extension_run: Path | None,
) -> dict[str, bool]:
    units = {
        "reference": run_source_suboptimality_is_unit_reward(reference_run),
        "swapped": run_source_suboptimality_is_unit_reward(swapped_run),
    }
    if swapped_extension_run is not None and (
        run_source_suboptimality_is_unit_reward(swapped_extension_run) != units["swapped"]
    ):
        raise ValueError("swapped base and extension use different source-reward units")
    return units


def _validate_runtime_pair(
    reference_runtime: ZeroSumApplicationRuntime,
    swapped_runtime: ZeroSumApplicationRuntime,
) -> None:
    for name, reference_value, swapped_value in (
        ("features", reference_runtime.problem.features, swapped_runtime.problem.features),
        ("true omega", reference_runtime.true_omega, swapped_runtime.true_omega),
    ):
        if not np.array_equal(reference_value, swapped_value):
            raise ValueError(f"factorial runs disagree on {name}")


def _factorial_arrays(
    *,
    expected_seeds: tuple[int, ...],
    run_traces: dict[str, dict[MethodKey, list[SeedTrace]]],
    runtime: ZeroSumApplicationRuntime,
    source_metrics_are_unit_reward: dict[str, bool],
    leader_config: LeaderOptimizationConfig,
    transfer_tasks: FloatArray,
    transfer_cache: Path,
    cache_source: str,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"seeds": np.asarray(expected_seeds, dtype=np.int64)}
    raw_reward_range = float(runtime.metadata["leader_reward_normalization"]["raw_range"])
    cache_dir = (
        transfer_cache / "warmup-factorial" / f"v{TRANSFER_CACHE_SCHEMA_VERSION}" / cache_source
    )
    for slug, (run_name, key) in ARMS.items():
        traces = _final_only(run_traces[run_name][key])
        arrays[f"{slug}__final_omega"] = np.asarray(
            [trace.omega_history[-1] for trace in traces], dtype=np.float64
        )
        for exported_name, raw_name in TARGET_METRICS.items():
            if raw_name == "stackelberg_suboptimality":
                values = source_suboptimality_values_in_unit_reward(
                    traces,
                    raw_reward_range=raw_reward_range,
                    already_unit_reward=source_metrics_are_unit_reward[run_name],
                )[:, 0]
            else:
                values = np.asarray(
                    [trace.metrics[raw_name][-1] for trace in traces], dtype=np.float64
                )
            arrays[f"{slug}__{exported_name}"] = values
        task_suboptimality, task_fisher_error = _cached_transfer_metrics(
            cache_dir=cache_dir,
            slug=slug,
            runtime=runtime,
            traces=traces,
            leader_config=leader_config,
            transfer_tasks=transfer_tasks,
        )
        arrays[f"{slug}__taskwise_transfer_suboptimality"] = task_suboptimality[:, :, 0]
        arrays[f"{slug}__taskwise_transfer_fisher_error"] = task_fisher_error[:, :, 0]
        arrays[f"{slug}__averaged_transfer_suboptimality"] = task_suboptimality.mean(axis=0)[:, 0]
        arrays[f"{slug}__averaged_transfer_fisher_error"] = task_fisher_error.mean(axis=0)[:, 0]
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"factorial array {name} contains non-finite values")
    return arrays


def _factorial_effects(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float | int]]]:
    effects = {
        f"warm_minus_no_warm_at_lambda{multiplier}": {
            metric: _paired_interval(
                arrays[f"warm_lambda{multiplier}__{metric}"]
                - arrays[f"no_warm_lambda{multiplier}__{metric}"]
            )
            for metric in EFFECT_METRICS
        }
        for multiplier in (3, 12)
    }
    effects["factorial_interaction_lambda12_minus_lambda3"] = {
        metric: _paired_interval(
            (arrays[f"warm_lambda12__{metric}"] - arrays[f"no_warm_lambda12__{metric}"])
            - (arrays[f"warm_lambda3__{metric}"] - arrays[f"no_warm_lambda3__{metric}"])
        )
        for metric in EFFECT_METRICS
    }
    return effects


def _source_provenance(
    run_dir: Path,
    manifest: dict[str, object],
    protocol: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "run_id": manifest["run_id"],
        "git_commit": manifest["git"]["commit"],
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "result_set_sha256": _result_set_sha256(run_dir, manifest["expected_jobs"]),
        "protocol": protocol,
    }


def _generation_command(
    reference_run: Path,
    swapped_run: Path,
    swapped_extension_run: Path | None,
    output_dir: Path,
) -> str:
    command = (
        "python -m experiments.guard.paper.figures.export_warmup_factorial "
        f"--reference-run {reference_run} --swapped-run {swapped_run} "
    )
    if swapped_extension_run is not None:
        command += f"--swapped-extension-run {swapped_extension_run} "
    return command + f"--output-dir {output_dir}"


def export(
    reference_run: Path,
    swapped_run: Path,
    output_dir: Path,
    transfer_cache: Path = DEFAULT_TRANSFER_CACHE,
    swapped_extension_run: Path | None = None,
) -> tuple[Path, Path]:
    """Write the four matched Active arms, final diagnostics, and paired effects."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite factorial export: {output_dir}")

    reference_manifest = _load_manifest(reference_run)
    swapped_manifest = _load_manifest(swapped_run)
    swapped_traces, expected_seeds, extension_manifest = _load_swapped_traces(
        swapped_run,
        swapped_extension_run,
    )
    reference_traces = _select_reference_traces(load_run_traces(reference_run), expected_seeds)
    protocols = _validated_protocols(reference_run, swapped_run, swapped_extension_run)
    reference_runtime = load_runtime(reference_run)
    swapped_runtime = load_runtime(swapped_run)
    _validate_runtime_pair(reference_runtime, swapped_runtime)
    source_metrics_are_unit_reward = _source_reward_units(
        reference_run, swapped_run, swapped_extension_run
    )

    swapped_config = OmegaConf.load(swapped_run / "resolved_config.yaml")
    numerics = instantiate(swapped_config.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    transfer_tasks = normalized_structured_tasks(swapped_runtime, TRANSFER_EVALUATION_TASKS)
    cache_source = str(swapped_manifest["run_id"])
    if extension_manifest is not None:
        cache_source += f"-{extension_manifest['run_id']}"
    arrays = _factorial_arrays(
        expected_seeds=expected_seeds,
        run_traces={"reference": reference_traces, "swapped": swapped_traces},
        runtime=swapped_runtime,
        source_metrics_are_unit_reward=source_metrics_are_unit_reward,
        leader_config=leader_config,
        transfer_tasks=transfer_tasks,
        transfer_cache=transfer_cache,
        cache_source=cache_source,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"
    atomic_write_npz(data_path, arrays)
    sources: dict[str, object] = {
        "reference": _source_provenance(reference_run, reference_manifest, protocols["reference"]),
        "swapped": _source_provenance(swapped_run, swapped_manifest, protocols["swapped"]),
    }
    if extension_manifest is not None and swapped_extension_run is not None:
        sources["swapped_extension"] = _source_provenance(
            swapped_extension_run,
            extension_manifest,
            protocols["swapped_extension"],
        )
    provenance = {
        "schema_version": 2,
        "experiment": "matched Active IRL warm-up/lambda factorial",
        "sources": sources,
        "design": {
            "paired_seeds": list(expected_seeds),
            "seed_count": len(expected_seeds),
            "warmup_budgets": [0, 200],
            "lambda_information_multipliers": [3, 12],
            "total_samples": 1600,
            "rng": "purpose-separated streams keyed by seed and absolute sample index",
            "evaluation_policy": asdict(leader_config),
        },
        "transfer_task_policy": {
            "name": TRANSFER_EVALUATION_TASKS.name,
            "count": TRANSFER_EVALUATION_TASKS.count,
            "seed": TRANSFER_EVALUATION_TASKS.seed,
            "conditional_on_fixed_suite": True,
            "normalization": "positive-affine to [0,1] per task; R_max=1",
        },
        "leader_reward_normalization": evaluation_reward_normalization_provenance(
            swapped_runtime.metadata["leader_reward_normalization"]
        ),
        "arms": {
            slug: {"source": source, "method": key[0], "warmup_enabled": key[1]}
            for slug, (source, key) in ARMS.items()
        },
        "effects": _factorial_effects(arrays),
        "arrays": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in arrays.items()
        },
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generation_command": _generation_command(
            reference_run, swapped_run, swapped_extension_run, output_dir
        ),
    }
    provenance_path = output_dir / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    return data_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--swapped-run", type=Path, required=True)
    parser.add_argument("--swapped-extension-run", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transfer-cache", type=Path, default=DEFAULT_TRANSFER_CACHE)
    args = parser.parse_args()
    data_path, provenance_path = export(
        args.reference_run,
        args.swapped_run,
        args.output_dir,
        args.transfer_cache,
        args.swapped_extension_run,
    )
    print(f"wrote {data_path}")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
