"""Paper-facing Lobeke traces with the paired-action OLS correction applied."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.data import MethodKey, SeedTrace, load_run_traces
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import sha256_file

DEFAULT_CORRECTION_DIR = REPOSITORY_ROOT / "reproducibility/data/lobeke/paired_ols_correction"
_CORRECTED_METHODS: dict[MethodKey, str] = {
    ("zero_sum_ols", False): "paired_actions_ref22",
    ("follower_ols", False): "follower_only_ref22",
}


def comparison_config(run_dir: Path) -> dict[str, object]:
    """Return the scientific config after removing run-scheduling fields."""

    semantic = json.loads((run_dir / "semantic_config.json").read_text())
    experiment = dict(semantic["experiment"])
    for key in ("name", "methods", "warmup_modes"):
        experiment.pop(key, None)
    return {"application": semantic["application"], "numerics": semantic["numerics"], "experiment": experiment}


def _validate_native_paired_run(run_dir: Path, manifest: dict[str, object]) -> None:
    """Reject a new run unless its stored OLS protocol is the corrected one."""

    semantic = json.loads((run_dir / "semantic_config.json").read_text())
    ols = semantic.get("experiment", {}).get("ols", {})
    if ols.get("follower_reference_action") != 22:
        raise ValueError("native Lobeke run does not use follower reference action 22")
    if ols.get("leader_reference_action") != 22:
        raise ValueError("native Lobeke run does not use leader reference action 22")

    zero_sum_jobs = [entry for entry in manifest["expected_jobs"] if entry["spec"]["method"] == "zero_sum_ols"]
    if not zero_sum_jobs:
        raise ValueError("native Lobeke run has no zero-sum OLS jobs")
    for entry in zero_sum_jobs:
        job_dir = run_dir / "jobs" / entry["job_id"]
        summary = json.loads((job_dir / "summary.json").read_text())
        if summary.get("sampling_protocol") != ("paired_empirical_leader_and_follower_actions"):
            raise ValueError(f"zero-sum OLS job {entry['job_id']} is not paired-action")
        counts = summary.get("observed_action_counts", {})
        if counts.get("leader") != counts.get("follower") or not counts.get("leader"):
            raise ValueError(f"zero-sum OLS job {entry['job_id']} has invalid action counts")
        with np.load(job_dir / "result.npz", allow_pickle=False) as arrays:
            if "leader_actions" not in arrays:
                raise ValueError(f"zero-sum OLS job {entry['job_id']} has no leader samples")
            if arrays["leader_actions"].shape != arrays["follower_actions"].shape:
                raise ValueError(f"zero-sum OLS job {entry['job_id']} is not sample-paired")


def load_corrected_run_traces(run_dir: Path,
        correction_dir: Path = DEFAULT_CORRECTION_DIR) -> dict[MethodKey, list[SeedTrace]]:  # fmt: skip
    """Load the protected run and replace only its two corrected OLS traces.

    Every unaffected protocol remains byte-for-byte sourced from the protected
    run. The correction retains its follower samples, adds empirical leader
    samples for zero-sum OLS, and uses reference action 22 for both OLS systems.
    """

    traces = load_run_traces(run_dir)
    source_manifest = json.loads((run_dir / "manifest.json").read_text())
    provenance_path = correction_dir / "provenance.json"
    data_path = correction_dir / "data.npz"
    provenance = json.loads(provenance_path.read_text())

    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported paired-OLS correction schema")
    source = provenance.get("source", {})
    if source.get("run_id") != source_manifest.get("run_id"):
        _validate_native_paired_run(run_dir, source_manifest)
        return traces
    if source.get("manifest_sha256") != sha256_file(run_dir / "manifest.json"):
        raise ValueError("protected run manifest differs from the paired-OLS correction")
    if provenance.get("data_sha256") != sha256_file(data_path):
        raise ValueError("paired-OLS correction data failed its checksum")

    with np.load(data_path, allow_pickle=False) as arrays:
        seeds = arrays["seeds"].astype(np.int64, copy=True)
        checkpoints = arrays["checkpoints"].astype(np.int64, copy=True)
        if seeds.tolist() != provenance["seed_policy"]["seeds"]:
            raise ValueError("paired-OLS correction seed metadata is inconsistent")
        if checkpoints.tolist() != provenance["checkpoints"]:
            raise ValueError("paired-OLS correction checkpoint metadata is inconsistent")

        for key, prefix in _CORRECTED_METHODS.items():
            original = traces.get(key)
            if original is None:
                raise ValueError(f"protected run is missing corrected protocol {key}")
            if tuple(trace.seed for trace in original) != tuple(int(seed) for seed in seeds):
                raise ValueError(f"unpaired seeds for corrected protocol {key}")
            if any(not np.array_equal(trace.checkpoints, checkpoints) for trace in original):
                raise ValueError(f"checkpoint mismatch for corrected protocol {key}")

            omega = arrays[f"{prefix}__omega"].copy()
            metrics = {
                name: arrays[f"{prefix}__{name}"].copy()
                for name in ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")
            }
            expected_shape = (seeds.size, checkpoints.size)
            if omega.shape != (*expected_shape, original[0].omega_history.shape[1]):
                raise ValueError(f"invalid corrected coefficient trace for {key}")
            if any(values.shape != expected_shape for values in metrics.values()):
                raise ValueError(f"invalid corrected metric trace for {key}")
            if not np.all(np.isfinite(omega)) or any(not np.all(np.isfinite(values)) for values in metrics.values()):
                raise ValueError(f"non-finite corrected trace for {key}")

            traces[key] = [
                SeedTrace(
                    seed=int(seed),
                    checkpoints=checkpoints.copy(),
                    metrics={name: values[index].copy() for name, values in metrics.items()},
                    omega_history=omega[index].copy(),
                )
                for index, seed in enumerate(seeds)
            ]

    return traces


def load_figure_run_traces(run_dirs: list[Path] | tuple[Path, ...],
        correction_dir: Path = DEFAULT_CORRECTION_DIR) -> dict[MethodKey, list[SeedTrace]]:  # fmt: skip
    """Load one historical run or compose disjoint protocols from clean runs.

    A single run retains the historical correction fallback for reproducibility.
    A composed current-protocol export must instead contain native paired-action
    OLS results.  Duplicate protocols are rejected rather than silently letting
    command-line order decide which data appear in a figure.
    """

    paths = tuple(Path(path) for path in run_dirs)
    if not paths:
        raise ValueError("at least one Lobeke run directory is required")
    if len(paths) == 1:
        return load_corrected_run_traces(paths[0], correction_dir)

    composed: dict[MethodKey, list[SeedTrace]] = {}
    manifests = [json.loads((run_dir / "manifest.json").read_text()) for run_dir in paths]
    commits = {manifest["git"]["commit"] for manifest in manifests}
    if len(commits) != 1 or any(manifest["git"]["dirty"] for manifest in manifests):
        raise ValueError("composed Lobeke runs must share one clean git commit")
    configs = [comparison_config(run_dir) for run_dir in paths]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("composed Lobeke runs use different data-generating configurations")
    for run_dir, manifest in zip(paths, manifests, strict=True):
        methods = {entry["spec"]["method"] for entry in manifest["expected_jobs"]}
        if "zero_sum_ols" in methods:
            _validate_native_paired_run(run_dir, manifest)
        traces = load_run_traces(run_dir)
        duplicates = set(composed).intersection(traces)
        if duplicates:
            raise ValueError(f"duplicate Lobeke protocols across source runs: {duplicates}")
        composed.update(traces)
    return composed
