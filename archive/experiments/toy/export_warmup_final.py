"""Validate and compact the protected structured-game warm-up ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.toy.application import make_toy_problem
from experiments.toy.final_export import (
    validate_canonical_semantic_config,
    validate_protocol_fields,
    validate_trace_arrays,
)
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, solve_mle
from mairl_exp.artifacts import atomic_write_json, atomic_write_npz, sha256_file
from mairl_exp.batching import evaluation_checkpoints

METHODS = ("active", "fixed_random", "random_each_round")
WARMUP_BUDGETS = (0, 24, 60, 120, 240, 500, 800, 1000)
SEEDS = tuple(range(82_000, 82_100))
METRICS = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")


def _expected_specs() -> list[dict[str, object]]:
    return [
        {"method": method, "warmup_budget": budget, "seed": seed}
        for budget in WARMUP_BUDGETS
        for method in METHODS
        for seed in SEEDS
    ]


def export(run_dir: Path, output_stem: Path) -> tuple[Path, Path]:
    """Fail closed on the raw run, then retain only final per-seed metrics."""

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    semantic = json.loads((run_dir / "semantic_config.json").read_text())
    validate_canonical_semantic_config(semantic)
    validate_protocol_fields(
        semantic["experiment"],
        {
            "name": "structured-toy-warmup-final",
            "seed_start": 82_000,
            "seed_count": 100,
            "warmup_budgets": list(WARMUP_BUDGETS),
            "total_samples": 1000,
            "evaluation_interval": 25,
            "lambda_information_multiplier": 1.0,
            "lambda_warm": 0.01,
            "confidence_delta": 0.05,
            "methods": list(METHODS),
        },
        label="structured-toy warm-up experiment",
    )
    progress = manifest.get("progress", {})
    required = {"complete": 2400, "failed": 0, "pending": 0, "running": 0, "total": 2400}
    if manifest.get("status") != "complete" or any(
        progress.get(key) != value for key, value in required.items()
    ):
        raise ValueError("run is not the complete 2,400-job protected warm-up sweep")
    expected = manifest["expected_jobs"]
    if [item["spec"] for item in expected] != _expected_specs():
        raise ValueError("warm-up grid or paired protected seeds differ from the frozen protocol")

    final = {metric: np.empty((8, 3, 100), dtype=np.float64) for metric in METRICS}
    checksums: dict[str, str] = {}
    maximum_mle_residual = 0.0
    application = semantic["application"]
    problem, _true_omega, _leader_reward = make_toy_problem(
        int(application["problem_seed"]),
        eta=float(application["eta"]),
        radius=float(application["radius"]),
        true_omega=application["true_omega"],
        leader_reward=application["leader_reward"],
    )
    mle_config = MLEConfig(
        **{key: value for key, value in semantic["numerics"]["mle"].items() if key != "_target_"}
    )
    full_warmup_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
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
        budget_index = WARMUP_BUDGETS.index(spec["warmup_budget"])
        method_index = METHODS.index(spec["method"])
        seed_index = spec["seed"] - SEEDS[0]
        with np.load(result_path, allow_pickle=False) as arrays:
            checkpoints = arrays["evaluation_sample_counts"]
            expected_checkpoints = evaluation_checkpoints(spec["warmup_budget"], 1000, 25)
            try:
                stored_residual = validate_trace_arrays(
                    arrays,
                    total_samples=1000,
                    checkpoints=expected_checkpoints,
                )
                if stored_residual is None:
                    cached = full_warmup_data.get(spec["seed"])
                    policies = arrays["policies"]
                    actions = arrays["follower_actions"]
                    stored_omega = arrays["omega_history"][-1]
                    if cached is None:
                        certified = solve_mle(
                            problem,
                            InteractionDataset(policies, actions),
                            mle_config,
                            initial=stored_omega,
                        )
                        if not np.allclose(
                            certified.omega,
                            stored_omega,
                            rtol=0.0,
                            atol=mle_config.projected_gradient_tolerance,
                        ):
                            raise ValueError("stored full-warm-up MLE disagrees with recomputation")
                        cached = (
                            policies.copy(),
                            actions.copy(),
                            stored_omega.copy(),
                            certified.projected_gradient_residual,
                        )
                        full_warmup_data[spec["seed"]] = cached
                    elif not (
                        np.array_equal(policies, cached[0])
                        and np.array_equal(actions, cached[1])
                        and np.array_equal(stored_omega, cached[2])
                    ):
                        raise ValueError("methods do not share identical full-warm-up data and MLE")
                    stored_residual = cached[3]
                maximum_mle_residual = max(maximum_mle_residual, stored_residual)
            except ValueError as error:
                raise ValueError(f"invalid raw trajectory in {job_id}: {error}") from error
            for metric in METRICS:
                values = arrays[f"metric__{metric}"]
                if values.shape != checkpoints.shape or not np.all(np.isfinite(values)):
                    raise ValueError(f"bad metric {metric} in {job_id}")
                final[metric][budget_index, method_index, seed_index] = values[-1]

    for metric, values in final.items():
        common_warmup = values[-1]
        if not np.allclose(common_warmup, common_warmup[0][None, :], rtol=0.0, atol=1e-12):
            raise ValueError(f"methods disagree for {metric} when warm-up consumes the full budget")

    arrays = {
        "warmup_budgets": np.asarray(WARMUP_BUDGETS),
        "warmup_fractions": np.asarray(WARMUP_BUDGETS, dtype=np.float64) / 1000.0,
        "methods": np.asarray(METHODS),
        "seeds": np.asarray(SEEDS),
        **{f"final__{metric}": values for metric, values in final.items()},
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
            "experiment": "structured-toy protected warm-up ablation",
            "run_id": manifest["run_id"],
            "semantic_config_sha256": manifest["semantic_config_sha256"],
            "source_commit": manifest["git"]["commit"],
            "source_git_fingerprint": manifest["git"]["fingerprint"],
            "source_manifest_sha256": sha256_file(manifest_path),
            "raw_result_checksum_aggregate_sha256": aggregate,
            "derived_npz_sha256": sha256_file(npz_path),
            "operational_batching": {
                "first_100_jobs": 100,
                "remaining_jobs": 64,
                "affects_scientific_identity": False,
                "note": (
                    "The interrupted first attempt completed the first 100 jobs as one batch; "
                    "the exact same content-addressed run resumed with the profiled batch size 64."
                ),
            },
            "seed_policy": {"paired": True, "seeds": list(SEEDS), "count": len(SEEDS)},
            "warmup_budgets": list(WARMUP_BUDGETS),
            "total_samples": 1000,
            "methods": list(METHODS),
            "maximum_mle_residual": maximum_mle_residual,
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
