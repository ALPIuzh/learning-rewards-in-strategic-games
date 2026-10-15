"""Recompute full source curves for the corrected fixed-expert OLS protocols."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.downstream import load_runtime
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.inference.fixed_leader_ols import FixedLeaderOLSConfig, fit_fixed_leader_ols
from mairl.zero_sum.liao import fit_liao_zero_sum_paired_actions
from mairl_exp.artifacts import atomic_write_json, atomic_write_npz

DEFAULT_RUN = Path(
    "artifacts/runs/guard-lobeke-paper/a4b1b2a1cdf886dd"
)
DEFAULT_OUTPUT = Path("investigations/warmup_baselines/corrected_ols_curves_v1")


def _jobs(run_dir: Path) -> list[tuple[int, Path]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return sorted(
        (
            int(entry["spec"]["seed"]),
            run_dir / "jobs" / str(entry["job_id"]) / "result.npz",
        )
        for entry in manifest["expected_jobs"]
        if entry["spec"]["method"] == "zero_sum_ols" and not bool(entry["spec"]["warmup_enabled"])
    )


def recompute(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    data_path = output_dir / "data.npz"
    summary_path = output_dir / "summary.json"
    if data_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    runtime = load_runtime(run_dir)
    jobs = _jobs(run_dir)
    seeds = np.asarray([seed for seed, _ in jobs], dtype=np.int64)
    follower_samples = []
    checkpoints: np.ndarray | None = None
    for _, path in jobs:
        with np.load(path, allow_pickle=False) as arrays:
            current = arrays["evaluation_sample_counts"].copy()
            if checkpoints is None:
                checkpoints = current
            else:
                np.testing.assert_array_equal(current, checkpoints)
            follower_samples.append(arrays["follower_actions"].copy())
    assert checkpoints is not None

    config = FixedLeaderOLSConfig(
        pseudocount=0.5,
        smoothing_warning_ratio=0.1,
        identifiability="require_unique",
        constraint="ball",
        follower_reference_action=22,
        leader_reference_action=22,
        rank_relative_tolerance=1.0e-10,
        rank_absolute_tolerance=1.0e-12,
        ball_tolerance=1.0e-6,
        ball_root_tolerance=1.0e-12,
        ball_max_iterations=400,
    )
    paired_histories = []
    follower_histories = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed, follower_actions in zip(seeds, follower_samples, strict=True):
            leader_actions = experiment_rng(int(seed), RandomPurpose.EXPERT_LEADER).choice(
                runtime.game.n_max_actions,
                size=len(follower_actions),
                p=runtime.expert.max_policy,
            )
            paired_histories.append(
                np.asarray(
                    [
                        fit_liao_zero_sum_paired_actions(
                            runtime.game,
                            runtime.expert,
                            leader_actions[:checkpoint],
                            follower_actions[:checkpoint],
                            config,
                        ).omega
                        for checkpoint in checkpoints
                    ]
                )
            )
            follower_histories.append(
                np.asarray(
                    [
                        fit_fixed_leader_ols(
                            runtime.problem,
                            runtime.fixed_response,
                            follower_actions[:checkpoint],
                            config,
                        ).omega
                        for checkpoint in checkpoints
                    ]
                )
            )
    paired_histories = np.asarray(paired_histories)
    follower_histories = np.asarray(follower_histories)

    arrays: dict[str, np.ndarray] = {
        "seeds": seeds,
        "checkpoints": checkpoints,
        "paired_actions_ref22__omega": paired_histories,
        "follower_only_ref22__omega": follower_histories,
    }
    for name, histories in (
        ("paired_actions_ref22", paired_histories),
        ("follower_only_ref22", follower_histories),
    ):
        metrics = [
            runtime.evaluate(history, int(seed))
            for seed, history in zip(seeds, histories, strict=True)
        ]
        for metric_name in metrics[0]:
            arrays[f"{name}__{metric_name}"] = np.asarray(
                [metrics_for_seed[metric_name] for metrics_for_seed in metrics]
            )
        arrays[f"{name}__coefficient_l2_error_direct"] = np.linalg.norm(
            histories - runtime.true_omega[None, None, :], axis=2
        )
        print(f"evaluated corrected source curve for {name}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(data_path, arrays)
    atomic_write_json(
        summary_path,
        {
            "source_run_id": json.loads((run_dir / "manifest.json").read_text())["run_id"],
            "seeds": seeds.tolist(),
            "checkpoints": checkpoints.tolist(),
            "reference_action": 22,
            "retained_follower_samples": True,
            "paired_leader_samples": True,
            "protocol_note": (
                "leader and follower marginals are empirical; the x-axis counts follower "
                "actions and the paired estimator observes the same number of leader actions"
            ),
        },
    )
    return data_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(recompute(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
