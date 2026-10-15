"""Reanalyze retained follower samples under corrected OLS protocols.

This script never modifies the protected run.  It isolates two corrections:

1. use the conditioning-minimizing reference action recorded in the config audit;
2. replace the noiseless expert leader distribution with empirical leader actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.task_sets import (
    TRANSFER_EVALUATION_TASKS,
    normalized_structured_tasks,
)
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.inference.fixed_leader_ols import FixedLeaderOLSConfig, fit_fixed_leader_ols
from mairl.planning.leader import LeaderOptimizationConfig
from mairl.zero_sum.liao import (
    fit_liao_zero_sum_ols,
    fit_liao_zero_sum_paired_actions,
)
from mairl_exp.artifacts import atomic_write_json, atomic_write_npz

DEFAULT_RUN = Path(
    "artifacts/runs/guard-lobeke-paper/a4b1b2a1cdf886dd"
)
DEFAULT_OUTPUT = Path("investigations/warmup_baselines/corrected_ols_reanalysis")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(values: np.ndarray) -> dict[str, object]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    radius = float(student_t.ppf(0.975, values.size - 1) * standard_error)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95": [mean - radius, mean + radius],
        "median": float(np.median(values)),
    }


def _job_results(run_dir: Path, method: str) -> list[tuple[int, Path]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [
        (
            int(entry["spec"]["seed"]),
            run_dir / "jobs" / str(entry["job_id"]) / "result.npz",
        )
        for entry in manifest["expected_jobs"]
        if entry["spec"]["method"] == method and not bool(entry["spec"]["warmup_enabled"])
    ]
    return sorted(rows)


def _target_metrics(
    runtime: object, seeds: np.ndarray, omegas: np.ndarray
) -> dict[str, np.ndarray]:
    rows = [
        runtime.evaluate(omega[None, :], int(seed))
        for seed, omega in zip(seeds, omegas, strict=True)
    ]
    return {name: np.asarray([row[name][0] for row in rows], dtype=np.float64) for name in rows[0]}


def _transfer_metrics(
    runtime: object,
    seeds: np.ndarray,
    omegas: np.ndarray,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(config.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    tasks = normalized_structured_tasks(runtime, TRANSFER_EVALUATION_TASKS)
    traces = [
        SeedTrace(
            seed=int(seed),
            checkpoints=np.asarray([1600], dtype=np.int64),
            metrics={},
            omega_history=omega[None, :],
        )
        for seed, omega in zip(seeds, omegas, strict=True)
    ]
    task_suboptimality = []
    task_fisher = []
    for task_index, reward in enumerate(tasks):
        suboptimality, fisher = task_transfer_metrics(
            runtime.problem,
            runtime.true_omega,
            reward,
            leader_config,
            traces,
            task_seed=TRANSFER_EVALUATION_TASKS.seed,
            task_index=task_index,
        )
        task_suboptimality.append(suboptimality[:, 0])
        task_fisher.append(fisher[:, 0])
        print(f"corrected paired OLS transfer task {task_index + 1}/{tasks.shape[0]}", flush=True)
    return np.asarray(task_suboptimality), np.asarray(task_fisher)


def recompute(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    data_path = output_dir / "data.npz"
    summary_path = output_dir / "summary.json"
    if data_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing corrected reanalysis in {output_dir}"
        )

    runtime = load_runtime(run_dir)
    jobs = _job_results(run_dir, "zero_sum_ols")
    seeds = np.asarray([seed for seed, _ in jobs], dtype=np.int64)
    follower_actions = []
    legacy_exact = []
    for _seed, path in jobs:
        with np.load(path, allow_pickle=False) as arrays:
            follower_actions.append(arrays["follower_actions"].copy())
            legacy_exact.append(arrays["omega_history"][-1].copy())
    follower_actions = np.asarray(follower_actions)
    legacy_exact = np.asarray(legacy_exact)

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
    exact_ref22 = []
    paired_ref22 = []
    follower_ref22 = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed, actions in zip(seeds, follower_actions, strict=True):
            leader_actions = experiment_rng(int(seed), RandomPurpose.EXPERT_LEADER).choice(
                runtime.game.n_max_actions,
                size=actions.size,
                p=runtime.expert.max_policy,
            )
            exact_ref22.append(
                fit_liao_zero_sum_ols(
                    runtime.game,
                    runtime.expert,
                    actions,
                    config,
                ).omega
            )
            paired_ref22.append(
                fit_liao_zero_sum_paired_actions(
                    runtime.game,
                    runtime.expert,
                    leader_actions,
                    actions,
                    config,
                ).omega
            )
            follower_ref22.append(
                fit_fixed_leader_ols(
                    runtime.problem,
                    runtime.fixed_response,
                    actions,
                    config,
                ).omega
            )
    exact_ref22 = np.asarray(exact_ref22)
    paired_ref22 = np.asarray(paired_ref22)
    follower_ref22 = np.asarray(follower_ref22)

    protocols = {
        "legacy_exact_leader_ref0": legacy_exact,
        "exact_leader_ref22": exact_ref22,
        "paired_actions_ref22": paired_ref22,
        "follower_only_ref22": follower_ref22,
    }
    arrays: dict[str, np.ndarray] = {"seeds": seeds}
    summaries: dict[str, object] = {}
    for name, omegas in protocols.items():
        arrays[f"{name}__omega"] = omegas
        target = _target_metrics(runtime, seeds, omegas)
        coefficient_error = np.linalg.norm(omegas - runtime.true_omega[None, :], axis=1)
        arrays[f"{name}__coefficient_l2_error"] = coefficient_error
        summaries[name] = {
            "target_suboptimality": _interval(target["stackelberg_suboptimality"]),
            "target_fisher_error": _interval(target["fisher_transfer_error"]),
            "coefficient_l2_error": _interval(coefficient_error),
            "boundary_fraction": float(
                np.mean(
                    np.isclose(np.linalg.norm(omegas, axis=1), runtime.problem.radius, atol=1e-6)
                )
            ),
        }
        arrays[f"{name}__target_suboptimality"] = target["stackelberg_suboptimality"]
        arrays[f"{name}__target_fisher_error"] = target["fisher_transfer_error"]

    paired_task_subopt, paired_task_fisher = _transfer_metrics(
        runtime,
        seeds,
        paired_ref22,
        run_dir,
    )
    arrays["paired_actions_ref22__taskwise_transfer_suboptimality"] = paired_task_subopt
    arrays["paired_actions_ref22__taskwise_transfer_fisher_error"] = paired_task_fisher
    arrays["paired_actions_ref22__averaged_transfer_suboptimality"] = np.mean(
        paired_task_subopt, axis=0
    )
    arrays["paired_actions_ref22__averaged_transfer_fisher_error"] = np.mean(
        paired_task_fisher, axis=0
    )
    summaries["paired_actions_ref22"]["averaged_transfer_suboptimality"] = _interval(
        arrays["paired_actions_ref22__averaged_transfer_suboptimality"]
    )
    summaries["paired_actions_ref22"]["averaged_transfer_fisher_error"] = _interval(
        arrays["paired_actions_ref22__averaged_transfer_fisher_error"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(data_path, arrays)
    atomic_write_json(
        summary_path,
        {
            "protocols": summaries,
            "source": {
                "run_id": json.loads((run_dir / "manifest.json").read_text())["run_id"],
                "manifest_sha256": _sha256(run_dir / "manifest.json"),
                "retained_follower_samples": True,
                "leader_sampling": "independent keyed RNG from exact expert QRE marginal",
                "reference_action": 22,
                "seeds": seeds.tolist(),
            },
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
