"""Refit saved 400-sample datasets over a shared MLE-ridge grid.

This is a fixed-design sensitivity audit.  It deliberately holds every saved
leader policy and sampled follower action fixed, so it isolates estimator
regularization from adaptive data-collection feedback.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.guard.paper.figures.data import MethodKey, SeedTrace
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from investigations.behavioral_transfer.pilot import METHODS
from mairl.inference.mle import solve_mle_batch
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.artifacts import (
    atomic_write_json,
    atomic_write_npz,
    collect_git_provenance,
    find_repository,
)

DEFAULT_RIDGES = (1.0e-4, 1.0e-2, 1.0e-1, 1.0, 3.0)


@dataclass(frozen=True, slots=True)
class SavedDataset:
    key: MethodKey
    seeds: tuple[int, ...]
    policies: np.ndarray
    follower_actions: np.ndarray


def _slug(key: MethodKey) -> str:
    return METHODS[key]


def _load_saved_datasets(run_dir: Path) -> dict[MethodKey, SavedDataset]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    grouped: dict[MethodKey, list[tuple[int, np.ndarray, np.ndarray]]] = {
        key: [] for key in METHODS
    }
    for entry in manifest["expected_jobs"]:
        spec = entry["spec"]
        key: MethodKey = (spec["method"], bool(spec["warmup_enabled"]))
        if key not in grouped:
            continue
        with np.load(run_dir / "jobs" / entry["job_id"] / "result.npz") as arrays:
            grouped[key].append(
                (
                    int(spec["seed"]),
                    arrays["policies"].copy(),
                    arrays["follower_actions"].copy(),
                )
            )
    datasets = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        datasets[key] = SavedDataset(
            key=key,
            seeds=tuple(row[0] for row in rows),
            policies=np.stack([row[1] for row in rows]),
            follower_actions=np.stack([row[2] for row in rows]),
        )
    if any(not dataset.seeds for dataset in datasets.values()):
        raise ValueError("source run is missing a ridge-audit method")
    common_seeds = {dataset.seeds for dataset in datasets.values()}
    if len(common_seeds) != 1:
        raise ValueError("ridge-audit methods do not share paired seeds")
    return datasets


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "standard_error": float(np.std(values, ddof=1) / np.sqrt(values.size)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _source_metrics(runtime: Any, estimates: np.ndarray, seeds: tuple[int, ...]) -> dict[str, Any]:
    names = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")
    values = {name: [] for name in names}
    for estimate, seed in zip(estimates, seeds, strict=True):
        evaluated = runtime.evaluate(estimate[None, :], seed)
        for name in names:
            values[name].append(float(evaluated[name][0]))
    return {name: _summary(np.asarray(items)) for name, items in values.items()}


def _transfer_metrics(
    *,
    runtime: Any,
    estimates: np.ndarray,
    seeds: tuple[int, ...],
    rewards: np.ndarray,
    leader_config: LeaderOptimizationConfig,
    task_seed: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    traces = [
        SeedTrace(
            seed=seed,
            checkpoints=np.asarray([sample_count], dtype=np.int64),
            metrics={},
            omega_history=estimate[None, :],
        )
        for seed, estimate in zip(seeds, estimates, strict=True)
    ]
    suboptimality = []
    fisher_error = []
    for task_index, reward in enumerate(rewards):
        task_suboptimality, task_fisher = task_transfer_metrics(
            runtime.problem,
            runtime.true_omega,
            reward,
            leader_config,
            traces,
            task_seed=task_seed,
            task_index=task_index,
        )
        suboptimality.append(task_suboptimality[:, 0])
        fisher_error.append(task_fisher[:, 0])
    return np.stack(suboptimality), np.stack(fisher_error)


def _render(results: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    colors = {
        "active_warm": "#1671B9",
        "active_no_warm": "#6A4C93",
        "fixed_expert_mle": "#D97706",
    }
    for slug, color in colors.items():
        rows = sorted(
            (result for result in results if result["method"] == slug),
            key=lambda item: item["ridge"],
        )
        ridges = np.asarray([row["ridge"] for row in rows])
        axes[0].plot(
            ridges,
            [row["source_metrics"]["stackelberg_suboptimality"]["mean"] for row in rows],
            marker="o",
            color=color,
            label=slug,
        )
        axes[1].plot(
            ridges,
            [row["source_metrics"]["coefficient_l2_error"]["mean"] for row in rows],
            marker="o",
            color=color,
        )
        axes[2].plot(
            ridges,
            [row["transfer_suboptimality"]["mean"] for row in rows],
            marker="o",
            color=color,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("MLE ridge")
    axes[0].set_ylabel("mean source SubOpt")
    axes[1].set_ylabel("mean coefficient L2 error")
    axes[2].set_ylabel("mean transfer SubOpt")
    axes[0].legend(frameon=False, fontsize=7)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite ridge audit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir.resolve()
    datasets = _load_saved_datasets(run_dir)
    runtime = load_runtime(run_dir)
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    with np.load(args.tasks_dir.resolve() / "candidate_tasks.npz") as arrays:
        rewards = arrays["rewards"].copy()
    task_manifest = json.loads((args.tasks_dir.resolve() / "task_manifest.json").read_text())
    if args.task_limit is not None:
        rewards = rewards[: args.task_limit]

    audit_manifest = {
        "schema_version": 1,
        "mode": (
            "fixed-design ridge sensitivity: saved policies/actions are held fixed; "
            "Active collection is not rerun under counterfactual estimates"
        ),
        "source_run": json.loads((run_dir / "manifest.json").read_text())["run_id"],
        "source_tasks": str(args.tasks_dir.resolve()),
        "task_seed": task_manifest["task_seed"],
        "task_count": int(rewards.shape[0]),
        "ridges": list(args.ridges),
        "evaluation_git": collect_git_provenance(find_repository(Path.cwd())).manifest(),
    }
    atomic_write_json(output_dir / "manifest.json", audit_manifest)

    public_results = []
    for key, dataset in datasets.items():
        slug = _slug(key)
        for ridge in args.ridges:
            fits = solve_mle_batch(
                runtime.problem,
                dataset.policies,
                dataset.follower_actions,
                replace(numerics.mle, ridge=ridge),
            )
            estimates = np.stack([fit.omega for fit in fits])
            transfer_suboptimality, transfer_fisher = _transfer_metrics(
                runtime=runtime,
                estimates=estimates,
                seeds=dataset.seeds,
                rewards=rewards,
                leader_config=leader_config,
                task_seed=int(task_manifest["task_seed"]),
                sample_count=dataset.policies.shape[1],
            )
            result = {
                "method": slug,
                "ridge": ridge,
                "seeds": list(dataset.seeds),
                "source_metrics": _source_metrics(runtime, estimates, dataset.seeds),
                "transfer_suboptimality": _summary(transfer_suboptimality.mean(axis=0)),
                "transfer_fisher_error": _summary(transfer_fisher.mean(axis=0)),
                "mean_estimate": np.mean(estimates, axis=0).tolist(),
                "boundary_fraction": float(
                    np.mean(
                        np.isclose(
                            np.linalg.norm(estimates, axis=1),
                            runtime.problem.radius,
                            atol=1.0e-7,
                        )
                    )
                ),
            }
            stem = f"{slug}__ridge_{ridge:g}".replace(".", "p")
            atomic_write_json(output_dir / f"{stem}.json", result)
            atomic_write_npz(
                output_dir / f"{stem}.npz",
                {
                    "estimates": estimates,
                    "transfer_suboptimality": transfer_suboptimality,
                    "transfer_fisher_error": transfer_fisher,
                },
            )
            public_results.append(result)
            print(f"completed {slug} ridge={ridge:g}", flush=True)

    selections = {}
    for slug in METHODS.values():
        candidates = [result for result in public_results if result["method"] == slug]
        selections[slug] = min(
            candidates,
            key=lambda result: result["source_metrics"]["stackelberg_suboptimality"]["mean"],
        )["ridge"]
    atomic_write_json(
        output_dir / "summary.json",
        {
            **audit_manifest,
            "posthoc_source_suboptimality_minimizers": selections,
            "results": public_results,
        },
    )
    _render(public_results, output_dir / "ridge_sensitivity.png")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridges", type=float, nargs="+", default=DEFAULT_RIDGES)
    parser.add_argument("--task-limit", type=int)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
