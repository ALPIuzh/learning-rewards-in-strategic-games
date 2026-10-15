"""Joint small-budget audit of MLE ridge and Active information regularization.

The paper settings were selected at 1,600 samples.  This diagnostic instead
evaluates a deliberately small, explicit budget while preserving every sampled
trajectory.  It compares three data-collection protocols:

* sequential Active IRL without warm-up, over a ridge/lambda grid;
* an all-warm-up D-optimal design, over the same ridge grid; and
* Stackelberg-expert fixed collection, over the same ridge grid.

Every configuration is evaluated on paired application seeds.  Optional
downstream rewards are fixed before fitting and every supplied reward is kept;
the script never selects tasks within a bucket.  Output directories must be
new or empty.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.core.math import cumulative_fisher_information, follower_probabilities
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.inference.mle import solve_mle_batch
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.artifacts import (
    atomic_write_json,
    atomic_write_npz,
    collect_git_provenance,
    find_repository,
)
from mairl_exp.batching import run_seed_batch

DEFAULT_RIDGES = (1.0e-4, 1.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0, 10.0)
DEFAULT_MULTIPLIERS = (1.0, 3.0, 12.0)
PROTOCOLS = ("active_all_warmup", "active_no_warmup", "fixed_expert_mle")


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_error": float(np.std(array, ddof=1) / np.sqrt(array.size)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _source_metrics(runtime: Any, estimates: np.ndarray, seeds: tuple[int, ...]) -> dict[str, Any]:
    names = (
        "stackelberg_suboptimality",
        "fisher_transfer_error",
        "coefficient_l2_error",
        "self_concordance_distance",
    )
    values = {name: [] for name in names}
    for estimate, seed in zip(estimates, seeds, strict=True):
        evaluated = runtime.evaluate(estimate[None, :], seed)
        for name in names:
            values[name].append(float(evaluated[name][0]))
    return {name: _summary(np.asarray(items)) for name, items in values.items()}


def _information_geometry(runtime: Any, policies: np.ndarray) -> dict[str, Any]:
    eigenvalues = []
    conditions = []
    trace_inverses = []
    log_determinants = []
    for seed_policies in policies:
        information = cumulative_fisher_information(
            runtime.problem,
            seed_policies,
            runtime.true_omega,
        )
        values = np.linalg.eigvalsh(information)
        eigenvalues.append(values)
        if values[0] <= 0.0:
            conditions.append(float("inf"))
            trace_inverses.append(float("inf"))
            log_determinants.append(float("-inf"))
        else:
            conditions.append(float(values[-1] / values[0]))
            trace_inverses.append(float(np.sum(1.0 / values)))
            log_determinants.append(float(np.sum(np.log(values))))
    return {
        "mean_eigenvalues": np.mean(np.stack(eigenvalues), axis=0).tolist(),
        "mean_condition": float(np.mean(conditions)),
        "mean_trace_inverse": float(np.mean(trace_inverses)),
        "mean_log_determinant": float(np.mean(log_determinants)),
    }


def _transfer_metrics(
    *,
    runtime: Any,
    estimates: np.ndarray,
    seeds: tuple[int, ...],
    rewards: np.ndarray,
    task_seed: int,
    sample_count: int,
    leader_config: LeaderOptimizationConfig,
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


def _record(
    *,
    protocol: str,
    ridge: float,
    multiplier: float | None,
    seeds: tuple[int, ...],
    policies: np.ndarray,
    actions: np.ndarray,
    estimates: np.ndarray,
    runtime: Any,
    rewards: np.ndarray | None,
    task_seed: int | None,
    sample_count: int,
    leader_config: LeaderOptimizationConfig,
    elapsed: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = _source_metrics(runtime, estimates, seeds)
    payload: dict[str, Any] = {
        "protocol": protocol,
        "ridge": ridge,
        "lambda_information_multiplier": multiplier,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "sample_count": sample_count,
        "boundary_fraction": float(
            np.mean(
                np.isclose(
                    np.linalg.norm(estimates, axis=1),
                    runtime.problem.radius,
                    atol=1.0e-7,
                )
            )
        ),
        "mean_estimate": np.mean(estimates, axis=0).tolist(),
        "source_metrics": source,
        "information_geometry": _information_geometry(runtime, policies),
        "algorithm_seconds": elapsed,
    }
    arrays = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "policies": policies,
        "follower_actions": actions,
        "estimates": estimates,
    }
    if rewards is not None:
        if task_seed is None:
            raise ValueError("task_seed is required when downstream rewards are supplied")
        transfer_suboptimality, transfer_fisher = _transfer_metrics(
            runtime=runtime,
            estimates=estimates,
            seeds=seeds,
            rewards=rewards,
            task_seed=task_seed,
            sample_count=sample_count,
            leader_config=leader_config,
        )
        payload["transfer_suboptimality"] = _summary(transfer_suboptimality.mean(axis=0))
        payload["transfer_fisher_error"] = _summary(transfer_fisher.mean(axis=0))
        payload["task_count"] = int(rewards.shape[0])
        arrays["taskwise_transfer_suboptimality"] = transfer_suboptimality
        arrays["taskwise_transfer_fisher_error"] = transfer_fisher
    return payload, arrays


def _best_configurations(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    criteria = {
        "source_suboptimality": lambda row: row["source_metrics"][
            "stackelberg_suboptimality"
        ]["mean"],
        "source_fisher_error": lambda row: row["source_metrics"]["fisher_transfer_error"][
            "mean"
        ],
        "coefficient_l2_error": lambda row: row["source_metrics"]["coefficient_l2_error"][
            "mean"
        ],
    }
    if records and "transfer_suboptimality" in records[0]:
        criteria["transfer_suboptimality"] = lambda row: row["transfer_suboptimality"]["mean"]
        criteria["transfer_fisher_error"] = lambda row: row["transfer_fisher_error"]["mean"]
    result = {}
    for name, criterion in criteria.items():
        selected = min(records, key=criterion)
        result[name] = {
            "protocol": selected["protocol"],
            "ridge": selected["ridge"],
            "lambda_information_multiplier": selected["lambda_information_multiplier"],
            "value": float(criterion(selected)),
        }
    return result


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite small-K audit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.reference_run.resolve()
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    numerics = instantiate(cfg.numerics)
    if args.application_eta is None:
        runtime = load_runtime(run_dir)
    else:
        application_config = OmegaConf.merge(
            cfg.application,
            {"eta": float(args.application_eta)},
        )
        runtime = instantiate(application_config).build(numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))

    rewards: np.ndarray | None = None
    task_seed: int | None = None
    if args.tasks_dir is not None:
        tasks_dir = args.tasks_dir.resolve()
        with np.load(tasks_dir / "candidate_tasks.npz") as arrays:
            rewards = arrays["rewards"].copy()
        task_manifest = json.loads((tasks_dir / "task_manifest.json").read_text())
        task_seed = int(task_manifest["task_seed"])
        if args.task_limit is not None:
            rewards = rewards[: args.task_limit]

    manifest = {
        "schema_version": 1,
        "mode": "small-budget joint MLE-ridge and Active-information audit",
        "reference_run": str(run_dir),
        "sample_count": args.sample_count,
        "application_eta_input": float(runtime.metadata["eta"]),
        "compiled_problem_eta": float(runtime.problem.eta),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "ridges": list(args.ridges),
        "lambda_information_multipliers": list(args.multipliers),
        "subtheorem_multipliers_allowed": bool(args.allow_subtheorem_multipliers),
        "protocols": list(args.protocols),
        "task_seed": task_seed,
        "task_count": None if rewards is None else int(rewards.shape[0]),
        "evaluation_git": collect_git_provenance(find_repository(Path.cwd())).manifest(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    no_warmup = numerics.warmup.create(budget=0, enabled=False)
    all_warmup = numerics.warmup.create(budget=args.sample_count, enabled=True)
    from mairl.exploration.warmup import prepare_warmup

    prepared_no_warmup = prepare_warmup(runtime.problem, no_warmup)
    prepared_all_warmup = prepare_warmup(runtime.problem, all_warmup)

    records: list[dict[str, Any]] = []

    def save_record(
        protocol: str,
        ridge: float,
        multiplier: float | None,
        policies: np.ndarray,
        actions: np.ndarray,
        estimates: np.ndarray,
        elapsed: float,
    ) -> None:
        public, arrays = _record(
            protocol=protocol,
            ridge=ridge,
            multiplier=multiplier,
            seeds=seeds,
            policies=policies,
            actions=actions,
            estimates=estimates,
            runtime=runtime,
            rewards=rewards,
            task_seed=task_seed,
            sample_count=args.sample_count,
            leader_config=leader_config,
            elapsed=elapsed,
        )
        suffix = f"ridge_{_slug(ridge)}"
        if multiplier is not None:
            suffix += f"__lambda_{_slug(multiplier)}"
        stem = f"{protocol}__{suffix}"
        atomic_write_json(output_dir / f"{stem}.json", public)
        atomic_write_npz(output_dir / f"{stem}.npz", arrays)
        records.append(public)
        print(
            protocol,
            f"ridge={ridge:g}",
            "" if multiplier is None else f"lambda/eta2={multiplier:g}",
            f"source={public['source_metrics']['stackelberg_suboptimality']['mean']:.6g}",
            (
                ""
                if "transfer_suboptimality" not in public
                else f"transfer={public['transfer_suboptimality']['mean']:.6g}"
            ),
            flush=True,
        )

    if "active_all_warmup" in args.protocols:
        for ridge in args.ridges:
            candidate_numerics = replace(numerics, mle=replace(numerics.mle, ridge=ridge))
            started = perf_counter()
            outputs = run_seed_batch(
                method="active",
                warmup_enabled=True,
                seeds=seeds,
                total_samples=args.sample_count,
                evaluation_interval=args.sample_count,
                lambda_information_multiplier=1.0,
                lambda_warm=float(cfg.experiment.lambda_warm),
                application=runtime,
                numerics=candidate_numerics,
                prepared_warmup=prepared_all_warmup,
            )
            save_record(
                "active_all_warmup",
                ridge,
                None,
                np.stack([output.arrays["policies"] for output in outputs]),
                np.stack([output.arrays["follower_actions"] for output in outputs]),
                np.stack([output.arrays["omega_history"][-1] for output in outputs]),
                perf_counter() - started,
            )

    if "active_no_warmup" in args.protocols:
        for multiplier in args.multipliers:
            for ridge in args.ridges:
                candidate_numerics = replace(numerics, mle=replace(numerics.mle, ridge=ridge))
                started = perf_counter()
                outputs = run_seed_batch(
                    method="active",
                    warmup_enabled=False,
                    seeds=seeds,
                    total_samples=args.sample_count,
                    evaluation_interval=args.sample_count,
                    lambda_information_multiplier=multiplier,
                    lambda_warm=float(cfg.experiment.lambda_warm),
                    application=runtime,
                    numerics=candidate_numerics,
                    prepared_warmup=prepared_no_warmup,
                )
                save_record(
                    "active_no_warmup",
                    ridge,
                    multiplier,
                    np.stack([output.arrays["policies"] for output in outputs]),
                    np.stack([output.arrays["follower_actions"] for output in outputs]),
                    np.stack([output.arrays["omega_history"][-1] for output in outputs]),
                    perf_counter() - started,
                )

    fixed_policy = runtime.fixed_mle_leader_policy
    fixed_probabilities = follower_probabilities(runtime.problem, fixed_policy, runtime.true_omega)
    fixed_policies = np.broadcast_to(
        fixed_policy,
        (len(seeds), args.sample_count, runtime.problem.n_leader_actions),
    ).copy()
    fixed_actions = np.stack(
        [
            experiment_rng(seed, RandomPurpose.EXPERT_FOLLOWER).choice(
                runtime.problem.n_follower_actions,
                size=args.sample_count,
                p=fixed_probabilities,
            )
            for seed in seeds
        ]
    )
    if "fixed_expert_mle" in args.protocols:
        for ridge in args.ridges:
            started = perf_counter()
            fits = solve_mle_batch(
                runtime.problem,
                fixed_policies,
                fixed_actions,
                replace(numerics.mle, ridge=ridge),
            )
            save_record(
                "fixed_expert_mle",
                ridge,
                None,
                fixed_policies,
                fixed_actions,
                np.stack([fit.omega for fit in fits]),
                perf_counter() - started,
            )

    by_protocol = {
        protocol: [record for record in records if record["protocol"] == protocol]
        for protocol in sorted({record["protocol"] for record in records})
    }
    summary = {
        "manifest": manifest,
        "best_by_protocol_and_metric": {
            protocol: _best_configurations(protocol_records)
            for protocol, protocol_records in by_protocol.items()
        },
        "records": records,
    }
    atomic_write_json(output_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=63_000)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--ridges", type=float, nargs="+", default=DEFAULT_RIDGES)
    parser.add_argument("--multipliers", type=float, nargs="+", default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--protocols", choices=PROTOCOLS, nargs="+", default=PROTOCOLS)
    parser.add_argument(
        "--allow-subtheorem-multipliers",
        action="store_true",
        help=(
            "Permit positive lambda/eta^2 values below one as an explicitly "
            "theorem-invalid mechanism diagnostic"
        ),
    )
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument(
        "--application-eta",
        type=float,
        help=(
            "Rebuild the application at this raw-payoff inverse temperature. "
            "This is an environment sensitivity, not a tuned algorithm parameter."
        ),
    )
    args = parser.parse_args()
    if args.sample_count < 1 or args.seed_count < 2:
        parser.error("sample-count must be positive and seed-count must be at least two")
    if not args.ridges or any(value < 0.0 for value in args.ridges):
        parser.error("ridges must be nonempty and nonnegative")
    if not args.multipliers or any(value <= 0.0 for value in args.multipliers):
        parser.error("multipliers must be nonempty and strictly positive")
    if not args.allow_subtheorem_multipliers and any(
        value < 1.0 for value in args.multipliers
    ):
        parser.error(
            "multipliers below one require --allow-subtheorem-multipliers because "
            "they violate the paper regularizer condition"
        )
    if args.task_limit is not None and args.task_limit < 1:
        parser.error("task-limit must be positive")
    if args.application_eta is not None and args.application_eta <= 0.0:
        parser.error("application-eta must be strictly positive")
    run(args)


if __name__ == "__main__":
    main()
