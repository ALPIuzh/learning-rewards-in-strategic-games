"""Lightweight warm-up budget sweep for the GUARD behavioral feature mode.

Mirrors the intent of mairl_exp.warmup_sweep.WarmupSweepStudy / WarmupSweepPlot (as used
in conf/experiment/toy_warmup_sweep.yaml): for a fixed total sample budget, reallocating
more of it to the offline D-optimal warm-up trades off against fewer adaptively-selected
online rounds. This script runs that comparison directly through run_seed_batch instead
of the full resumable/artifact-tracked Hydra pipeline, since the goal here is picking a
reasonable budget for a validation run, not a publishable tracked study.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.guard.application import GuardApplication
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.exploration.warmup import prepare_warmup
from mairl.inference.mle import MLEConfig
from mairl_exp.batching import run_seed_batch
from mairl_exp.config import ExperimentNumerics, WarmupNumerics

OUTPUT_DIR = Path(__file__).resolve().parent


def _numerics(random_starts: int = 16) -> ExperimentNumerics:
    policy_config = PolicyOptimizationConfig(method="multistart", random_starts=random_starts)
    return ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupNumerics(),
        active_policy=policy_config,
        evaluation_policy=policy_config,
    )


def sweep(
    game_name: str,
    *,
    budgets: tuple[int, ...],
    total_samples: int,
    seed_count: int,
    seed_start: int = 5000,
) -> dict[int, dict[str, np.ndarray]]:
    numerics = _numerics()
    app = GuardApplication(game_name=game_name, feature_mode="behavioral")
    runtime = app.build(numerics)
    problem = runtime.problem
    seeds = tuple(range(seed_start, seed_start + seed_count))

    results: dict[int, dict[str, np.ndarray]] = {}

    # budget = 0: no warm-up at all, for either method.
    zero_config = numerics.warmup.create(budget=0, enabled=False)
    zero_prepared = prepare_warmup(problem, zero_config)
    results[0] = {}
    for method in ("active", "random_each_round"):
        outputs = run_seed_batch(
            method=method,
            warmup_enabled=False,
            seeds=seeds,
            total_samples=total_samples,
            evaluation_interval=total_samples,
            lambda_information_multiplier=1.0,
            lambda_warm=0.01,
            application=runtime,
            numerics=numerics,
            prepared_warmup=zero_prepared,
        )
        results[0][method] = np.array(
            [o.summary["final_metrics"]["stackelberg_suboptimality"] for o in outputs]
        )

    for budget in budgets:
        config = numerics.warmup.create(budget=budget, enabled=True)
        prepared = prepare_warmup(problem, config)
        results[budget] = {}
        for method in ("active", "random_each_round"):
            outputs = run_seed_batch(
                method=method,
                warmup_enabled=True,
                seeds=seeds,
                total_samples=total_samples,
                evaluation_interval=total_samples,
                lambda_information_multiplier=1.0,
                lambda_warm=0.01,
                application=runtime,
                numerics=numerics,
                prepared_warmup=prepared,
            )
            results[budget][method] = np.array(
                [o.summary["final_metrics"]["stackelberg_suboptimality"] for o in outputs]
            )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="nw_chinatown_sfg_isg_1")
    parser.add_argument("--budgets", type=int, nargs="+", default=[40, 80, 120, 200, 300])
    parser.add_argument("--total-samples", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=24)
    args = parser.parse_args()

    started = time.perf_counter()
    results = sweep(
        args.game,
        budgets=tuple(args.budgets),
        total_samples=args.total_samples,
        seed_count=args.seed_count,
    )
    print(f"sweep wall time: {time.perf_counter() - started:0.1f}s", flush=True)

    budgets = sorted(results)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for method, label in (("active", "Active IRL"), ("random_each_round", "Random each round")):
        means = np.array([results[b][method].mean() for b in budgets])
        ses = np.array(
            [results[b][method].std(ddof=1) / np.sqrt(len(results[b][method])) for b in budgets]
        )
        print(f"{method}:")
        for b, m, s in zip(budgets, means, ses, strict=True):
            print(f"    budget={b:>4d}  final SubOpt = {m:.5f} +/- {s:.5f}")
        axis.errorbar(budgets, means, yerr=1.96 * ses, marker="o", capsize=3, label=label)
    axis.set(
        xlabel=f"Warm-up budget (out of fixed total={args.total_samples} samples)",
        ylabel="Final Stackelberg sub-optimality",
        yscale="log",
        title=f"Warm-up budget sweep: {args.game}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = OUTPUT_DIR / f"warmup_sweep_{args.game}.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
