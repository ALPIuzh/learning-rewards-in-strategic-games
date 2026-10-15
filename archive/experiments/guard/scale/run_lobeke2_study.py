"""Checkpointed, resumable Tier-2 study on lobeke_sfg_gsg_2 (1,849 actions).

Designed to run unattended for hours: every (method, warmup_enabled, seed) unit of
work is saved to its own .npz file the moment it completes, and the script skips any
unit whose file already exists on startup. If interrupted (crash, machine sleep,
manual kill) at any point, rerunning this exact command resumes from the last
completed unit -- nothing already on disk is recomputed.

Stages, in order:
  1. active   -- own tuned warm-up, the real training run.
  2. fixed_random / random_each_round -- with and without warm-up (the impact-of-
     warm-up comparison), at the same total sample budget as active.
  3. downstream-task transfer -- reuses stage 1's final omega estimates (no
     retraining) against several held-out leader-reward matrices.

Direct standalone script, not going through the Hydra artifact pipeline: at this
game's scale a single SLSQP refinement costs ~70-150s (see experiments/guard/README.md
and the vertex-screening commit), and the generic pipeline's per-batch bookkeeping
showed an unexplained slowdown specifically on warm-up-disabled batches during smoke
testing tonight that there wasn't time to root-cause before an unattended run. This
script calls the same underlying, directly-benchmarked mairl_exp.batching.run_seed_batch
core used everywhere else in this investigation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.guard.application import GuardApplication, build_guard_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.exploration.warmup import prepare_warmup
from mairl.inference.mle import MLEConfig
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy_batch,
)
from mairl_exp.batching import run_seed_batch
from mairl_exp.config import ExperimentNumerics, WarmupNumerics

GAME_NAME = "lobeke_sfg_gsg_2"
SEED_COUNT = 6
SEED_START = 100
TOTAL_SAMPLES = 120
WARMUP_BUDGET = 100
EVALUATION_INTERVAL = 120
RANDOM_STARTS = 16
BATCH_REFINE_STARTS = 1
NUM_DOWNSTREAM_TASKS = 3
DOWNSTREAM_TASK_SEED = 20260804

RESULTS_DIR = Path(__file__).resolve().parent / "lobeke2_results"
LOG_PREFIX = "[lobeke2]"


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} {LOG_PREFIX} {message}", flush=True)


def _numerics() -> ExperimentNumerics:
    policy_config = PolicyOptimizationConfig(
        method="multistart", random_starts=RANDOM_STARTS, batch_refine_starts=BATCH_REFINE_STARTS
    )
    return ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupNumerics(),
        active_policy=policy_config,
        evaluation_policy=policy_config,
    )


def _unit_path(method: str, warmup_enabled: bool, seed: int) -> Path:
    return RESULTS_DIR / f"{method}_warmup{int(warmup_enabled)}_seed{seed}.npz"


def _run_unit(runtime, numerics, *, method: str, warmup_enabled: bool, seed: int, prepared) -> None:
    path = _unit_path(method, warmup_enabled, seed)
    if path.exists():
        _log(f"skip (already complete): {path.name}")
        return
    started = time.perf_counter()
    outputs = run_seed_batch(
        method=method,
        warmup_enabled=warmup_enabled,
        seeds=(seed,),
        total_samples=TOTAL_SAMPLES,
        evaluation_interval=EVALUATION_INTERVAL,
        lambda_information_multiplier=1.0,
        lambda_warm=0.01,
        application=runtime,
        numerics=numerics,
        prepared_warmup=prepared,
    )
    output = outputs[0]
    elapsed = time.perf_counter() - started
    final_subopt = float(output.summary["final_metrics"]["stackelberg_suboptimality"])
    np.savez(
        path,
        trace_sample_counts=output.arrays["trace_sample_counts"],
        omega_history=output.arrays["omega_history"],
        evaluation_sample_counts=output.arrays["evaluation_sample_counts"],
        metric__stackelberg_suboptimality=output.arrays["metric__stackelberg_suboptimality"],
    )
    _log(f"done: {path.name}  final_subopt={final_subopt:.5f}  elapsed={elapsed:.1f}s")


def stage_active(runtime, numerics, prepared_with_warmup) -> None:
    _log("=== stage 1: active (own warm-up) ===")
    for seed in range(SEED_START, SEED_START + SEED_COUNT):
        _run_unit(
            runtime,
            numerics,
            method="active",
            warmup_enabled=True,
            seed=seed,
            prepared=prepared_with_warmup,
        )


def stage_baselines(runtime, numerics, prepared_with_warmup, prepared_no_warmup) -> None:
    _log("=== stage 2: baselines (fixed_random, random_each_round; warm-up on/off) ===")
    for method in ("fixed_random", "random_each_round"):
        for warmup_enabled, prepared in ((True, prepared_with_warmup), (False, prepared_no_warmup)):
            for seed in range(SEED_START, SEED_START + SEED_COUNT):
                _run_unit(
                    runtime,
                    numerics,
                    method=method,
                    warmup_enabled=warmup_enabled,
                    seed=seed,
                    prepared=prepared,
                )


def _draw_diverse_tasks(*, shape, num_tasks, seed, scale=1.0, candidate_pool=50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(-scale, scale, size=(candidate_pool, *shape))
    flattened = candidates.reshape(candidate_pool, -1)
    unit = flattened / np.linalg.norm(flattened, axis=1, keepdims=True)
    selected = [0]
    while len(selected) < num_tasks:
        similarity = np.abs(unit @ unit[selected].T).max(axis=1)
        similarity[selected] = np.inf
        selected.append(int(np.argmin(similarity)))
    return candidates[selected]


def stage_downstream(problem, true_omega, defender_reward, numerics) -> None:
    _log("=== stage 3: downstream-task transfer ===")
    summary_path = RESULTS_DIR / "downstream_summary.json"
    if summary_path.exists():
        _log(f"skip (already complete): {summary_path.name}")
        return

    final_omegas = []
    seeds = list(range(SEED_START, SEED_START + SEED_COUNT))
    for seed in seeds:
        path = _unit_path("active", True, seed)
        with np.load(path) as arrays:
            final_omegas.append(arrays["omega_history"][-1])
    omega_estimates = np.stack(final_omegas)

    tasks = _draw_diverse_tasks(
        shape=defender_reward.shape, num_tasks=NUM_DOWNSTREAM_TASKS, seed=DOWNSTREAM_TASK_SEED
    )
    config = LeaderOptimizationConfig(numerics.evaluation_policy)
    task_results = []
    for task_index, task_reward in enumerate(tasks):
        started = time.perf_counter()
        true_solution = optimize_leader_policy_batch(
            problem, true_omega[None, :], task_reward, config, (0,)
        )[0]
        estimated_solutions = optimize_leader_policy_batch(
            problem, omega_estimates, task_reward, config, tuple(seeds)
        )
        suboptimalities = np.empty(len(seeds), dtype=np.float64)
        for index, solution in enumerate(estimated_solutions):
            deployed_value, _ = leader_value_and_gradient(
                problem, true_omega, task_reward, solution.policy
            )
            suboptimalities[index] = max(0.0, true_solution.value - deployed_value)
        task_results.append(suboptimalities)
        elapsed = time.perf_counter() - started
        mean_subopt = suboptimalities.mean()
        _log(
            f"task {task_index + 1}/{NUM_DOWNSTREAM_TASKS}: mean SubOpt={mean_subopt:.5f} "
            f"+/- {suboptimalities.std():.5f}  elapsed={elapsed:.1f}s"
        )

    summary = {
        "seeds": seeds,
        "task_results": [
            {"mean": float(r.mean()), "std": float(r.std()), "per_seed": r.tolist()}
            for r in task_results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {summary_path}")


def render_plots() -> None:
    _log("=== rendering plots from completed units ===")

    def _load(method: str, warmup_enabled: bool) -> tuple[np.ndarray, np.ndarray] | None:
        finals = []
        for seed in range(SEED_START, SEED_START + SEED_COUNT):
            path = _unit_path(method, warmup_enabled, seed)
            if not path.exists():
                return None
            with np.load(path) as arrays:
                finals.append(float(arrays["metric__stackelberg_suboptimality"][-1]))
        return np.array(finals), np.array(finals)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    labels, means, errs = [], [], []
    for method, warmup_enabled, label in (
        ("active", True, "Active\n(own warm-up)"),
        ("fixed_random", False, "Fixed random\n(no warm-up)"),
        ("random_each_round", False, "Random each round\n(no warm-up)"),
        ("fixed_random", True, "Fixed random\n(shared warm-up)"),
        ("random_each_round", True, "Random each round\n(shared warm-up)"),
    ):
        loaded = _load(method, warmup_enabled)
        if loaded is None:
            continue
        finals, _ = loaded
        labels.append(label)
        means.append(finals.mean())
        errs.append(finals.std())
    if labels:
        positions = np.arange(len(labels))
        axis.bar(positions, means, yerr=errs, capsize=4, color="tab:blue")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("Final Stackelberg sub-optimality")
        axis.set_title(f"Lobeke ({SEED_COUNT} seeds, {TOTAL_SAMPLES}-sample budget)")
        axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        output_path = RESULTS_DIR / "lobeke2_methods_comparison.png"
        figure.savefig(output_path, dpi=170, bbox_inches="tight")
        _log(f"wrote {output_path}")
    plt.close(figure)

    downstream_path = RESULTS_DIR / "downstream_summary.json"
    if downstream_path.exists():
        summary = json.loads(downstream_path.read_text())
        figure, axis = plt.subplots(figsize=(7.0, 4.8))
        positions = np.arange(len(summary["task_results"]))
        means = [r["mean"] for r in summary["task_results"]]
        errs = [r["std"] for r in summary["task_results"]]
        axis.bar(positions, means, yerr=errs, capsize=4, color="tab:green")
        axis.set_xticks(positions)
        axis.set_xticklabels([f"Task {i + 1}" for i in positions])
        axis.set_ylabel("Stackelberg sub-optimality")
        axis.set_title("One trained omega deployed against held-out leader rewards")
        axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        output_path = RESULTS_DIR / "lobeke2_downstream_tasks.png"
        figure.savefig(output_path, dpi=170, bbox_inches="tight")
        _log(f"wrote {output_path}")
        plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"starting; results directory: {RESULTS_DIR}")
    numerics = _numerics()
    app = GuardApplication(game_name=GAME_NAME, feature_mode="behavioral")
    runtime = app.build(numerics)
    problem = runtime.problem
    _, true_omega, defender_reward = build_guard_problem(
        GAME_NAME, feature_mode="behavioral", eta=app.eta
    )
    _log(
        f"game={GAME_NAME}  n_leader_actions={problem.n_leader_actions}  "
        f"n_follower_actions={problem.n_follower_actions}"
    )

    warmup_config_on = numerics.warmup.create(budget=WARMUP_BUDGET, enabled=True)
    warmup_config_off = numerics.warmup.create(budget=0, enabled=False)
    _log("preparing warm-up design (with warm-up)...")
    prepared_with_warmup = prepare_warmup(problem, warmup_config_on)
    _log(f"warm-up support size={prepared_with_warmup.design.support_size}")
    prepared_no_warmup = prepare_warmup(problem, warmup_config_off)

    stage_active(runtime, numerics, prepared_with_warmup)
    stage_baselines(runtime, numerics, prepared_with_warmup, prepared_no_warmup)
    stage_downstream(problem, true_omega, defender_reward, numerics)
    render_plots()
    _log("=== all stages complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log(
            "FATAL ERROR -- traceback follows; completed units on disk are safe and "
            "will be skipped on rerun"
        )
        traceback.print_exc()
        sys.exit(1)
