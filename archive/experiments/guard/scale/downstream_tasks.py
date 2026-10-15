"""Downstream-task transfer for the GUARD lobeke_sfg_gsg_2 active-IRL run.

Reuses the reward estimates stored by a completed guard-lobeke2-active run: for each
seed, takes the final omega_hat (at the full sample budget) and asks how well it
supports planning against several leader-reward objectives the estimate was never
fit on -- the "how does one trained omega behave for multiple leader rewards"
question. Every task shares the same game geometry (features, eta, radius, the
learned follower behavior); only the defender's payoff matrix differs, which is
exactly the object the Stackelberg reward-transfer lemma quantifies over.

Mirrors experiments/zero_sum_comparison/downstream_tasks.py's methodology, adapted to
use the batched leader-policy optimizer (mandatory at this game's 1849-action scale --
see experiments/guard/README.md) and a single estimator (active) rather than three.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.guard.application import build_guard_problem
from mairl.exceptions import ConfigurationError
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy_batch,
)
from mairl_exp.artifacts import sha256_file

_ROOT = Path(__file__).resolve().parents[3]
GAME_NAME = "lobeke_sfg_gsg_2"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _draw_diverse_tasks(
    *, shape: tuple[int, int], num_tasks: int, seed: int, scale: float, candidate_pool: int
) -> np.ndarray:
    """Greedily pick num_tasks leader-reward matrices that are pairwise as
    dissimilar as possible (by cosine similarity), out of a random candidate pool."""

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


def _load_final_estimates(run_dir: Path) -> tuple[dict[int, np.ndarray], list[dict[str, str]]]:
    """Per seed: the omega estimate at the final stored checkpoint of a completed
    guard-lobeke2-active run (single method, single warmup mode)."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigurationError(f"run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ConfigurationError(
            f"run is not complete (status={manifest.get('status')!r}); "
            "wait for it to finish, or check with mairl-status"
        )

    estimates: dict[int, np.ndarray] = {}
    inputs: list[dict[str, str]] = []
    for expected in manifest["expected_jobs"]:
        result_path = run_dir / "jobs" / expected["job_id"] / "result.npz"
        status_path = result_path.with_name("status.json")
        if not result_path.exists() or not status_path.exists():
            raise ConfigurationError(f"missing job {expected['label']}")
        status = json.loads(status_path.read_text())
        digest = sha256_file(result_path)
        if status.get("status") != "complete" or status.get("result_sha256") != digest:
            raise ConfigurationError(f"invalid job {expected['label']}")
        with np.load(result_path, allow_pickle=False) as arrays:
            final_omega = arrays["omega_history"][-1].copy()
        seed = int(expected["spec"]["seed"])
        estimates[seed] = final_omega
        inputs.append({"job_id": str(expected["job_id"]), "result_sha256": digest})
    return estimates, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/guard/scale"))
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--task-seed", type=int, default=20260804)
    parser.add_argument("--task-scale", type=float, default=1.0)
    parser.add_argument("--task-candidate-pool", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()

    active_run = _resolve(args.active_run)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading final estimates from {active_run}", flush=True)
    estimates, inputs = _load_final_estimates(active_run)
    seeds = sorted(estimates)
    print(f"loaded {len(seeds)} seeds: {seeds}", flush=True)

    problem, true_omega, leader_reward = build_guard_problem(
        GAME_NAME, feature_mode="behavioral", eta=1.25
    )
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=1)
    )

    tasks = _draw_diverse_tasks(
        shape=leader_reward.shape,
        num_tasks=args.num_tasks,
        seed=args.task_seed,
        scale=args.task_scale,
        candidate_pool=args.task_candidate_pool,
    )
    print(f"drew {args.num_tasks} held-out leader-reward tasks", flush=True)

    omega_estimates = np.stack([estimates[seed] for seed in seeds])
    task_results: list[dict[str, np.ndarray]] = []
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
        task_results.append(
            {"seeds": np.array(seeds), "stackelberg_suboptimality": suboptimalities}
        )
        elapsed = time.perf_counter() - started
        print(
            f"task {task_index + 1}/{args.num_tasks}: "
            f"mean SubOpt={suboptimalities.mean():.4f} +/- {suboptimalities.std():.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    positions = np.arange(args.num_tasks)
    means = [result["stackelberg_suboptimality"].mean() for result in task_results]
    errors = [result["stackelberg_suboptimality"].std() for result in task_results]
    axis.bar(positions, means, yerr=errors, capsize=4, color="tab:blue")
    axis.set(
        xticks=positions,
        xticklabels=[f"Held-out task {i + 1}" for i in range(args.num_tasks)],
        ylabel="Stackelberg sub-optimality",
        title=(
            f"Lobeke ({problem.n_leader_actions} actions): one trained omega "
            f"deployed against {args.num_tasks} held-out leader rewards\n"
            f"(mean +/- 1 std over {len(seeds)} seeds)"
        ),
    )
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure_path = output_dir / "lobeke2_downstream_tasks.png"
    figure.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "active_run_inputs": inputs,
        "seeds": seeds,
        "num_tasks": args.num_tasks,
        "task_results": [
            {
                "mean": float(result["stackelberg_suboptimality"].mean()),
                "std": float(result["stackelberg_suboptimality"].std()),
                "per_seed": result["stackelberg_suboptimality"].tolist(),
            }
            for result in task_results
        ],
    }
    summary_path = output_dir / "lobeke2_downstream_tasks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {figure_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
