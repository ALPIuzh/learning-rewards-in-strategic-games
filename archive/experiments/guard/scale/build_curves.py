"""Build proper convergence curves from lobeke2_results' already-stored trajectories.

The main study only evaluated Stackelberg sub-optimality at two checkpoints (start,
end) to keep cost down. But every unit's .npz already stores the full per-round omega
trajectory (run_seed_batch's omega_history), so building a real curve doesn't need to
re-run any sampling -- the expensive part -- only additional evaluation calls (SubOpt
only, not fisher-transfer, to keep this affordable) at a handful of intermediate
sample counts, batched across seeds exactly like the main study.

Checkpointed the same way as run_lobeke2_study.py: each (method, warmup_enabled)
curve is saved to its own .json the moment it's fully computed, and skipped on rerun.
"""

from __future__ import annotations

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

import experiments.guard.scale.run_lobeke2_study as study
from experiments.guard.application import build_guard_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy_batch,
)

GAME_NAME = study.GAME_NAME
SEED_START = study.SEED_START
SEED_COUNT = study.SEED_COUNT
RESULTS_DIR = study.RESULTS_DIR
CURVES_DIR = RESULTS_DIR / "curves"

# (method, warmup_enabled, label, checkpoints)
TARGETS = [
    ("active", True, "Active IRL (own warm-up)", [100, 105, 110, 115, 120]),
    ("fixed_random", False, "Fixed random (no warm-up)", [0, 30, 60, 90, 120]),
    ("random_each_round", False, "Random each round (no warm-up)", [0, 30, 60, 90, 120]),
]


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [curves] {message}", flush=True)


def _load_trajectories(method: str, warmup_enabled: bool) -> dict[int, np.ndarray]:
    seeds = list(range(SEED_START, SEED_START + SEED_COUNT))
    per_seed = []
    for seed in seeds:
        path = study._unit_path(method, warmup_enabled, seed)
        with np.load(path) as arrays:
            per_seed.append((arrays["trace_sample_counts"].copy(), arrays["omega_history"].copy()))
    return per_seed


def build_curve(
    problem,
    leader_reward,
    config,
    *,
    method: str,
    warmup_enabled: bool,
    label: str,
    checkpoints: list[int],
) -> None:
    curve_path = CURVES_DIR / f"{method}_warmup{int(warmup_enabled)}.json"
    if curve_path.exists():
        _log(f"skip (already complete): {curve_path.name}")
        return

    trajectories = _load_trajectories(method, warmup_enabled)
    seeds = tuple(range(SEED_START, SEED_START + SEED_COUNT))

    results = {}
    for checkpoint in checkpoints:
        omegas = []
        for trace_counts, omega_history in trajectories:
            position = np.searchsorted(trace_counts, checkpoint)
            if position >= trace_counts.size or trace_counts[position] != checkpoint:
                raise ValueError(f"checkpoint {checkpoint} not found in stored trace for {method}")
            omegas.append(omega_history[position])
        omegas = np.stack(omegas)

        started = time.perf_counter()
        solutions = optimize_leader_policy_batch(problem, omegas, leader_reward, config, seeds)
        true_solution = _true_solution_cache["value"]
        suboptimalities = []
        for solution in solutions:
            deployed_value, _ = leader_value_and_gradient(
                problem, _true_solution_cache["omega"], leader_reward, solution.policy
            )
            suboptimalities.append(max(0.0, true_solution - deployed_value))
        elapsed = time.perf_counter() - started
        results[checkpoint] = suboptimalities
        _log(
            f"{label} @ {checkpoint} samples: mean SubOpt={np.mean(suboptimalities):.5f} "
            f"elapsed={elapsed:.1f}s"
        )

    curve_path.write_text(
        json.dumps(
            {"checkpoints": checkpoints, "per_checkpoint": results, "label": label}, indent=2
        )
    )
    _log(f"wrote {curve_path}")


_true_solution_cache: dict[str, object] = {}


def main() -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    problem, true_omega, leader_reward = build_guard_problem(
        GAME_NAME, feature_mode="behavioral", eta=1.25
    )
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=1)
    )
    _log("computing true optimal leader value (one-time)...")
    true_solution = optimize_leader_policy_batch(
        problem, true_omega[None, :], leader_reward, config, (0,)
    )[0]
    _true_solution_cache["value"] = true_solution.value
    _true_solution_cache["omega"] = true_omega
    _log(f"true optimal value: {true_solution.value:.4f}")

    for method, warmup_enabled, label, checkpoints in TARGETS:
        build_curve(
            problem,
            leader_reward,
            config,
            method=method,
            warmup_enabled=warmup_enabled,
            label=label,
            checkpoints=checkpoints,
        )

    render()


def render() -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    warmup_end = 100
    for method, warmup_enabled, label, _ in TARGETS:
        curve_path = CURVES_DIR / f"{method}_warmup{int(warmup_enabled)}.json"
        if not curve_path.exists():
            continue
        data = json.loads(curve_path.read_text())
        checkpoints = sorted(int(c) for c in data["per_checkpoint"])
        medians, lowers, uppers = [], [], []
        for checkpoint in checkpoints:
            values = np.array(data["per_checkpoint"][str(checkpoint)])
            medians.append(np.median(values))
            lowers.append(np.percentile(values, 10))
            uppers.append(np.percentile(values, 90))
        line = axis.plot(
            checkpoints, np.maximum(medians, 1e-9), marker="o", label=label, linewidth=1.8
        )[0]
        axis.fill_between(
            checkpoints,
            np.maximum(lowers, 1e-9),
            np.maximum(uppers, 1e-9),
            color=line.get_color(),
            alpha=0.15,
        )
    axis.axvspan(0, warmup_end, color="black", alpha=0.035, zorder=0)
    axis.axvline(warmup_end, color="0.35", linestyle="--", linewidth=1.0)
    axis.text(
        warmup_end,
        0.02,
        f"Active warm-up ends: {warmup_end}",
        transform=axis.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        color="0.35",
        fontsize=8,
    )
    axis.set(
        xlabel="Follower samples",
        ylabel="Stackelberg sub-optimality (median, 10-90th pct.)",
        yscale="log",
        title=f"Lobeke ({SEED_COUNT} seeds, 120-sample budget)",
    )
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = RESULTS_DIR / "lobeke2_convergence_curves.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    _log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
