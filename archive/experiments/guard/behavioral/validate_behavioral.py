"""Standalone smoke validation of the 'behavioral' (d=3) GUARD feature mode.

Compares Active IRL (Algorithm 2, including its own required D-optimal warm-up) against
a naive random-policy baseline that gets no warm-up at all, on real GUARD security games
(Etosha anti-poaching, Chinatown infrastructure protection) using the low-dimensional
SUQR-style feature map. This is the fair comparison: nobody would hand-design a Frank-
Wolfe D-optimal experiment for a policy that is about to act uniformly at random anyway,
so giving the random baseline a free structured warm-up (as an earlier version of this
script did) dilutes exactly the effect the paper is making a claim about.

This is a fast correctness check, not the full paper-grade multi-seed study (that goes
through the Hydra/artifact pipeline in mairl_exp).
"""

from __future__ import annotations

import argparse
import os
import time
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
from mairl_exp.batching import run_seed_batch
from mairl_exp.config import ExperimentNumerics, WarmupNumerics

OUTPUT_DIR = Path(__file__).resolve().parent
SEED_COUNT = 32
SEED_START = 1000
TOTAL_SAMPLES = 400
WARMUP_BUDGET = 120
EVALUATION_INTERVAL = 20
LAMBDA_INFORMATION_MULTIPLIER = 1.0
LAMBDA_WARM = 0.01
RANDOM_STARTS = 16

# Chinatown's budget=120 is empirically tuned (see tune_warmup.py); Etosha reuses it
# as a reasonable default pending its own sweep.
GAMES = {
    "etosha_sfg_gsg_1": ("Etosha anti-poaching (7 targets, 169 patrols)", WARMUP_BUDGET),
    "nw_chinatown_sfg_isg_1": ("Chinatown infrastructure (8 targets, 23 patrols)", WARMUP_BUDGET),
}


def _numerics(random_starts: int) -> ExperimentNumerics:
    policy_config = PolicyOptimizationConfig(method="multistart", random_starts=random_starts)
    return ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupNumerics(),
        active_policy=policy_config,
        evaluation_policy=policy_config,
    )


def _run_method(
    runtime,
    *,
    method: str,
    warmup_enabled: bool,
    warmup_budget: int,
    seed_count: int,
    total_samples: int,
    evaluation_interval: int,
    numerics: ExperimentNumerics,
    log_prefix: str,
):
    problem = runtime.problem
    effective_budget = warmup_budget if warmup_enabled else 0
    warmup_config = numerics.warmup.create(budget=effective_budget, enabled=warmup_enabled)
    prepared = prepare_warmup(problem, warmup_config)
    seeds = tuple(range(SEED_START, SEED_START + seed_count))

    started = time.perf_counter()
    last_reported = -1

    def _progress(seed_index: int, completed: int, total: int) -> None:
        nonlocal last_reported
        if completed != last_reported:
            last_reported = completed
            elapsed = time.perf_counter() - started
            print(
                f"    [{log_prefix}] {completed}/{total} samples "
                f"(seed_index={seed_index}, elapsed={elapsed:0.1f}s)",
                flush=True,
            )

    outputs = run_seed_batch(
        method=method,
        warmup_enabled=warmup_enabled,
        seeds=seeds,
        total_samples=total_samples,
        evaluation_interval=evaluation_interval,
        lambda_information_multiplier=LAMBDA_INFORMATION_MULTIPLIER,
        lambda_warm=LAMBDA_WARM,
        application=runtime,
        numerics=numerics,
        prepared_warmup=prepared,
    )
    print(f"  [{log_prefix}] done in {time.perf_counter() - started:0.1f}s", flush=True)
    return outputs, effective_budget


def _run_game(
    game_name: str,
    *,
    warmup_budget: int,
    seed_count: int,
    total_samples: int,
    evaluation_interval: int,
    random_starts: int,
) -> dict[str, object]:
    numerics = _numerics(random_starts)
    app = GuardApplication(game_name=game_name, feature_mode="behavioral")
    runtime = app.build(numerics)
    problem = runtime.problem
    _, true_omega, _ = build_guard_problem(game_name, feature_mode="behavioral", eta=app.eta)
    print(
        f"  [{game_name}] n_leader_actions={problem.n_leader_actions} "
        f"n_follower_actions={problem.n_follower_actions} feature_dim={problem.feature_dimension}",
        flush=True,
    )

    active_outputs, active_warmup_end = _run_method(
        runtime,
        method="active",
        warmup_enabled=True,
        warmup_budget=warmup_budget,
        seed_count=seed_count,
        total_samples=total_samples,
        evaluation_interval=evaluation_interval,
        numerics=numerics,
        log_prefix=f"{game_name}/active (own warm-up={warmup_budget})",
    )
    random_outputs, _ = _run_method(
        runtime,
        method="random_each_round",
        warmup_enabled=False,
        warmup_budget=0,
        seed_count=seed_count,
        total_samples=total_samples,
        evaluation_interval=evaluation_interval,
        numerics=numerics,
        log_prefix=f"{game_name}/random (no warm-up)",
    )

    return {
        "game_name": game_name,
        "problem": problem,
        "true_omega": true_omega,
        "active_outputs": active_outputs,
        "random_outputs": random_outputs,
        "active_warmup_end": active_warmup_end,
    }


def _mean_ci(
    values: np.ndarray, *, floor_at_zero: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, mean.copy(), mean.copy()
    se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    lower = mean - 1.96 * se
    if floor_at_zero:
        lower = np.maximum(0.0, lower)
    return mean, lower, mean + 1.96 * se


def _median_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median with a 10th-90th percentile band.

    Stackelberg sub-optimality is bounded at zero and can be exactly (or near) zero for
    individual seeds on well-conditioned problems, which breaks the mean +/- 1.96*SE
    normal approximation once the lower bound goes negative and has to be floor-clipped
    for a log-scale plot. Percentiles are the honest choice for a skewed, zero-inflated
    nonnegative metric.
    """

    median = np.median(values, axis=0)
    if values.shape[0] < 2:
        return median, median.copy(), median.copy()
    lower = np.percentile(values, 10, axis=0)
    upper = np.percentile(values, 90, axis=0)
    return median, lower, upper


def _plot_comparison(record: dict, title: str, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    metrics = [
        ("stackelberg_suboptimality", "Stackelberg sub-optimality (median, 10-90th pct.)"),
        ("coefficient_l2_error", "Coefficient L2 error"),
    ]
    warmup_end = record["active_warmup_end"]
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        band_fn = (
            _median_band
            if metric == "stackelberg_suboptimality"
            else (lambda v: _mean_ci(v, floor_at_zero=True))
        )
        for outputs, method_label, color in (
            (record["active_outputs"], "Active IRL (own warm-up)", "tab:blue"),
            (record["random_outputs"], "Random policy each round (no warm-up)", "tab:orange"),
        ):
            checkpoints = outputs[0].arrays["evaluation_sample_counts"]
            values = np.stack([o.arrays[f"metric__{metric}"] for o in outputs])
            center, lower, upper = band_fn(values)
            line = axis.plot(
                checkpoints,
                np.maximum(center, 1e-12),
                label=method_label,
                linewidth=1.8,
                color=color,
            )[0]
            axis.fill_between(
                checkpoints,
                np.maximum(lower, 1e-12),
                np.maximum(upper, 1e-12),
                color=line.get_color(),
                alpha=0.15,
            )
        axis.axvspan(0, warmup_end, color="black", alpha=0.035, zorder=0)
        axis.axvline(warmup_end, color="0.35", linestyle="--", linewidth=1.0)
        axis.set(xlabel="Follower samples", ylabel=label, yscale="log")
        axis.grid(alpha=0.25)
    axes[0].text(
        warmup_end,
        0.02,
        f"Active warm-up ends: {warmup_end}",
        transform=axes[0].get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        color="0.35",
        fontsize=8,
    )
    axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_coefficients(record: dict, title: str, output_path: Path) -> None:
    true_omega = record["true_omega"]
    labels = ["coverage", "value", "coverage x value"]
    figure, axis = plt.subplots(figsize=(6.0, 4.4))
    width = 0.25
    x = np.arange(len(labels))
    axis.bar(x - width, true_omega, width=width, label="True omega", color="0.25")
    for offset, outputs, method_label in (
        (0.0, record["active_outputs"], "Active IRL (own warm-up)"),
        (width, record["random_outputs"], "Random each round (no warm-up)"),
    ):
        finals = np.stack([o.arrays["omega_history"][-1] for o in outputs])
        mean, lower, upper = _mean_ci(finals)
        errors = np.stack([mean - lower, upper - mean])
        axis.bar(
            x + offset,
            mean,
            width=width,
            yerr=errors,
            capsize=3,
            label=f"{method_label} (final estimate)",
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Coefficient value")
    axis.legend(fontsize=8)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", choices=list(GAMES), default=list(GAMES))
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--total-samples", type=int, default=TOTAL_SAMPLES)
    parser.add_argument("--evaluation-interval", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--random-starts", type=int, default=RANDOM_STARTS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for game_name in args.games:
        description, warmup_budget = GAMES[game_name]
        print(f"=== {game_name} ({description}) ===", flush=True)
        started = time.perf_counter()
        record = _run_game(
            game_name,
            warmup_budget=warmup_budget,
            seed_count=args.seed_count,
            total_samples=args.total_samples,
            evaluation_interval=args.evaluation_interval,
            random_starts=args.random_starts,
        )
        print(f"  total wall time: {time.perf_counter() - started:0.1f}s", flush=True)
        print(f"  true omega = {np.round(record['true_omega'], 4)}", flush=True)

        final_active = np.array(
            [
                o.summary["final_metrics"]["stackelberg_suboptimality"]
                for o in record["active_outputs"]
            ]
        )
        final_random = np.array(
            [
                o.summary["final_metrics"]["stackelberg_suboptimality"]
                for o in record["random_outputs"]
            ]
        )
        active_mean = final_active.mean()
        active_std = final_active.std()
        random_mean = final_random.mean()
        random_std = final_random.std()
        print(
            f"  [active, own warm-up]  final SubOpt = {active_mean:.5f} +/- {active_std:.5f}",
            flush=True,
        )
        print(
            f"  [random, no warm-up]   final SubOpt = {random_mean:.5f} +/- {random_std:.5f}",
            flush=True,
        )
        print(
            f"  ratio (random / active) = {random_mean / active_mean:.2f}x",
            flush=True,
        )

        comparison_path = OUTPUT_DIR / f"behavioral_comparison_{game_name}.png"
        coefficients_path = OUTPUT_DIR / f"behavioral_coefficients_{game_name}.png"
        _plot_comparison(
            record,
            f"{description}\nActive IRL (own warm-up) vs. random with no warm-up "
            f"({args.seed_count} seeds)",
            comparison_path,
        )
        _plot_coefficients(record, description, coefficients_path)
        print(f"  wrote {comparison_path}", flush=True)
        print(f"  wrote {coefficients_path}", flush=True)


if __name__ == "__main__":
    main()
