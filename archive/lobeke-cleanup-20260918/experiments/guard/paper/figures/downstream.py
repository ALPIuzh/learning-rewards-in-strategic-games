"""Downstream-task transfer: same omega and features, varying leader reward.

Reuses the estimator trajectories already collected by the main comparison run
(no resampling) and asks, for a handful of diverse synthetic leader-reward
matrices unrelated to the poaching payoff structure, how each method's
estimate would have performed had the leader deployed it against that reward
instead. One curve per method per task, Stackelberg sub-optimality vs. the same
follower-sample-count axis as the main figure -- not a bar chart, so the reader
sees *when* each method's transfer quality stabilizes, not just its endpoint.

Usage::

    python -m experiments.guard.paper.figures.downstream --run-dir <path> [--num-tasks 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_corrected_run_traces
from experiments.guard.paper.figures.data import (
    METHOD_STYLE,
    SeedTrace,
    draw_diverse_structured_tasks,
    infer_warmup_budget,
    median_interval,
)
from experiments.guard.paper.figures.style import GRID, PAPER_WIDTH, clean_axis, paper_style, save_paper_figure
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy_batch
from mairl_exp.evaluation.mpl import plt


def load_runtime(run_dir: Path):
    """Rebuild the exact application/problem the run used, from its own stored config."""

    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    return application.build(numerics)


def task_suboptimality_trace(problem: CompiledProblem, true_omega: FloatArray, leader_reward: FloatArray,
        leader_config: LeaderOptimizationConfig, traces: list[SeedTrace]) -> FloatArray:  # fmt: skip
    """Return an ``(n_seeds, n_checkpoints)`` Stackelberg sub-optimality array for one task."""

    checkpoints = traces[0].checkpoints
    if any(not np.array_equal(item.checkpoints, checkpoints) for item in traces):
        raise ValueError("inconsistent checkpoints across seeds")

    true_solution = optimize_leader_policy_batch(problem, true_omega[None, :], leader_reward, leader_config, (0,))[0]

    n_seeds, n_checkpoints = len(traces), checkpoints.size
    subopt = np.empty((n_seeds, n_checkpoints), dtype=np.float64)
    for seed_index, trace in enumerate(traces):
        solutions = optimize_leader_policy_batch(
            problem, trace.omega_history, leader_reward, leader_config, tuple(range(n_checkpoints))
        )
        for checkpoint_index, solution in enumerate(solutions):
            deployed_value, _ = leader_value_and_gradient(problem, true_omega, leader_reward, solution.policy)
            subopt[seed_index, checkpoint_index] = max(0.0, true_solution.value - deployed_value)
    return subopt


def render_task_lines(task_labels: list[str], task_traces: dict[str, dict[tuple[str, bool], FloatArray]],
        checkpoints_by_method: dict[tuple[str, bool], FloatArray], *, warmup_budget: int,
        lower_percentile: float, upper_percentile: float, band_alpha: float,
        dpi: int, output_path: Path, floor: float = 3e-5) -> None:  # fmt: skip
    """Render one sub-optimality panel per task.

    Downstream tasks (arbitrary leader rewards) almost always have a *pure*
    optimal leader strategy, unlike the in-distribution task's genuine mixture --
    so once an estimate's implied best action matches the task's true best
    action, sub-optimality is exactly zero, not just small. Those exact zeros
    are real, not noise, but they make the per-seed distribution a large point
    mass near zero plus a heavy tail, not the roughly-unimodal shape a
    mean+Student-t band assumes -- so this uses median + percentile band
    (:func:`median_interval`) instead, which isn't dominated by a handful of
    outlier seeds. ``floor`` clips display only, avoiding wasting most of the
    log-scale axis on exact-zero values (matching the smallest *meaningful*
    non-zero values seen in the in-distribution panel).
    """

    n_tasks = len(task_labels)
    with paper_style():
        figure, axes = plt.subplots(
            1, n_tasks, figsize=(PAPER_WIDTH, 2.55), squeeze=False, sharey=True, gridspec_kw={"wspace": 0.16}
        )
        axes = axes[0]

        for task_index, (axis, label) in enumerate(zip(axes, task_labels, strict=True)):
            for key, style in METHOD_STYLE.items():
                per_checkpoint = task_traces[label].get(key)
                if per_checkpoint is None:
                    continue
                checkpoints = checkpoints_by_method[key]
                median, lower, upper = median_interval(
                    per_checkpoint, lower_percentile=lower_percentile, upper_percentile=upper_percentile
                )
                axis.plot(
                    checkpoints,
                    np.maximum(median, floor),
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                )
                axis.fill_between(
                    checkpoints,
                    np.maximum(lower, floor),
                    np.maximum(upper, floor),
                    color=style["color"],
                    alpha=band_alpha * 0.7 * float(style["alpha"]),
                    linewidth=0,
                    zorder=int(style["zorder"]) - 1,
                )
            axis.axvspan(0, warmup_budget, color="#DDEAF4", alpha=0.45, zorder=0)
            axis.axvline(warmup_budget, color=GRID, linewidth=0.7, zorder=1)
            title = "Original objective" if task_index == 0 else f"New priority {task_index}"
            axis.set(xlabel="Poacher samples", yscale="log", title=title)
            axis.set_ylim(bottom=floor)
            clean_axis(axis)

        axes[0].set_ylabel("Deployment regret (lower is better)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.02),
            columnspacing=1.1,
            handlelength=2.3,
        )
        figure.subplots_adjust(left=0.09, right=0.995, top=0.95, bottom=0.29)
        save_paper_figure(figure, output_path, dpi=dpi)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--task-seed", type=int, default=2000)
    parser.add_argument("--task-scale", type=float, default=1.0)
    parser.add_argument("--task-candidate-pool", type=int, default=200)
    parser.add_argument("--warmup-budget", type=int, default=None)
    parser.add_argument("--lower-percentile", type=float, default=10.0)
    parser.add_argument("--upper-percentile", type=float, default=90.0)
    parser.add_argument("--band-alpha", type=float, default=0.12)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    traces = load_corrected_run_traces(args.run_dir, args.correction_dir)
    runtime = load_runtime(args.run_dir)
    leader_config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=2)
    )

    warmup_budget = infer_warmup_budget(traces) if args.warmup_budget is None else args.warmup_budget

    coverage_indicator = runtime.metadata.get("coverage_indicator")
    if coverage_indicator is None:
        raise ValueError("coverage_indicator missing from application metadata")
    tasks, pairwise = draw_diverse_structured_tasks(
        np.asarray(coverage_indicator, dtype=np.float64),
        num_tasks=args.num_tasks,
        seed=args.task_seed,
        scale=args.task_scale,
        candidate_pool=args.task_candidate_pool,
    )
    print(f"drew {args.num_tasks} downstream tasks; pairwise cosine similarity:\n{pairwise}")

    task_labels = ["In-distribution"] + [f"Downstream task {i + 1}" for i in range(args.num_tasks)]
    in_distribution_reward = runtime.metadata.get("defender_reward")
    if in_distribution_reward is None:
        raise ValueError("in-distribution defender reward missing from application metadata")
    task_rewards = [np.asarray(in_distribution_reward, dtype=np.float64)] + [tasks[i] for i in range(args.num_tasks)]

    checkpoints_by_method = {key: item[0].checkpoints for key, item in traces.items()}
    task_traces: dict[str, dict[tuple[str, bool], FloatArray]] = {}

    for label, reward_array in zip(task_labels, task_rewards, strict=True):
        per_method: dict[tuple[str, bool], FloatArray] = {}
        for key, seed_traces in traces.items():
            per_method[key] = task_suboptimality_trace(
                runtime.problem, runtime.true_omega, reward_array, leader_config, seed_traces
            )
        task_traces[label] = per_method
        print(f"{label}: computed {len(per_method)} method curves")

    output_path = args.output or (args.run_dir / "lobeke_paper_downstream_tasks.png")
    render_task_lines(
        task_labels,
        task_traces,
        checkpoints_by_method,
        warmup_budget=warmup_budget,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        band_alpha=args.band_alpha,
        dpi=args.dpi,
        output_path=output_path,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
