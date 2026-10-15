"""Concrete main-text illustration of transfer across changing ranger priorities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.guard.geographic.map_lobeke import _grid_array, build_game
from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_corrected_run_traces
from experiments.guard.paper.figures.data import draw_diverse_structured_tasks
from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.story import _home_base_position
from experiments.guard.paper.figures.style import (
    ACTIVE,
    FOLLOWER_OLS,
    GRID,
    INK,
    MUTED,
    PAPER_WIDTH,
    paper_style,
    save_paper_figure,
)
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy
from mairl_exp.config import RESULTS_DIR
from mairl_exp.evaluation.mpl import LinearSegmentedColormap, path_effects, plt

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

_ROW_STYLE = {
    "oracle": {"label": "Oracle", "detail": "true behavior", "color": "#344054"},
    "active": {"label": "Active IRL", "detail": "median learned model", "color": ACTIVE},
    "follower_ols": {"label": "Follower-only OLS", "detail": "median learned model", "color": FOLLOWER_OLS},
}


def _median_final_estimate(run_dir: Path, method: str, warmup_enabled: bool,
        correction_dir: Path) -> tuple[int, np.ndarray]:  # fmt: skip
    traces = load_corrected_run_traces(run_dir, correction_dir)[(method, warmup_enabled)]
    finals = np.asarray([trace.metrics["coefficient_l2_error"][-1] for trace in traces])
    median_index = int(np.argsort(finals)[len(finals) // 2])
    trace = traces[median_index]
    return trace.seed, trace.omega_history[-1]


def _coverage_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("row_coverage", ["#FAFBFC", "#D9E4EC", color])


def _patrol_panel(axis: plt.Axes, coverage_grid: np.ndarray, *, color: str, home_base_xy: tuple[float, float]) -> None:
    axis.imshow(
        coverage_grid,
        origin="upper",
        extent=(0, 1, 0, 1),
        cmap=_coverage_cmap(color),
        vmin=0.0,
        vmax=0.85,
        interpolation="nearest",
    )
    axis.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
    for boundary in np.linspace(0, 1, 6):
        axis.axhline(boundary, color="white", linewidth=0.55)
        axis.axvline(boundary, color="white", linewidth=0.55)
    axis.scatter(
        [home_base_xy[0]], [home_base_xy[1]], marker="^", s=18, color=INK, edgecolor="white", linewidth=0.4, zorder=4
    )
    for row, col in np.argwhere(coverage_grid >= 0.08):
        value = coverage_grid[row, col]
        axis.text(
            (col + 0.5) / 5,
            1.0 - (row + 0.5) / 5,
            f"{value:.0%}",
            ha="center",
            va="center",
            fontsize=4.7,
            fontweight="semibold",
            color="white" if value > 0.28 else INK,
            path_effects=([path_effects.withStroke(linewidth=0.7, foreground=color)] if value > 0.28 else []),
        )
    for spine in axis.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.6)


def render(run_dir: Path, *, num_days: int, task_seed: int, task_scale: float,
        output_path: Path | None, correction_dir: Path = DEFAULT_CORRECTION_DIR) -> Path:  # fmt: skip
    gsg = build_game()
    runtime = load_runtime(run_dir)
    problem = runtime.problem
    true_omega = runtime.true_omega
    coverage_indicator = np.asarray(runtime.metadata["coverage_indicator"], dtype=np.float64)
    target_nodes = gsg.schedule_form_dict["targets"]

    active_seed, active_omega = _median_final_estimate(run_dir, "active", True, correction_dir)
    follower_seed, follower_omega = _median_final_estimate(run_dir, "follower_ols", False, correction_dir)
    print(f"active representative seed={active_seed}: {np.round(active_omega, 4)}")
    print(f"follower-OLS representative seed={follower_seed}: {np.round(follower_omega, 4)}")

    omegas = {"oracle": true_omega, "active": active_omega, "follower_ols": follower_omega}
    tasks, _ = draw_diverse_structured_tasks(
        coverage_indicator, num_tasks=num_days, seed=task_seed, scale=task_scale, candidate_pool=300
    )
    leader_config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=2)
    )

    rows = ("oracle", "active", "follower_ols")
    coverages: dict[tuple[str, int], np.ndarray] = {}
    regrets: dict[str, list[float]] = {row: [] for row in rows}
    for day_index, reward in enumerate(tasks):
        oracle_solution = optimize_leader_policy(problem, true_omega, reward, leader_config, seed=day_index)
        for row in rows:
            solution = (
                oracle_solution
                if row == "oracle"
                else optimize_leader_policy(problem, omegas[row], reward, leader_config, seed=day_index)
            )
            patrol = coverage_indicator.T @ solution.policy
            coverages[(row, day_index)] = _grid_array(gsg, dict(zip(target_nodes, patrol, strict=True)))
            deployed_value, _ = leader_value_and_gradient(problem, true_omega, reward, solution.policy)
            regrets[row].append(max(0.0, oracle_solution.value - deployed_value))

    with paper_style():
        figure, axes = plt.subplots(
            len(rows), num_days, figsize=(PAPER_WIDTH, 3.45), gridspec_kw={"wspace": 0.06, "hspace": 0.10}
        )
        home_base_xy = _home_base_position()
        for row_index, row in enumerate(rows):
            style = _ROW_STYLE[row]
            for day_index in range(num_days):
                axis = axes[row_index, day_index]
                _patrol_panel(axis, coverages[(row, day_index)], color=style["color"], home_base_xy=home_base_xy)
                if row_index == 0:
                    day_name = DAY_NAMES[day_index] if day_index < len(DAY_NAMES) else f"Day {day_index + 1}"
                    axis.set_title(day_name, fontsize=7.5, pad=4, color=INK)
            axes[row_index, 0].text(
                -0.13,
                0.58,
                style["label"],
                transform=axes[row_index, 0].transAxes,
                ha="right",
                va="center",
                fontsize=7.5,
                fontweight="semibold",
                color=style["color"],
            )
            axes[row_index, 0].text(
                -0.13,
                0.37,
                style["detail"],
                transform=axes[row_index, 0].transAxes,
                ha="right",
                va="center",
                fontsize=5.9,
                color=MUTED,
            )
            mean_regret = float(np.mean(regrets[row]))
            axes[row_index, -1].text(
                1.08,
                0.52,
                "target" if row == "oracle" else f"mean regret\n{mean_regret:.1e}",
                transform=axes[row_index, -1].transAxes,
                ha="left",
                va="center",
                fontsize=6.1,
                color=style["color"],
            )

        figure.subplots_adjust(left=0.20, right=0.91, top=0.92, bottom=0.02)
        output_path = output_path or (RESULTS_DIR / "lobeke_paper_temporal_evolution.png")
        save_paper_figure(figure, output_path)
        plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-days", type=int, default=5)
    parser.add_argument("--task-seed", type=int, default=2000)
    parser.add_argument("--task-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    args = parser.parse_args()
    output_path = render(
        args.run_dir,
        num_days=args.num_days,
        task_seed=args.task_seed,
        task_scale=args.task_scale,
        output_path=args.output,
        correction_dir=args.correction_dir,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
