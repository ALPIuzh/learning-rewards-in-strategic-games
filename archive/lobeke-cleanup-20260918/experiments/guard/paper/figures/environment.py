"""Show the Lobeke park map under several leader rewards, same omega and features.

Extends ``geographic/map_lobeke.py``'s single-map figure into a small-multiples
grid: one column per task (the real anti-poaching reward, plus a few diverse
synthetic downstream rewards), two rows (ranger coverage / poacher attack
probability), all solved against the *same* true omega. Meant to make the
downstream-task premise -- same ecological reality and follower behavior
model, only the leader's objective changes -- concrete rather than abstract.

Usage::

    python -m experiments.guard.paper.figures.environment [--num-tasks 3] [--output ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.guard.geographic.map_lobeke import _grid_array, build_game, build_lobeke_zero_sum_game
from experiments.guard.paper.figures.data import draw_diverse_structured_tasks
from experiments.guard.paper.figures.style import INK, PAPER_WIDTH, paper_style, save_paper_figure
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.config import RESULTS_DIR
from mairl_exp.evaluation.mpl import plt


def render_task_grid(*, num_tasks: int = 3, task_seed: int = 2000, task_scale: float = 1.0,
        task_candidate_pool: int = 200, eta: float = 1.25, output_path: Path | None = None) -> Path:  # fmt: skip
    gsg = build_game()
    game, true_omega, defender_mat, target_nodes, coverage_indicator = build_lobeke_zero_sum_game(gsg, eta=eta)
    problem = game.problem

    tasks, _pairwise = draw_diverse_structured_tasks(
        coverage_indicator, num_tasks=num_tasks, seed=task_seed, scale=task_scale, candidate_pool=task_candidate_pool
    )
    labels = ["Real anti-poaching reward"] + [f"Downstream task {i + 1}" for i in range(num_tasks)]
    rewards = [defender_mat] + [tasks[i] for i in range(num_tasks)]

    leader_config = LeaderOptimizationConfig(PolicyOptimizationConfig(method="multistart", random_starts=16))

    coverage_grids = []
    attack_grids = []
    for label, reward in zip(labels, rewards, strict=True):
        solution = optimize_leader_policy(problem, true_omega, reward, leader_config, seed=0)
        coverage_per_target = coverage_indicator.T @ solution.policy
        attack_probability = solution.follower_policy
        coverage_grids.append(_grid_array(gsg, dict(zip(target_nodes, coverage_per_target, strict=True))))
        attack_grids.append(_grid_array(gsg, dict(zip(target_nodes, attack_probability, strict=True))))
        print(
            f"{label}: defender value={solution.value:.4f}, "
            f"max coverage={coverage_per_target.max():.3f}, "
            f"max attack prob={attack_probability.max():.3f}"
        )

    with paper_style():
        figure, axes = plt.subplots(
            2, len(labels), figsize=(PAPER_WIDTH, 3.65), gridspec_kw={"wspace": 0.08, "hspace": 0.10}
        )
        coverage_vmax = max(float(grid.max()) for grid in coverage_grids)
        attack_vmax = max(float(grid.max()) for grid in attack_grids)
        images = []
        for column, _label in enumerate(labels):
            for row, (grid, cmap, vmax) in enumerate(
                ((coverage_grids[column], "Blues", coverage_vmax), (attack_grids[column], "Reds", attack_vmax))
            ):
                axis = axes[row, column]
                image = axis.imshow(grid, origin="upper", cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
                images.append(image)
                axis.set(xticks=[], yticks=[])
                for boundary in np.arange(-0.5, 5.5, 1.0):
                    axis.axhline(boundary, color="white", linewidth=0.6)
                    axis.axvline(boundary, color="white", linewidth=0.6)
                threshold = 0.08 if row == 0 else 0.05
                for grid_row, grid_col in np.argwhere(grid >= threshold):
                    value = grid[grid_row, grid_col]
                    axis.text(
                        grid_col,
                        grid_row,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        fontweight="semibold",
                        color="white" if value > 0.45 * vmax else INK,
                    )
                for spine in axis.spines.values():
                    spine.set_visible(False)
            axes[0, column].set_title(
                "Current\nanti-poaching" if column == 0 else f"New\npriority {column}", fontsize=7.6, pad=4, color=INK
            )

        axes[0, 0].text(
            -0.16,
            0.5,
            "Ranger\npatrol",
            transform=axes[0, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="semibold",
            color="#1769AA",
        )
        axes[1, 0].text(
            -0.16,
            0.5,
            "Poacher\nresponse",
            transform=axes[1, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="semibold",
            color="#C4493D",
        )
        coverage_bar = figure.colorbar(images[0], ax=axes[0, :], fraction=0.018, pad=0.012, aspect=28)
        coverage_bar.set_label("probability", fontsize=6.5)
        attack_bar = figure.colorbar(images[1], ax=axes[1, :], fraction=0.018, pad=0.012, aspect=28)
        attack_bar.set_label("probability", fontsize=6.5)
        figure.subplots_adjust(left=0.14, right=0.90, top=0.92, bottom=0.03)
        output_path = output_path or (RESULTS_DIR / "lobeke_paper_environment_maps.png")
        save_paper_figure(figure, output_path)
        plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--task-seed", type=int, default=2000)
    parser.add_argument("--task-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = render_task_grid(
        num_tasks=args.num_tasks, task_seed=args.task_seed, task_scale=args.task_scale, output_path=args.output
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
