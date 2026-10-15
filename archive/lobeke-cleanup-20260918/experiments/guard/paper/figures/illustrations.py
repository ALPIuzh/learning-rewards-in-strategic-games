"""Export independent, title-free illustration assets for LaTeX composition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.guard.geographic.map_lobeke import _grid_array, build_game, build_lobeke_zero_sum_game
from experiments.guard.paper.figures.data import draw_diverse_structured_tasks
from experiments.guard.paper.figures.story import (
    _COVERAGE_CMAP,
    _coverage_grid,
    _home_base_position,
    _node_position,
    _park_axis,
)
from experiments.guard.paper.figures.style import (
    ELEPHANT,
    FOLLOWER_OLS,
    GRID,
    INK,
    MUTED,
    RANGER,
    paper_style,
    save_paper_figure,
)
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.config import RESULTS_DIR
from mairl_exp.evaluation.mpl import Line2D, plt


def _environment_asset(gsg, target_nodes, target_density, density_grid, output_dir: Path) -> None:
    with paper_style():
        figure, axis = plt.subplots(figsize=(2.15, 2.15))
        _park_axis(axis, density_grid)
        positions = np.asarray([_node_position(gsg, node) for node in target_nodes])
        relative = target_density / max(float(target_density.max()), 1e-12)
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=10 + 105 * relative**0.9,
            color=ELEPHANT,
            alpha=0.84,
            edgecolor="white",
            linewidth=0.5,
        )
        base_x, base_y = _home_base_position()
        axis.scatter([base_x], [base_y], marker="^", s=42, color=INK, edgecolor="white", linewidth=0.7)
        figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        save_paper_figure(figure, output_dir / "environment.png")
        plt.close(figure)


def _interaction_asset(gsg, target_nodes, density_grid, coverage_per_target,
        attack_probability, output_dir: Path) -> None:  # fmt: skip
    with paper_style():
        figure, axis = plt.subplots(figsize=(2.15, 2.35))
        _park_axis(axis, density_grid)
        axis.imshow(
            _coverage_grid(gsg, target_nodes, coverage_per_target),
            extent=(0, 1, 0, 1),
            origin="upper",
            cmap=_COVERAGE_CMAP,
            vmin=0.0,
            vmax=max(0.75, float(coverage_per_target.max())),
            alpha=0.84,
            interpolation="nearest",
        )
        positions = np.asarray([_node_position(gsg, node) for node in target_nodes])
        relative = attack_probability / max(float(attack_probability.max()), 1e-12)
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=7 + 52 * relative**1.4,
            color=FOLLOWER_OLS,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        axis.legend(
            handles=(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="none",
                    markerfacecolor=RANGER,
                    markeredgecolor="none",
                    markersize=5,
                    label="patrol",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=FOLLOWER_OLS,
                    markeredgecolor="white",
                    markersize=5,
                    label="attack",
                ),
            ),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            handletextpad=0.35,
            columnspacing=0.8,
        )
        figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.13)
        save_paper_figure(figure, output_dir / "interaction.png")
        plt.close(figure)


def _model_asset(true_omega: np.ndarray, output_dir: Path) -> None:
    coefficients = np.asarray(true_omega).copy()
    coefficients[np.abs(coefficients) < 5e-3] = 0.0
    names = ("coverage", "value", r"coverage $\cdot$ value")
    colors = (MUTED, ELEPHANT, FOLLOWER_OLS)
    y = np.arange(3)[::-1]
    with paper_style():
        figure, axis = plt.subplots(figsize=(2.65, 1.75))
        axis.axvline(0.0, color=GRID, linewidth=0.8)
        axis.barh(y, coefficients, color=colors, height=0.34)
        for position, coefficient in zip(y, coefficients, strict=True):
            axis.text(
                coefficient + (-0.04 if coefficient < 0 else 0.04),
                position,
                f"{coefficient:+.2f}",
                ha="left" if coefficient < 0 else "right",
                va="center",
                fontsize=7.0,
                fontweight="semibold",
                color=INK,
            )
        axis.set(
            xlim=(-0.9, 1.15),
            ylim=(-0.6, 2.6),
            yticks=y,
            yticklabels=names,
            xticks=(-0.5, 0.0, 0.5, 1.0),
            xlabel="effect on attack utility",
        )
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.subplots_adjust(left=0.34, right=0.98, top=0.98, bottom=0.20)
        save_paper_figure(figure, output_dir / "behavior-model.png")
        plt.close(figure)


def _transfer_asset(gsg, target_nodes, coverages: list[np.ndarray], output_dir: Path) -> None:
    with paper_style():
        figure, axes = plt.subplots(1, len(coverages), figsize=(3.55, 1.25), gridspec_kw={"wspace": 0.08})
        for axis, patrol in zip(axes, coverages, strict=True):
            axis.imshow(
                _coverage_grid(gsg, target_nodes, patrol),
                origin="upper",
                cmap=_COVERAGE_CMAP,
                vmin=0.0,
                vmax=0.85,
                interpolation="nearest",
            )
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_color("white")
                spine.set_linewidth(0.7)
        figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        save_paper_figure(figure, output_dir / "transfer-patrols.png")
        plt.close(figure)


def render_assets(output_dir: Path, *, eta: float = 1.25) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gsg = build_game()
    game, true_omega, defender_mat, target_nodes, coverage = build_lobeke_zero_sum_game(gsg, eta=eta)
    problem = game.problem
    density_grid = _grid_array(gsg, {node: gsg.graph.nodes[node]["score"] for node in gsg.graph.nodes})
    target_density = np.asarray([gsg.graph.nodes[node]["score"] for node in target_nodes])
    leader_config = LeaderOptimizationConfig(PolicyOptimizationConfig(method="multistart", random_starts=16))
    real_solution = optimize_leader_policy(problem, true_omega, defender_mat, leader_config, seed=0)
    real_coverage = coverage.T @ real_solution.policy
    downstream, _ = draw_diverse_structured_tasks(coverage, num_tasks=2, seed=2000, scale=1.0, candidate_pool=200)
    rewards = (defender_mat, downstream[0], downstream[1])
    transfer_coverages = [
        coverage.T @ optimize_leader_policy(problem, true_omega, reward, leader_config, seed=index).policy
        for index, reward in enumerate(rewards)
    ]

    _environment_asset(gsg, target_nodes, target_density, density_grid, output_dir)
    _interaction_asset(gsg, target_nodes, density_grid, real_coverage, real_solution.follower_policy, output_dir)
    _model_asset(true_omega, output_dir)
    _transfer_asset(gsg, target_nodes, transfer_coverages, output_dir)
    return tuple(sorted(output_dir.glob("*.pdf")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "illustrations")
    args = parser.parse_args()
    for path in render_assets(args.output_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
