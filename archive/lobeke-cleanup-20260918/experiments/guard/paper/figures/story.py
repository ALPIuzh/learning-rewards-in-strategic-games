"""Paper-facing graphical abstract for the Lobéké application.

The figure explains the scientific use case before presenting performance
curves: real ecological targets, repeated ranger--poacher interactions, a
learned behavioral model, and transfer to new ranger objectives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.guard.geographic.map_lobeke import (
    COORDINATE_RECTANGLE,
    HOME_BASE,
    _grid_array,
    build_game,
    build_lobeke_zero_sum_game,
)
from experiments.guard.paper.figures.data import draw_diverse_structured_tasks
from experiments.guard.paper.figures.style import (
    ELEPHANT,
    FOLLOWER_OLS,
    GRID,
    INK,
    MUTED,
    PAPER_WIDTH,
    RANGER,
    paper_style,
    save_paper_figure,
)
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.config import RESULTS_DIR
from mairl_exp.evaluation.mpl import Line2D, LinearSegmentedColormap, plt

_TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "lobeke_terrain", ["#F7F9F4", "#E7EFDF", "#C9DDB8", "#91B973", "#5F8F48"]
)
_COVERAGE_CMAP = LinearSegmentedColormap.from_list("coverage", ["#F7F9FB", "#C9DDED", "#70A9D1", RANGER])


def _node_position(gsg, node) -> tuple[float, float]:
    """Return a target position normalized to the displayed park rectangle."""

    row, col = gsg.graph.nodes[node]["position"]
    x = (col + 0.5) / gsg.num_columns
    y = 1.0 - (row + 0.5) / gsg.num_rows
    return x, y


def _home_base_position() -> tuple[float, float]:
    min_lat, max_lat, min_lon, max_lon = COORDINATE_RECTANGLE
    return ((HOME_BASE[1] - min_lon) / (max_lon - min_lon), (HOME_BASE[0] - min_lat) / (max_lat - min_lat))


def _park_axis(axis: plt.Axes, density_grid: np.ndarray) -> None:
    axis.imshow(
        density_grid,
        extent=(0, 1, 0, 1),
        origin="upper",
        cmap=_TERRAIN_CMAP,
        interpolation="bicubic",
        aspect="equal",
        vmin=0.0,
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_color("#DCE5D6")
        spine.set_linewidth(0.7)


def _panel_heading(axis: plt.Axes, number: str, title: str, subtitle: str) -> None:
    axis.text(
        0.0,
        1.17,
        number,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "circle,pad=0.28", "facecolor": INK, "edgecolor": "none"},
    )
    axis.text(
        0.13,
        1.205,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        fontweight="semibold",
        color=INK,
    )
    axis.text(0.13, 1.09, subtitle, transform=axis.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED)


def _coverage_grid(gsg, target_nodes, coverage_per_target: np.ndarray) -> np.ndarray:
    return _grid_array(gsg, dict(zip(target_nodes, coverage_per_target, strict=True)))


def render(*, eta: float = 1.25, output_path: Path | None = None) -> Path:
    gsg = build_game()
    game, true_omega, defender_mat, target_nodes, coverage = build_lobeke_zero_sum_game(gsg, eta=eta)
    problem = game.problem
    target_positions = np.asarray([_node_position(gsg, node) for node in target_nodes])
    target_density = np.asarray([gsg.graph.nodes[node]["score"] for node in target_nodes])
    density_grid = _grid_array(gsg, {node: gsg.graph.nodes[node]["score"] for node in gsg.graph.nodes})

    leader_config = LeaderOptimizationConfig(PolicyOptimizationConfig(method="multistart", random_starts=16))
    real_solution = optimize_leader_policy(problem, true_omega, defender_mat, leader_config, seed=0)
    real_coverage = coverage.T @ real_solution.policy
    attack_probability = real_solution.follower_policy

    downstream_rewards, _ = draw_diverse_structured_tasks(
        coverage, num_tasks=2, seed=2000, scale=1.0, candidate_pool=200
    )
    transfer_rewards = (defender_mat, downstream_rewards[0], downstream_rewards[1])
    transfer_coverages = [
        coverage.T @ optimize_leader_policy(problem, true_omega, reward, leader_config, seed=index).policy
        for index, reward in enumerate(transfer_rewards)
    ]

    with paper_style():
        figure = plt.figure(figsize=(PAPER_WIDTH, 2.55))
        grid = figure.add_gridspec(
            1, 4, width_ratios=(1.05, 1.05, 1.05, 1.35), left=0.035, right=0.99, top=0.72, bottom=0.12, wspace=0.40
        )
        environment_axis = figure.add_subplot(grid[0, 0])
        interaction_axis = figure.add_subplot(grid[0, 1])
        model_axis = figure.add_subplot(grid[0, 2])
        transfer_axis = figure.add_subplot(grid[0, 3])

        # 1. Real ecological signal.
        _park_axis(environment_axis, density_grid)
        relative_density = target_density / max(float(target_density.max()), 1e-12)
        environment_axis.scatter(
            target_positions[:, 0],
            target_positions[:, 1],
            s=12 + 120 * relative_density**0.9,
            color=ELEPHANT,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
        base_x, base_y = _home_base_position()
        environment_axis.scatter([base_x], [base_y], marker="^", s=44, color=INK, edgecolor="white", linewidth=0.7)
        _panel_heading(environment_axis, "1", "A real conservation game", "23 patrol routes · 23 targets")

        # 2. One representative ranger--poacher interaction under the true model.
        _park_axis(interaction_axis, density_grid)
        interaction_axis.imshow(
            _coverage_grid(gsg, target_nodes, real_coverage),
            extent=(0, 1, 0, 1),
            origin="upper",
            cmap=_COVERAGE_CMAP,
            vmin=0.0,
            vmax=max(0.75, float(real_coverage.max())),
            alpha=0.82,
            interpolation="nearest",
        )
        relative_attack = attack_probability / max(float(attack_probability.max()), 1e-12)
        interaction_axis.scatter(
            target_positions[:, 0],
            target_positions[:, 1],
            s=8 + 55 * relative_attack**1.4,
            color=FOLLOWER_OLS,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        interaction_axis.legend(
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
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            handletextpad=0.35,
            columnspacing=0.8,
        )
        _panel_heading(interaction_axis, "2", "Observe responses", r"patrol $\mu_t$, then attack $b_t$")

        # 3. Human-readable behavioral model.
        model_axis.set_facecolor("#F7F8FA")
        coefficients = np.asarray(true_omega).copy()
        coefficients[np.abs(coefficients) < 5e-3] = 0.0
        names = ("coverage", "value", r"coverage $\cdot$ value")
        colors = (MUTED, ELEPHANT, FOLLOWER_OLS)
        y = np.arange(3)[::-1]
        model_axis.axvline(0.0, color=GRID, linewidth=0.8)
        model_axis.barh(y, coefficients, color=colors, height=0.34)
        for position, coefficient in zip(y, coefficients, strict=True):
            side = "left" if coefficient < 0 else "right"
            offset = -0.05 if coefficient < 0 else 0.05
            model_axis.text(
                coefficient + offset,
                position,
                f"{coefficient:+.2f}",
                ha=side,
                va="center",
                fontsize=6.5,
                fontweight="semibold",
                color=INK,
            )
        model_axis.set(xlim=(-0.9, 1.15), ylim=(-0.65, 2.65), yticks=y, yticklabels=names, xticks=(-0.5, 0.0, 0.5, 1.0))
        model_axis.tick_params(axis="y", length=0, pad=2)
        model_axis.grid(axis="x", color=GRID, linewidth=0.6)
        for spine in model_axis.spines.values():
            spine.set_visible(False)
        model_axis.text(
            0.5,
            -0.22,
            "deters          effect on attack utility          attracts",
            transform=model_axis.transAxes,
            ha="center",
            va="top",
            fontsize=5.9,
            color=MUTED,
        )
        _panel_heading(model_axis, "3", "Learn behavior", r"estimate $\hat\omega$ from field data")

        # 4. The same omega reused under three leader rewards.
        transfer_axis.set_axis_off()
        _panel_heading(transfer_axis, "4", "Redeploy", r"new ranger reward, same $\hat\omega$")
        mini_labels = ("anti-poaching", "priority A", "priority B")
        for index, (label, patrol) in enumerate(zip(mini_labels, transfer_coverages, strict=True)):
            inset = transfer_axis.inset_axes((0.01 + 0.335 * index, 0.08, 0.29, 0.74))
            inset.imshow(
                _coverage_grid(gsg, target_nodes, patrol),
                origin="upper",
                cmap=_COVERAGE_CMAP,
                vmin=0.0,
                vmax=0.85,
                interpolation="nearest",
            )
            inset.set(xticks=[], yticks=[])
            inset.set_title(label, fontsize=5.8, pad=2, color=INK)
            for spine in inset.spines.values():
                spine.set_color("white")
                spine.set_linewidth(0.8)
        transfer_axis.text(
            0.5,
            -0.03,
            "one model; different optimal patrols",
            transform=transfer_axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.2,
            color=MUTED,
        )

        figure.suptitle(
            "Learn poacher behavior once, reuse it as ranger priorities change",
            x=0.035,
            y=0.98,
            ha="left",
            fontsize=11.0,
            fontweight="semibold",
            color=INK,
        )
        figure.text(
            0.035,
            0.885,
            "Active IRL chooses informative patrols, observes attacks, and learns a compact "
            "response model that remains useful beyond the original objective.",
            ha="left",
            va="top",
            fontsize=7.1,
            color=MUTED,
        )

        output_path = output_path or (RESULTS_DIR / "lobeke_paper_story.png")
        save_paper_figure(figure, output_path)
        plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_path = render(output_path=args.output)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
