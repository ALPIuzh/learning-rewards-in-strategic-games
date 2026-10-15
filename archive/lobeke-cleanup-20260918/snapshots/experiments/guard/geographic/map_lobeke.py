"""Regenerate the Lobeke green security game from its upstream data and plot a map.

Tier 3 of the GUARD investigation (see ../README.md): the precomputed predefined_games
pkl files used for Tier 1 carry no spatial metadata, only abstract target indices, so
there is no way to draw an actual park map from them. This script instead runs GUARD's
own GreenSecurityGame generator against the pinned submodule's elephant-tracking CSVs
(GUARD/data/lobeke/*.csv, fully offline after checkout) to recover real lat/lon grid
positions and target-level animal-density covariates, then solves for the optimal
defender coverage against the exact behavioral omega and plots it against the poacher's
resulting attack distribution on the park grid.

Requires the "guard" optional dependency group (networkx, geopy, scikit-learn, pandas,
h5py) -- not part of the core mairl/mairl_exp install.
"""

# GUARD is a pinned source dependency rather than an installed Python package, so its
# import follows the explicit submodule path insertion below.
# ruff: noqa: I001

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mairl_exp.evaluation.mpl import plt

from mairl_exp.config import GUARD_ROOT as _GUARD_ROOT, PLOTS_ROOT

sys.path.insert(0, str(_GUARD_ROOT))

from security_game.green_security_game import GreenSecurityGame
from utils import target_utils as _guard_target_utils

from experiments.guard.application import (
    _behavioral_features,
    _scale_behavioral_features,
    coverage_indicator as _coverage_indicator,
)
from mairl.core.problem import FloatArray
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl.zero_sum import CompiledZeroSumGame, compile_zero_sum_game

# GUARD revision 58964e7 uses ``np.floor`` in this helper without importing NumPy.
# Supply the missing module name from our adapter rather than modifying the public
# submodule. Remove this shim once the pinned upstream revision contains that import.
_guard_target_utils.np = np

# Lobeke National Park bounding box, matching GUARD's own notebook example.
COORDINATE_RECTANGLE = [2.05522, 2.2837, 15.8790, 16.2038]
HOME_BASE = (2.17, 16.04)
OUTPUT_DIR = PLOTS_ROOT / "guard"


def _load_lobeke_data() -> pd.DataFrame:
    frames = []
    for path in sorted((_GUARD_ROOT / "data" / "lobeke").glob("*.csv")):
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"location-lat": "lat", "location-long": "long"})
        frame["animal_id"] = frame["individual-local-identifier"].fillna(frame["tag-local-identifier"])
        frames.append(frame[["animal_id", "lat", "long", "timestamp"]])
    return pd.concat(frames, ignore_index=True).dropna()


def build_game(*, num_rows: int = 5, num_columns: int = 5, num_timesteps: int = 10):
    data = _load_lobeke_data()
    gsg = GreenSecurityGame(data, COORDINATE_RECTANGLE, "density", num_rows=num_rows, num_columns=num_columns)
    gsg.generate(
        num_attackers=2,
        num_defenders=1,
        home_base_assignments=[[HOME_BASE]],
        num_timesteps=num_timesteps,
        schedule_form=True,
        generate_utility_matrix=True,
        simple=True,
    )
    return gsg


def build_lobeke_zero_sum_game(
    gsg, *, eta: float = 1.25, radius_multiplier: float = 1.5, behavioral_fit_tolerance: float = 1e-6
) -> tuple[CompiledZeroSumGame, FloatArray, FloatArray, list, FloatArray]:
    """Compile the Lobeke schedule-form game as a zero-sum game with exact behavioral features.

    Returns ``(game, true_omega, defender_mat, target_nodes, coverage)``. ``defender_mat``
    is GUARD's own raw payoff matrix, kept for evaluation; it matches
    ``game.leader_features @ true_omega`` up to the behavioral fit residual, which
    for Lobeke is exact to floating-point precision (verified fit error ~1e-16).
    ``coverage`` is the (n_leader, n_follower) fraction-covered indicator, reused
    by downstream-task generation to build structured (not i.i.d.-noise) rewards.
    """

    schedule = gsg.schedule_form_dict
    attacker_mat = np.asarray(schedule["attacker_utility_matrix"], dtype=np.float64)
    defender_mat = np.asarray(schedule["defender_utility_matrix"], dtype=np.float64)
    target_utils = np.asarray(schedule["target_utilities"], dtype=np.float64)
    target_nodes = schedule["targets"]
    coverage = _coverage_indicator(attacker_mat, target_utils)

    phi_raw, true_omega, fit_error = _behavioral_features(
        attacker_mat, target_utils, fit_tolerance=behavioral_fit_tolerance
    )
    print(f"behavioral feature fit error: {fit_error:.3e}")
    print(f"true omega (coverage, value, coverage*value) = {np.round(true_omega, 4)}")

    phi, problem_eta, radius = _scale_behavioral_features(
        phi_raw, true_omega, eta=eta, radius_multiplier=radius_multiplier
    )
    game = compile_zero_sum_game(-phi, phi, eta=problem_eta, radius=radius)
    return game, true_omega, defender_mat, target_nodes, coverage


def solve_and_plot(gsg, *, eta: float = 1.25) -> Path:
    game, true_omega, defender_mat, target_nodes, coverage = build_lobeke_zero_sum_game(gsg, eta=eta)
    problem = game.problem

    config = LeaderOptimizationConfig(PolicyOptimizationConfig(method="multistart", random_starts=16))
    solution = optimize_leader_policy(problem, true_omega, defender_mat, config, seed=0)
    coverage_per_target = coverage.T @ solution.policy
    attack_probability = solution.follower_policy

    print(f"defender value at optimum: {solution.value:.4f}")
    print(f"max coverage probability: {coverage_per_target.max():.3f}")
    print(f"max attack probability: {attack_probability.max():.3f}")

    return _plot(gsg, target_nodes, coverage_per_target, attack_probability)


def _grid_array(gsg, node_values: dict) -> np.ndarray:
    grid = np.zeros((gsg.num_rows, gsg.num_columns))
    for node, value in node_values.items():
        row, col = gsg.graph.nodes[node]["position"]
        grid[row, col] = value
    return grid


def _panel(figure, axis, grid: np.ndarray, *, extent: list[float], cmap: str, title: str, colorbar_label: str,
        annotate_threshold: float | None, annotate_format: str, show_home_base: bool) -> None:  # fmt: skip
    image = axis.imshow(
        grid, extent=extent, origin="upper", cmap=cmap, aspect="auto", vmin=0.0, vmax=max(float(grid.max()), 1e-9)
    )
    min_lon, max_lon, min_lat, max_lat = extent
    n_rows, n_cols = grid.shape
    lon_edges = np.linspace(min_lon, max_lon, n_cols + 1)
    lat_edges = np.linspace(min_lat, max_lat, n_rows + 1)
    axis.set_xticks(lon_edges, minor=True)
    axis.set_yticks(lat_edges, minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", length=0)
    axis.set_xticks([])
    axis.set_yticks([])

    if annotate_threshold is not None:
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        # imshow(origin="upper") draws grid row 0 at the top (max_lat), so row-to-latitude
        # centers must run north-to-south to match, not south-to-north like lat_edges.
        lat_centers = ((lat_edges[:-1] + lat_edges[1:]) / 2)[::-1]
        norm = image.norm
        for row in range(n_rows):
            for col in range(n_cols):
                value = grid[row, col]
                if value < annotate_threshold:
                    continue
                text_color = "white" if norm(value) > 0.55 else "0.15"
                axis.annotate(
                    format(value, annotate_format),
                    (lon_centers[col], lat_centers[row]),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=text_color,
                    zorder=6,
                )

    if show_home_base:
        axis.scatter(
            [HOME_BASE[1]], [HOME_BASE[0]], marker="^", s=160, color="1.0", edgecolors="0.1", linewidths=1.4, zorder=5
        )
    axis.set_title(title, fontsize=11)
    figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.82, pad=0.02)


def _plot(gsg, target_nodes, coverage_per_target, attack_probability) -> Path:
    min_lat, max_lat, min_lon, max_lon = COORDINATE_RECTANGLE
    extent = [min_lon, max_lon, min_lat, max_lat]

    density = _grid_array(gsg, {n: gsg.graph.nodes[n]["score"] for n in gsg.graph.nodes})
    coverage = _grid_array(gsg, dict(zip(target_nodes, coverage_per_target, strict=True)))
    attack = _grid_array(gsg, dict(zip(target_nodes, attack_probability, strict=True)))

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    _panel(
        figure,
        axes[0],
        density,
        extent=extent,
        cmap="Greens",
        title="Estimated elephant density",
        colorbar_label="Density score",
        annotate_threshold=None,
        annotate_format="",
        show_home_base=True,
    )
    _panel(
        figure,
        axes[1],
        coverage,
        extent=extent,
        cmap="Blues",
        title="Optimal ranger patrol coverage",
        colorbar_label="Coverage probability",
        annotate_threshold=0.01,
        annotate_format=".0%",
        show_home_base=False,
    )
    _panel(
        figure,
        axes[2],
        attack,
        extent=extent,
        cmap="Reds",
        title="Poacher attack probability at optimum",
        colorbar_label="Attack probability",
        annotate_threshold=0.05,
        annotate_format=".0%",
        show_home_base=False,
    )
    axes[0].scatter([], [], marker="^", s=160, color="1.0", edgecolors="0.1", label="Ranger home base")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    figure.suptitle(
        "Lobeke National Park (Cameroon): active Stackelberg IRL applied to a real anti-poaching game", fontsize=13
    )
    figure.text(
        0.5,
        0.925,
        "5x5 grid over the park; a single ranger patrol unit. Coverage concentrates where "
        "density is "
        "highest, suppressing attack probability there and pushing it toward lower-value cells.",
        ha="center",
        fontsize=9.5,
        color="0.25",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "lobeke_map.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    gsg = build_game()
    schedule = gsg.schedule_form_dict
    print(f"targets: {len(schedule['targets'])}, defender actions: {len(schedule['defender_actions'])}")
    output_path = solve_and_plot(gsg)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
