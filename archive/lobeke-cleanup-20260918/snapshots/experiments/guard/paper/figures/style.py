"""Shared visual language for the Lobeke paper figures.

The figures are designed at the final two-column paper width.  Keeping the
typography, colors, line hierarchy, and export path here prevents the paper's
illustrations from looking like unrelated exploratory notebooks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mairl_exp.evaluation.mpl import matplotlib as mpl
from mairl_exp.evaluation.mpl import plt

PAPER_WIDTH = 7.25

INK = "#17212B"
MUTED = "#667085"
LIGHT_MUTED = "#98A2B3"
GRID = "#D9DEE7"
PANEL = "#F7F8FA"

ACTIVE = "#1769AA"
ACTIVE_LIGHT = "#69A8D5"
EXPERT_MLE = "#D58B32"
LIAO = "#16856F"
FOLLOWER_OLS = "#C4493D"
FIXED_RANDOM = "#8A725D"
RANDOM_ROUND = "#8A6FB0"

RANGER = ACTIVE
POACHER = FOLLOWER_OLS
ELEPHANT = "#557A46"
TERRAIN = "#E9F0E3"


@contextmanager
def paper_style() -> Iterator[None]:
    """Apply publication-scale Matplotlib defaults within a local context."""

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.titleweight": "semibold",
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    ):
        yield


def clean_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    """Use a light frame and horizontal reading guides."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    if grid:
        axis.grid(axis="y")
        axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13, 1.08, label, transform=axis.transAxes, ha="left", va="top", fontsize=9.0, fontweight="bold", color=INK
    )


def save_paper_figure(figure: plt.Figure, output_path: Path, *, dpi: int = 300) -> None:
    """Save the requested raster plus a vector PDF beside it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    if output_path.suffix.lower() != ".pdf":
        figure.savefig(output_path.with_suffix(".pdf"))
