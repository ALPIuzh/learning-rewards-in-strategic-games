"""Shared visual language for the Lobeke paper figures.

The figures are designed at the final two-column paper width.  Keeping the
typography, colors, line hierarchy, and export path here prevents the paper's
illustrations from looking like unrelated exploratory notebooks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from mairl_exp.evaluation.mpl import matplotlib as mpl

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
