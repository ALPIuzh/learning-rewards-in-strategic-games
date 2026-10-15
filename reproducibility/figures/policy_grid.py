"""Shared TikZ component for every Lobeke downstream-policy grid.

Main-text and appendix grids must use this component. Keeping dimensions and
placement here makes map size equality a code invariant rather than a visual
convention.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import numpy as np

from mairl_exp.config import FIGURE_STYLE as _STYLE_SHEET

_SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
_D65 = np.array([0.95047, 1.0, 1.08883])


@cache
def _semantic_colors() -> Mapping[str, tuple[float, float, float]]:
    """Read the paper's semantic colours so they cannot drift from the style sheet."""

    definitions = re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}", _STYLE_SHEET.read_text())
    return {name: tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)) for name, value in definitions}


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (_SRGB_TO_XYZ @ linear) / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[0] + 16) / 116
    f = np.array([fy + lab[1] / 500, fy, fy - lab[2] / 200])
    xyz = np.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27)) * _D65
    linear = np.linalg.inv(_SRGB_TO_XYZ) @ xyz
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def _tint(color: str, fraction: float) -> str:
    """Blend white towards ``color`` in CIELAB rather than in RGB.

    Mixing in RGB spaces the intermediate tones unevenly, because perceived
    lightness is roughly a cube root of luminance. Interpolating L* instead
    makes equal steps in probability read as equal steps in shade. Both
    endpoints are unchanged, so this alters no value's meaning.
    """

    rgb = _semantic_colors().get(color)
    if rgb is None:
        raise ValueError(f"unknown semantic colour: {color}")
    white = _to_lab(np.ones(3))
    mixed = _to_rgb((1 - fraction) * white + fraction * _to_lab(np.asarray(rgb, dtype=np.float64)))
    red, green, blue = (f"{channel:.4f}" for channel in mixed)
    return f"{{rgb,1:red,{red};green,{green};blue,{blue}}}"


@dataclass(frozen=True, slots=True)
class PolicyGridLayout:
    """Fixed geometry shared by all policy-grid figures."""

    cell_cm: float = 0.38
    first_row_bottom_cm: float = -1.95
    row_step_cm: float = 2.15
    label_x_cm: float = 12.00
    separator_x_cm: float = 3.10
    column_centers_cm: tuple[float, ...] = (1.55, 4.75, 7.65, 10.55)

    @property
    def map_width_cm(self) -> float:
        return 5 * self.cell_cm

    def row_bottom_cm(self, index: int) -> float:
        return self.first_row_bottom_cm - self.row_step_cm * index


@dataclass(frozen=True, slots=True)
class PolicyGridRow:
    """One protocol row in a downstream-policy grid."""

    label: str
    slug: str
    color: str
    scores: tuple[str, ...] | None = None


# The darkest cell across every policy grid is 0.789, so anchoring the shared
# scale there rather than at 0.85 spends the full ink range without clipping.
POLICY_GRID_VALUE_MAX = 0.79

LAYOUT = PolicyGridLayout()
COLUMNS = (("Reference task", "reference"), ("Near shift", "task1"), ("Median shift", "task2"), ("Far shift", "task3"))


def render_lobeke_map(grid: np.ndarray | None, *, color: str, center_x_cm: float, bottom_y_cm: float,
        base_position: tuple[float, float], layout: PolicyGridLayout = LAYOUT,
        value_max: float = 0.85, show_base: bool = True) -> str:  # fmt: skip
    """Render one 5-by-5 Lobeke map with the publication-wide geometry.

    ``value_max`` controls only the color scale. Policy figures keep their
    shared probability scale of 0.85; environment schematics can instead use
    the natural maximum of each physical quantity without duplicating the map
    drawing code.
    """

    if grid is not None and (grid.shape != (5, 5) or not np.all(np.isfinite(grid))):
        raise ValueError("Lobeke map must be finite with shape (5, 5)")
    if not np.isfinite(value_max) or value_max <= 0:
        raise ValueError("Lobeke map value_max must be finite and strictly positive")
    x_cm = center_x_cm - layout.map_width_cm / 2
    commands = [
        rf"\begin{{scope}}[shift={{({x_cm:.3f}cm,{bottom_y_cm:.3f}cm)}}, "
        rf"x={layout.cell_cm:.3f}cm, y={layout.cell_cm:.3f}cm]"
    ]
    for row in range(5):
        for column in range(5):
            fraction = 0.03 if grid is None else float(np.clip(grid[row, column] / value_max, 0.0, 1.0))
            y_cell = 4 - row
            commands.append(
                rf"\fill[color={_tint(color, fraction)}] ({column},{y_cell}) rectangle "
                rf"({column + 1},{y_cell + 1});"
            )
    commands.extend(
        (
            r"\draw[white, line width=0.25pt, step=1] (0,0) grid (5,5);",
            r"\draw[mairlGrid, line width=0.35pt] (0,0) rectangle (5,5);",
        )
    )
    if show_base:
        base_x, base_y = base_position
        commands.append(
            rf"\fill[black] ({5 * base_x:.3f},{5 * base_y + 0.18:.3f}) -- "
            rf"({5 * base_x - 0.18:.3f},{5 * base_y - 0.14:.3f}) -- "
            rf"({5 * base_x + 0.18:.3f},{5 * base_y - 0.14:.3f}) -- cycle;"
        )
    commands.append(r"\end{scope}")
    return "\n".join(commands)


def render_policy_grid(rows: tuple[PolicyGridRow, ...],
        grids: Mapping[str, np.ndarray] | None, *, base_position: tuple[float, float],
        generated_by: str, layout: PolicyGridLayout = LAYOUT) -> str:  # fmt: skip
    """Render a complete grid with shared columns, maps, labels, and scores."""

    if not rows:
        raise ValueError("policy grid requires at least one protocol row")
    if len(layout.column_centers_cm) != len(COLUMNS):
        raise ValueError("policy-grid column geometry does not match the task columns")
    commands = [f"% Generated by {generated_by}; do not edit.", r"\begin{tikzpicture}[font=\small]"]
    for (label, _slug), center in zip(COLUMNS, layout.column_centers_cm, strict=True):
        commands.append(rf"\node[font=\scriptsize\bfseries] at ({center:.3f}cm,0.25cm) {{{label}}};")
    last_bottom = layout.row_bottom_cm(len(rows) - 1)
    commands.append(
        rf"\draw[black, line width=0.85pt] ({layout.separator_x_cm:.2f}cm,-0.05cm) -- "
        rf"({layout.separator_x_cm:.2f}cm,{last_bottom:.3f}cm);"
    )
    for row_index, row in enumerate(rows):
        bottom = layout.row_bottom_cm(row_index)
        label_y = bottom + layout.map_width_cm / 2
        commands.append(
            rf"\node[rotate=90, font=\scriptsize\bfseries, text={row.color}] "
            rf"at ({layout.label_x_cm:.2f}cm,{label_y:.3f}cm) {{{row.label}}};"
        )
        if row.scores is not None and len(row.scores) != len(COLUMNS):
            raise ValueError(f"row {row.slug!r} must provide one score per task")
        for column_index, ((_label, task), center) in enumerate(zip(COLUMNS, layout.column_centers_cm, strict=True)):
            key = f"{task}_{row.slug}"
            grid = None if grids is None else grids.get(key)
            if grids is not None and grid is None:
                raise ValueError(f"missing downstream policy grid: {key}")
            commands.append(
                render_lobeke_map(
                    grid,
                    color=row.color,
                    center_x_cm=center,
                    bottom_y_cm=bottom,
                    base_position=base_position,
                    layout=layout,
                    value_max=POLICY_GRID_VALUE_MAX,
                )
            )
            if row.scores is not None:
                commands.append(
                    rf"\node[mairl value box, anchor=north, text={row.color}, "
                    rf"draw={row.color}!45] at "
                    rf"({center:.3f}cm,{bottom + layout.map_width_cm - 0.06:.3f}cm) "
                    rf"{{${row.scores[column_index]}$}};"
                )
    commands.append(r"\end{tikzpicture}")
    return "\n".join(commands) + "\n"


@dataclass(frozen=True, slots=True)
class PairedPolicyGridLayout:
    """Geometry for the merged downstream grid.

    Each task column holds two equally sized maps, so the two data-access
    assumptions can be compared side by side within a task instead of in two
    separate figures. The reference row carries a single map, sized by
    ``reference_cell_cm`` so it can be emphasised over the methods it is
    compared against; at the method size it keeps the map-size equality
    :class:`PolicyGridLayout` documents.
    """

    cell_cm: float = 0.38
    reference_cell_cm: float = 0.38
    pair_gap_cm: float = 0.16
    group_gap_cm: float = 0.50
    left_margin_cm: float = 0.20
    header_y_cm: float = 2.75
    reference_bottom_cm: float = 0.40
    first_row_bottom_cm: float = -2.20
    row_step_cm: float = 2.90
    score_gap_cm: float = -0.04
    note_gap_cm: float = 0.00

    @property
    def map_width_cm(self) -> float:
        return 5 * self.cell_cm

    @property
    def reference_map_width_cm(self) -> float:
        return 5 * self.reference_cell_cm

    @property
    def group_width_cm(self) -> float:
        return 2 * self.map_width_cm + self.pair_gap_cm

    @property
    def slot_offset_cm(self) -> float:
        return (self.map_width_cm + self.pair_gap_cm) / 2

    @property
    def separator_x_cm(self) -> float:
        return self.group_center_cm(0) + self.group_width_cm / 2 + self.group_gap_cm / 2

    def group_center_cm(self, index: int) -> float:
        step = self.group_width_cm + self.group_gap_cm
        return self.left_margin_cm + self.group_width_cm / 2 + step * index

    def row_bottom_cm(self, index: int) -> float:
        return self.first_row_bottom_cm - self.row_step_cm * index


@dataclass(frozen=True, slots=True)
class PolicyGridCell:
    """One map in a paired grid, with its optional SubOpt value and footnote."""

    slug: str
    color: str
    scores: tuple[str, ...] | None = None
    notes: tuple[str, ...] | None = None


PAIRED_LAYOUT = PairedPolicyGridLayout()


def _paired_cell(cell: PolicyGridCell, grids: Mapping[str, np.ndarray], *, task: str,
        column_index: int, center_x_cm: float, bottom_y_cm: float, base_position: tuple[float, float],
        layout: PairedPolicyGridLayout, cell_layout: PolicyGridLayout) -> str:  # fmt: skip
    key = f"{task}_{cell.slug}"
    grid = grids.get(key)
    if grid is None:
        raise ValueError(f"missing downstream policy grid: {key}")
    commands = [
        render_lobeke_map(
            grid,
            color=cell.color,
            center_x_cm=center_x_cm,
            bottom_y_cm=bottom_y_cm,
            base_position=base_position,
            layout=cell_layout,
            value_max=POLICY_GRID_VALUE_MAX,
        )
    ]
    top_y_cm = bottom_y_cm + cell_layout.map_width_cm
    if cell.scores is not None:
        commands.append(
            rf"\node[font=\small, anchor=south, text={cell.color}] at "
            rf"({center_x_cm:.3f}cm,{top_y_cm + layout.score_gap_cm:.3f}cm) "
            rf"{{${cell.scores[column_index]}$}};"
        )
    if cell.notes is not None:
        commands.append(
            rf"\node[font=\fontsize{{6}}{{6.5}}\selectfont, anchor=north, text={cell.color}] at "
            rf"({center_x_cm:.3f}cm,{bottom_y_cm - layout.note_gap_cm:.3f}cm) "
            rf"{{(w/ ${cell.notes[column_index]}$)}};"
        )
    return "\n".join(commands)


def render_paired_policy_grid(reference: PolicyGridCell, rows: tuple[tuple[PolicyGridCell, PolicyGridCell], ...],
        grids: Mapping[str, np.ndarray], *, base_position: tuple[float, float],
        generated_by: str, layout: PairedPolicyGridLayout = PAIRED_LAYOUT) -> str:  # fmt: skip
    """Render one grid whose task columns each hold a left and a right method.

    The fragment scales itself to the text width. A row key at document size
    has to accompany this grid, and that key must not shrink with the maps, so
    the sizing decision belongs here rather than in the paper source.
    """

    if not rows:
        raise ValueError("paired policy grid requires at least one method row")
    for left, right in rows:
        for cell in (left, right):
            if cell.scores is not None and len(cell.scores) != len(COLUMNS):
                raise ValueError(f"cell {cell.slug!r} must provide one score per task")
            if cell.notes is not None and len(cell.notes) != len(COLUMNS):
                raise ValueError(f"cell {cell.slug!r} must provide one note per task")
    cell_layout = PolicyGridLayout(cell_cm=layout.cell_cm)
    reference_layout = PolicyGridLayout(cell_cm=layout.reference_cell_cm)
    centers = tuple(layout.group_center_cm(index) for index in range(len(COLUMNS)))

    commands = [r"\begin{tikzpicture}[font=\small]"]
    for (label, task), center in zip(COLUMNS, centers, strict=True):
        commands.append(
            rf"\node[font=\normalsize\bfseries] at "
            rf"({center:.3f}cm,{layout.header_y_cm:.2f}cm) {{{label}}};"
        )
        commands.append(
            _paired_cell(
                reference,
                grids,
                task=task,
                column_index=0,
                center_x_cm=center,
                bottom_y_cm=layout.reference_bottom_cm,
                base_position=base_position,
                layout=layout,
                cell_layout=reference_layout,
            )
        )
    separator_top = layout.reference_bottom_cm + reference_layout.map_width_cm + 0.15
    commands.append(
        rf"\draw[black, line width=0.85pt] "
        rf"({layout.separator_x_cm:.3f}cm,{separator_top:.3f}cm) -- "
        rf"({layout.separator_x_cm:.3f}cm,{layout.row_bottom_cm(len(rows) - 1):.3f}cm);"
    )
    for row_index, (left, right) in enumerate(rows):
        bottom = layout.row_bottom_cm(row_index)
        for column_index, ((_label, task), center) in enumerate(zip(COLUMNS, centers, strict=True)):
            for cell, direction in ((left, -1), (right, 1)):
                commands.append(
                    _paired_cell(
                        cell,
                        grids,
                        task=task,
                        column_index=column_index,
                        center_x_cm=center + direction * layout.slot_offset_cm,
                        bottom_y_cm=bottom,
                        base_position=base_position,
                        layout=layout,
                        cell_layout=cell_layout,
                    )
                )
    commands.append(r"\end{tikzpicture}")
    picture = "\n".join(commands)
    return f"% Generated by {generated_by}; do not edit.\n\\resizebox{{\\textwidth}}{{!}}{{%\n{picture}%\n}}\n"
