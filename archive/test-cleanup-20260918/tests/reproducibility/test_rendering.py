from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mairl_exp.config import GENERATED_ROOT, REPOSITORY_ROOT
from reproducibility.figures.manifest import load_manifest
from reproducibility.figures.policy_grid import COLUMNS, LAYOUT, PAIRED_LAYOUT
from reproducibility.figures.recipes.lobeke import render_convergence_figure, render_transfer_figure
from reproducibility.figures.render import render_figures, render_text, write_build_manifest


def test_expected_paper_figures_are_registered() -> None:
    manifest = load_manifest()
    identifiers = {figure.id for figure in manifest.figures}
    assert {
        "m1_source_convergence",
        "m2_downstream_transfer",
        "m2_downstream_transfer_compact",
        "i1_environment",
        "i1_source_diagnostics",
        "i1_downstream",
        "i1_behavioral_transfer_gap",
        "i1_hyperparameters",
        "i2_sparse_information",
        "i3_lower_bound",
    } <= identifiers


def test_every_registered_figure_renders_as_a_fragment() -> None:
    for figure in load_manifest().figures:
        rendered, _digest = render_text(figure)
        assert "\\begin{tikzpicture}" in rendered
        assert "\\end{tikzpicture}" in rendered
        assert "\\documentclass" not in rendered
        assert "\\begin{document}" not in rendered
        assert "\\caption" not in rendered


def test_tracked_tikz_and_build_manifest_are_current() -> None:
    figures = load_manifest().figures
    records = render_figures(figures, check=True)
    write_build_manifest(records, check=True)


def test_generated_outputs_stay_inside_reproducibility_tree() -> None:
    for figure in load_manifest().figures:
        figure.output_path.relative_to(GENERATED_ROOT)
        assert figure.output_path.suffix == ".tex"


def _lobeke_recipe_fragments() -> tuple[str, str]:
    checkpoints = np.asarray([400.0, 800.0, 1_200.0, 1_600.0])
    method_keys = (
        ("active", True),
        ("active", False),
        ("fixed_random", True),
        ("fixed_random", False),
        ("random_each_round", True),
        ("random_each_round", False),
        ("fixed_expert_mle", False),
        ("zero_sum_ols", False),
        ("follower_ols", False),
    )
    traces = {
        key: [
            SimpleNamespace(
                checkpoints=checkpoints,
                metrics={
                    "stackelberg_suboptimality": np.asarray([0.02, 0.008, 0.003, 0.001], dtype=np.float64)
                    * (1.0 + 0.05 * seed)
                },
            )
            for seed in range(3)
        ]
        for key in method_keys
    }
    convergence = render_convergence_figure(traces, confidence_level=0.95, warmup_budget=200)
    tasks = [{"mean": value} for value in (1e-3, 2e-3, 3e-3)]
    summary = {
        "methods": {
            "active": {"reference": {"mean": 5e-4}, "tasks": tasks},
            "expert": {"reference": {"mean": 8e-4}, "tasks": tasks},
        }
    }
    grids = {
        f"{task}_{method}": np.full((5, 5), 0.2, dtype=np.float64)
        for task in ("reference", "task1", "task2", "task3")
        for method in ("oracle", "active", "expert")
    }
    transfer = render_transfer_figure(
        summary, grids, base_position=(0.5, 0.5), baseline_key=("fixed_expert_mle", False)
    )
    return convergence, transfer


def test_main_and_appendix_policy_maps_share_exact_geometry() -> None:
    _convergence, main_transfer = _lobeke_recipe_fragments()
    manifest = load_manifest()
    figure = next(figure for figure in manifest.figures if figure.id == "i1_downstream")
    rendered, _digest = render_text(figure)

    method_token = f"x={LAYOUT.cell_cm:.3f}cm, y={LAYOUT.cell_cm:.3f}cm"
    assert method_token in main_transfer
    for center in LAYOUT.column_centers_cm:
        assert f"({center:.3f}cm,0.25cm)" in main_transfer

    # Every compared method keeps the main-text map size; only the reference row,
    # which is the target rather than a competitor, is drawn larger.
    reference_token = f"x={PAIRED_LAYOUT.reference_cell_cm:.3f}cm, y={PAIRED_LAYOUT.reference_cell_cm:.3f}cm"
    assert PAIRED_LAYOUT.cell_cm == LAYOUT.cell_cm
    methods, reference = 6 * len(COLUMNS), len(COLUMNS)
    if reference_token == method_token:
        assert rendered.count(method_token) == methods + reference
    else:
        assert rendered.count(method_token) == methods
        assert rendered.count(reference_token) == reference
    for index in range(len(COLUMNS)):
        center = PAIRED_LAYOUT.group_center_cm(index)
        assert f"({center:.3f}cm,{PAIRED_LAYOUT.header_y_cm:.2f}cm)" in rendered


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_shared_style_and_figure_fragments_compile(tmp_path: Path) -> None:
    convergence, transfer = _lobeke_recipe_fragments()
    convergence_path = tmp_path / "convergence.tex"
    transfer_path = tmp_path / "transfer.tex"
    convergence_path.write_text(convergence)
    transfer_path.write_text(transfer)
    document = tmp_path / "fragments.tex"
    document.write_text(
        rf"""\documentclass{{article}}
\input{{reproducibility/tex/tex_base}}
\begin{{document}}
\input{{{convergence_path.as_posix()}}}
\par\bigskip
\resizebox{{\textwidth}}{{!}}{{\input{{{transfer_path.as_posix()}}}}}
\clearpage
\input{{reproducibility/tex/generated/appendix/i1_source_diagnostics}}
\par\bigskip
\input{{reproducibility/tex/generated/appendix/i1_downstream}}
\end{{document}}
"""
    )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp_path}", str(document)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:]
