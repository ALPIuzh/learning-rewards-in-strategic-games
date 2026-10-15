from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mairl_exp.config import REPOSITORY_ROOT
from reproducibility.figures.recipes.lobeke import render_convergence_figure, render_transfer_figure


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
