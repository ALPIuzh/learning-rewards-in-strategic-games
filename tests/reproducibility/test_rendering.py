"""Compile the registered paper figures with the shared TeX preamble."""

import shutil
import subprocess

import pytest

from mairl_exp.config import REPOSITORY_ROOT
from reproducibility.figures.manifest import load_manifest


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_shared_style_and_figure_fragments_compile(tmp_path) -> None:
    inputs = "\n\\clearpage\n".join(rf"\input{{{figure.output_path.as_posix()}}}" for figure in load_manifest().figures)
    document = tmp_path / "fragments.tex"
    document.write_text(
        rf"""\documentclass{{article}}
\input{{reproducibility/tex/tex_base}}
\begin{{document}}
{inputs}
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
