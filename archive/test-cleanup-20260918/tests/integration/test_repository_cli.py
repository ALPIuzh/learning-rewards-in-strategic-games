"""Behavioral checks for the source-checkout experiment interface."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "expected_fragment"),
    (
        ("experiments.run", "experiments.toy.application.ToyApplication"),
        ("experiments.plot", "mairl_exp.evaluation.plotting.FisherTransferPlot"),
    ),
)
def test_repository_commands_resolve_tracked_configuration(repository_root,
        module: str, expected_fragment: str) -> None:  # fmt: skip
    result = subprocess.run(
        (sys.executable, "-m", module, "--cfg"), cwd=repository_root, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert expected_fragment in result.stdout
    assert "schema_version: 1" in result.stdout
