"""Behavioral checks for the source-checkout experiment interface."""

from __future__ import annotations

import subprocess
import sys

import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    ("module", "config_section"), (("experiments.run", "application"), ("experiments.plot", "plot"))
)
def test_repository_commands_resolve_tracked_configuration(repository_root,
        module: str, config_section: str) -> None:  # fmt: skip
    result = subprocess.run(
        (sys.executable, "-m", module, "--cfg"), cwd=repository_root, check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    config = OmegaConf.create(result.stdout)
    get_class(config[config_section]["_target_"])
