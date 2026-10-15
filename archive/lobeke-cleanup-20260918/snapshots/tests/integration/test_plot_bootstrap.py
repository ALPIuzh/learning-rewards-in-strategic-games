"""Plot entry points must initialize Matplotlib before it chooses a cache."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module", ["experiments.guard.paper.figures.joint_discovery", "experiments.guard.paper.figures.style"]
)
@pytest.mark.parametrize("custom_cache", [False, True])
def test_plot_import_configures_cache_in_fresh_process(repository_root, tmp_path, module, custom_cache):
    env = os.environ.copy()
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "MPLBACKEND"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join((str(repository_root), str(repository_root / "src")))
    expected = tmp_path / ("custom-cache" if custom_cache else ".matplotlib-cache")
    if custom_cache:
        env["MPLCONFIGDIR"] = str(expected)
    code = (
        f"import {module}; import json, matplotlib; "
        "print(json.dumps([matplotlib.get_configdir(), matplotlib.get_backend()]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=60
    )
    cache, backend = json.loads(result.stdout)
    assert Path(cache) == expected
    assert backend.casefold() == "agg"
