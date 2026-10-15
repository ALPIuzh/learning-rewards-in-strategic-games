"""Plot entry points must initialize Matplotlib before it chooses a cache.

`mairl_exp.evaluation.mpl` sets MPLCONFIGDIR and selects Agg *before* importing
matplotlib. Because the cache directory is fixed at matplotlib's first import in
a process, it cannot be corrected afterwards: a module that imports matplotlib
directly, above the shim, breaks the invariant silently. These tests pin both
halves — statically, that nothing bypasses the shim, and at runtime, that
importing a plot module really does configure the process.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SHIM = "mairl_exp.evaluation.mpl"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SEARCH_ROOTS = ("src", "experiments", "reproducibility")
# Vendored third-party code imports matplotlib directly by design.
_EXCLUDED = ("experiments/guard/GUARD", "archive")


def _source_files(repository_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in _SEARCH_ROOTS:
        for path in (repository_root / root).rglob("*.py"):
            relative = path.relative_to(repository_root).as_posix()
            if not any(relative.startswith(prefix) for prefix in _EXCLUDED):
                files.append(path)
    return files


def _module_name(repository_root: Path, path: Path) -> str:
    relative = path.relative_to(repository_root)
    parts = list(relative.with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _matplotlib_imports(path: Path) -> list[str]:
    """Top-level matplotlib modules imported by this file, shim included."""

    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.append(node.module)
    return [name for name in found if name == SHIM or name.split(".")[0] == "matplotlib"]


def _shim_consumers(repository_root: Path) -> list[str]:
    modules = [
        _module_name(repository_root, path)
        for path in _source_files(repository_root)
        if SHIM in _matplotlib_imports(path)
    ]
    return sorted(modules)


def test_every_matplotlib_import_goes_through_the_shim(repository_root) -> None:
    """Nothing may import matplotlib directly; the shim must be the only importer.

    This is the static half of the invariant, and it covers every module at once
    rather than an enumerated list that a new plot module would not join.
    """

    violations = []
    for path in _source_files(repository_root):
        module = _module_name(repository_root, path)
        if module == SHIM:
            continue
        direct = [name for name in _matplotlib_imports(path) if name != SHIM]
        violations.extend(f"{module} imports {name}" for name in direct)

    assert not violations, "matplotlib imported outside the shim:\n" + "\n".join(violations)


def test_the_shim_is_actually_used_somewhere(repository_root) -> None:
    """Guard against the static test above passing because nothing plots at all."""

    assert len(_shim_consumers(repository_root)) >= 5


def _cases(repository_root: Path) -> list[tuple[str, bool]]:
    return [(module, custom) for module in _shim_consumers(repository_root) for custom in (False, True)]


@pytest.fixture(scope="session")
def bootstrap_processes(tmp_path_factory):
    """Launch every fresh-process probe at once, then collect.

    Each child rebuilds matplotlib's font cache from cold, which dominates the
    runtime; running them concurrently keeps each process genuinely cold while
    paying for the rebuild once in wall-clock terms.
    """

    repository_root = _REPOSITORY_ROOT
    launched = {}
    for module, custom in _cases(repository_root):
        directory = tmp_path_factory.mktemp("bootstrap")
        expected = directory / ("custom-cache" if custom else ".matplotlib-cache")
        env = os.environ.copy()
        for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "MPLBACKEND"):
            env.pop(name, None)
        env["PYTHONPATH"] = os.pathsep.join((str(repository_root), str(repository_root / "src")))
        if custom:
            env["MPLCONFIGDIR"] = str(expected)
        code = (
            f"import {module}; import json, matplotlib; "
            "print(json.dumps([matplotlib.get_configdir(), matplotlib.get_backend()]))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        launched[(module, custom)] = (process, expected)

    results = {}
    for key, (process, expected) in launched.items():
        stdout, stderr = process.communicate(timeout=300)
        results[key] = (process.returncode, stdout, stderr, expected)
    return results


def pytest_generate_tests(metafunc):
    if "bootstrap_case" in metafunc.fixturenames:
        cases = _cases(_REPOSITORY_ROOT)
        metafunc.parametrize(
            "bootstrap_case",
            cases,
            ids=[f"{module.rsplit('.', 1)[-1]}-{'custom' if custom else 'default'}" for module, custom in cases],
        )


def test_plot_import_configures_cache_in_fresh_process(bootstrap_processes, bootstrap_case) -> None:
    returncode, stdout, stderr, expected = bootstrap_processes[bootstrap_case]
    assert returncode == 0, stderr
    cache, backend = json.loads(stdout)
    assert Path(cache) == expected
    assert backend.casefold() == "agg"
