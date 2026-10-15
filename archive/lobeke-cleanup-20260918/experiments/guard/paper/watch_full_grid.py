"""Refresh the Figure 7 TeX and preview whenever a tuning cell completes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from mairl_exp.config import REPOSITORY_ROOT


def _run(command: list[str]) -> None:
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(REPOSITORY_ROOT / ".matplotlib-cache"),
        "UV_CACHE_DIR": str(REPOSITORY_ROOT / ".uv-cache"),
    }
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def _refresh(run_dir: Path) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "experiments.guard.paper.figures.full_grid_tuning",
            "--completion-run-dir",
            str(run_dir),
        ]
    )
    _run(["uv", "run", "python", "-m", "reproducibility.figures", "generate", "i1_hyperparameters"])
    (REPOSITORY_ROOT / "output" / "pdf").mkdir(parents=True, exist_ok=True)
    _run(
        [
            "latexmk",
            "-g",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={REPOSITORY_ROOT / 'output' / 'pdf'}",
            "reproducibility/tex/preview.tex",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    last_complete = -1
    while True:
        if not manifest_path.exists():
            time.sleep(args.interval)
            continue
        manifest = json.loads(manifest_path.read_text())
        progress = manifest.get("progress", {})
        complete = int(progress.get("complete", 0))
        if complete != last_complete:
            _refresh(run_dir)
            last_complete = complete
        if manifest.get("status") in {"complete", "failed", "interrupted"}:
            if complete == last_complete:
                print(f"watcher stopped: run status={manifest['status']}, complete={complete}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
