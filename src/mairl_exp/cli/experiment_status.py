"""Inspect progress for a durable MAIRL experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mairl.exceptions import ConfigurationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    return parser


def _read_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise ConfigurationError(f"run manifest does not exist: {path}")
    return json.loads(path.read_text())


def _display(manifest: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    progress = manifest["progress"]
    if not isinstance(progress, dict):
        raise ConfigurationError("run manifest has invalid progress data")
    eta = progress.get("estimated_remaining_seconds")
    eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
    print(
        f"run={manifest['run_id']} status={manifest['status']} "
        f"complete={progress.get('complete', 0)}/{progress.get('total', 0)} "
        f"failed={progress.get('failed', 0)} running={progress.get('running', 0)} "
        f"pending={progress.get('pending', 0)} ETA={eta_text}"
    )
    if progress.get("current_job"):
        print(f"current_job={progress['current_job']}")
    current_progress = progress.get("current_job_progress")
    if isinstance(current_progress, dict) and current_progress.get("completed_steps") is not None:
        print(
            "job_progress="
            f"{current_progress['completed_steps']}/{current_progress['total_steps']} "
            f"heartbeat={current_progress['heartbeat_at']}"
        )
    print(f"updated_at={manifest['updated_at']}")


def main() -> None:
    args = _parser().parse_args()
    if args.interval <= 0:
        raise ConfigurationError("--interval must be strictly positive")
    while True:
        manifest = _read_manifest(args.run_dir)
        _display(manifest, as_json=args.json)
        if not args.watch or manifest["status"] in {"complete", "failed"}:
            return
        time.sleep(args.interval)
