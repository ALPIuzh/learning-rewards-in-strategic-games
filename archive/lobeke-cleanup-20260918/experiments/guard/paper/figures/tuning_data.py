"""Export the protected Lobeke tuning runs as a compact tracked data file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mairl.exceptions import ConfigurationError
from mairl_exp.config import REPOSITORY_ROOT, RUNS_ROOT
from mairl_exp.execution.artifacts import atomic_write_json, sha256_file

DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "data" / "lobeke" / "tuning.json"
DEFAULT_RUNS = {
    "warm_selection": RUNS_ROOT / "guard-lobeke-warm-selection" / "2ce0376fe643084a",
    "no_warm_selection": RUNS_ROOT / "guard-lobeke-no-warm-selection" / "9e985f76d0b8ed3a",
    "warm_confirmation": RUNS_ROOT / "guard-lobeke-warm-confirmation" / "29cf0e2e1ef7d309",
    "no_warm_confirmation": RUNS_ROOT / "guard-lobeke-no-warm-confirmation" / "c7c889048653fc2a",
}


def _load_complete_run(run_dir: Path, result_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    result_path = run_dir / result_name
    if not manifest_path.exists() or not result_path.exists():
        raise FileNotFoundError(f"incomplete tuning artifact directory: {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    if manifest["status"] != "complete" or manifest["progress"]["failed"] != 0:
        raise ConfigurationError(f"source tuning run is not complete: {run_dir}")
    return manifest, result


def _source(run_dir: Path, manifest: dict[str, Any], result_name: str) -> dict[str, Any]:
    return {
        "study_name": manifest["study_name"],
        "run_id": manifest["run_id"],
        "semantic_config_sha256": manifest["semantic_config_sha256"],
        "git_commit": manifest["git"]["commit"],
        "result_file": result_name,
        "result_sha256": sha256_file(run_dir / result_name),
    }


def _selection_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for key, metrics in payload["statistics"].items():
        warmup, multiplier = key.split(",")
        candidates.append(
            {
                "warmup_budget": int(warmup.split("=")[1]),
                "lambda_information_multiplier": float(multiplier.split("=")[1]),
                "stackelberg_suboptimality": metrics["stackelberg_suboptimality"],
                "fisher_transfer_error": metrics["fisher_transfer_error"],
            }
        )
    return sorted(candidates, key=lambda item: (item["warmup_budget"], item["lambda_information_multiplier"]))


def export(*, warm_selection_dir: Path, no_warm_selection_dir: Path,
        warm_confirmation_dir: Path, no_warm_confirmation_dir: Path, output: Path) -> None:  # fmt: skip
    run_dirs = {
        "warm_selection": warm_selection_dir,
        "no_warm_selection": no_warm_selection_dir,
        "warm_confirmation": warm_confirmation_dir,
        "no_warm_confirmation": no_warm_confirmation_dir,
    }
    loaded = {
        key: _load_complete_run(run_dir, "joint_discovery.json" if "selection" in key else "confirmation.json")
        for key, run_dir in run_dirs.items()
    }
    warm_selection = loaded["warm_selection"][1]
    no_warm_selection = loaded["no_warm_selection"][1]
    warm_confirmation = loaded["warm_confirmation"][1]
    no_warm_confirmation = loaded["no_warm_confirmation"][1]
    if warm_confirmation["selection_run_id"] != loaded["warm_selection"][0]["run_id"]:
        raise ConfigurationError("warm confirmation does not reference the supplied selection")
    if no_warm_confirmation["selection_run_id"] != loaded["no_warm_selection"][0]["run_id"]:
        raise ConfigurationError("no-warm confirmation does not reference the supplied selection")
    if warm_confirmation["status"] != "confirmed" or no_warm_confirmation["status"] != "confirmed":
        raise ConfigurationError("both frozen Lobeke candidates must pass confirmation")

    payload = {
        "schema_version": 1,
        "sources": {
            key: _source(run_dir, loaded[key][0], "joint_discovery.json" if "selection" in key else "confirmation.json")
            for key, run_dir in run_dirs.items()
        },
        "selection": {
            "warmup": {
                "selected": warm_confirmation["candidate"],
                "reference": warm_confirmation["reference"],
                "candidates": _selection_candidates(warm_selection),
            },
            "no_warmup": {
                "selected": no_warm_confirmation["candidate"],
                "reference": no_warm_confirmation["reference"],
                "candidates": _selection_candidates(no_warm_selection),
            },
        },
        "confirmation": [
            {
                "mode": mode,
                "candidate": confirmation["candidate"],
                "reference": confirmation["reference"],
                "paired_differences": confirmation["paired_differences_candidate_minus_reference"],
            }
            for mode, confirmation in (("warmup", warm_confirmation), ("no_warmup", no_warm_confirmation))
        ],
        "seed_policy": {
            "selection": "10000..10063",
            "confirmation": "20000..20063",
            "final_evaluation": "30000..30049 (reserved; not used in this data extract)",
            "selection_and_confirmation_disjoint": True,
        },
    }
    atomic_write_json(output, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULT_RUNS.items():
        parser.add_argument(f"--{key.replace('_', '-')}-dir", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    export(
        warm_selection_dir=args.warm_selection_dir,
        no_warm_selection_dir=args.no_warm_selection_dir,
        warm_confirmation_dir=args.warm_confirmation_dir,
        no_warm_confirmation_dir=args.no_warm_confirmation_dir,
        output=args.output,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
