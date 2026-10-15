"""Export all source-convergence checkpoints from verified runs, without simulations."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.common import _portable_path, _source_record
from experiments.guard.paper.figures.corrected_traces import load_figure_run_traces
from experiments.guard.paper.figures.data import (
    METHOD_SLUGS,
    infer_warmup_budget,
    load_runtime,
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_in_unit_reward,
)
from mairl_exp.execution.artifacts import sha256_file


def export(run_dirs: list[Path], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; preserve frozen exports")
    sources = [_source_record(path, json.loads((path / "manifest.json").read_text())) for path in run_dirs]
    traces = load_figure_run_traces(run_dirs)
    markers = {run_source_suboptimality_is_unit_reward(path) for path in run_dirs}
    if len(markers) != 1:
        raise ValueError("source runs use different reward units")
    normalization = load_runtime(run_dirs[0]).metadata["leader_reward_normalization"]
    traces = source_suboptimality_in_unit_reward(
        traces, raw_reward_range=float(normalization["raw_range"]), already_unit_reward=markers.pop()
    )
    arrays = {}
    protocols = []
    for key, slug in METHOD_SLUGS.items():
        items = traces[key]
        checkpoints = items[0].checkpoints
        if any(not np.array_equal(item.checkpoints, checkpoints) for item in items):
            raise ValueError(f"inconsistent checkpoints for {key}")
        arrays[f"{slug}__checkpoints"] = checkpoints
        arrays[f"{slug}__seeds"] = np.asarray([item.seed for item in items])
        arrays[f"{slug}__suboptimality"] = np.stack([item.metrics["stackelberg_suboptimality"] for item in items])
        protocols.append(
            {
                "method": key[0],
                "warmup": key[1],
                "slug": slug,
                "seeds": arrays[f"{slug}__seeds"].tolist(),
                "checkpoints": checkpoints.tolist(),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"
    np.savez_compressed(data_path, **arrays)
    provenance = {
        "schema_version": 1,
        "experiment": "full-resolution Lobeke source convergence",
        "sources": sources,
        "generation_command": shlex.join(
            [
                "python",
                "-m",
                "experiments.guard.paper.figures.export_source_convergence",
                *[part for path in run_dirs for part in ("--run-dir", _portable_path(path))],
                "--output-dir",
                _portable_path(output_dir),
            ]
        ),
        "protocols": protocols,
        "confidence_level": 0.95,
        "warmup_budget": infer_warmup_budget(traces),
        "reward_units": "positive-affine leader reward in [0,1]",
        "leader_reward_normalization": normalization,
        "checkpoint_stride": 1,
        "data_sha256": sha256_file(data_path),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
