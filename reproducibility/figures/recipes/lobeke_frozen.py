"""Render the main Lobeke figures from checksum-protected numerical inputs."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reproducibility.figures.recipes.lobeke import (
    MAIN_TRANSFER_BASELINE,
    render_compact_transfer_figure,
    render_convergence_figure,
    render_downstream_transfer_figure,
    render_transfer_figure,
)


def _checked(path: Path, digest: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"frozen figure input failed its checksum: {path.name}")
    return raw


def render_source(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    metadata = json.loads(raw)
    if metadata.get("schema_version") != 1 or metadata.get("checkpoint_stride") != 1:
        raise ValueError("invalid full-resolution source-convergence schema")
    data_path = path.with_name("data.npz")
    data = _checked(data_path, metadata["data_sha256"])
    traces = {}
    with np.load(data_path, allow_pickle=False) as arrays:
        for protocol in metadata["protocols"]:
            slug = protocol["slug"]
            checkpoints = arrays[f"{slug}__checkpoints"]
            seeds = arrays[f"{slug}__seeds"]
            values = arrays[f"{slug}__suboptimality"]
            if (
                checkpoints.tolist() != protocol["checkpoints"]
                or seeds.tolist() != protocol["seeds"]
                or values.shape != (len(seeds), len(checkpoints))
                or not np.isfinite(values).all()
                or np.any(np.diff(checkpoints) <= 0)
                or len(set(seeds.tolist())) != len(seeds)
            ):
                raise ValueError(f"invalid source-convergence arrays: {slug}")
            key = (protocol["method"], protocol["warmup"])
            if key in traces:
                raise ValueError("duplicate source-convergence protocol")
            traces[key] = [
                SimpleNamespace(checkpoints=checkpoints, metrics={"stackelberg_suboptimality": row}) for row in values
            ]
    return render_convergence_figure(
        traces, confidence_level=metadata["confidence_level"], warmup_budget=metadata["warmup_budget"]
    ), hashlib.sha256(raw + data).hexdigest()


def render_transfer(path: Path, recipe: str, repository_root: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    metadata = json.loads(raw)
    if metadata.get("schema_version") != 1:
        raise ValueError("invalid frozen-transfer schema")
    inputs = {}
    combined = raw
    for name in ("summary", "policy_grids", "environment"):
        record = metadata["inputs"][name]
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("frozen-transfer inputs must be repository-relative")
        input_path = repository_root / relative
        inputs[name] = _checked(input_path, record["sha256"])
        combined += inputs[name]
    summary = json.loads(inputs["summary"])
    environment = json.loads(inputs["environment"])
    base_position = tuple(environment["model"]["ranger_base_position"])
    with np.load(io.BytesIO(inputs["policy_grids"]), allow_pickle=False) as arrays:
        grids = {name: arrays[name].copy() for name in arrays.files}
    if any(not np.isfinite(grid).all() for grid in grids.values()):
        raise ValueError("non-finite frozen policy grid")
    if recipe == "lobeke_downstream":
        text = render_downstream_transfer_figure(summary, grids, base_position=base_position)
    else:
        renderer = render_compact_transfer_figure if recipe == "lobeke_transfer_compact" else render_transfer_figure
        text = renderer(summary, grids, base_position=base_position, baseline_key=MAIN_TRANSFER_BASELINE)
    return text, hashlib.sha256(combined).hexdigest()
