"""Deterministic rendering and provenance for registered figures."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mairl_exp.config import GENERATED_ROOT, REPOSITORY_ROOT
from mairl_exp.execution.artifacts import sha256_bytes
from reproducibility.figures.manifest import FigureSpec
from reproducibility.figures.recipes.directional_transfer import render_directional_transfer
from reproducibility.figures.recipes.lobeke_behavioral_transfer import render_lobeke_behavioral_transfer
from reproducibility.figures.recipes.lobeke_diagnostics import render_lobeke_diagnostics
from reproducibility.figures.recipes.lobeke_environment import render_lobeke_environment
from reproducibility.figures.recipes.lobeke_frozen import render_source, render_transfer
from reproducibility.figures.recipes.lobeke_tuning import render_lobeke_tuning
from reproducibility.figures.recipes.lower_bound import render_lower_bound
from reproducibility.figures.recipes.sparse_information import render_sparse_information
from reproducibility.figures.recipes.zero_sum import render_zero_sum

BUILD_MANIFEST = GENERATED_ROOT / "build_manifest.json"
FORBIDDEN_PUBLICATION_TERMS = ("legacy",)


@dataclass(frozen=True, slots=True)
class RenderedFigure:
    id: str
    output: str
    output_sha256: str
    source: str | None
    source_sha256: str | None
    data: str | None
    data_sha256: str | None


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def render_text(spec: FigureSpec) -> tuple[str, str | None]:
    """Render a figure to text and return its optional source digest."""

    if spec.recipe in {"lobeke_source_convergence", "lobeke_transfer", "lobeke_transfer_compact", "lobeke_downstream"}:
        assert spec.data is not None
        if spec.recipe == "lobeke_source_convergence":
            body, digest = render_source(spec.data)
        else:
            body, digest = render_transfer(spec.data, spec.recipe, REPOSITORY_ROOT)
        header = f"% Derived from {_relative(spec.data)} (combined sha256: {digest}).\n"
        return header + body, digest
    if spec.recipe == "copy":
        assert spec.source is not None
        source = spec.source.read_text()
        if "\\documentclass" in source or "\\begin{document}" in source:
            raise ValueError(f"figure source must be a fragment, not a document: {spec.source}")
        digest = sha256_bytes(source.encode())
        header = f"% Imported from {_relative(spec.source)} (sha256: {digest}).\n"
        return header + source, digest
    if spec.recipe == "lobeke_tuning":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        payload = json.loads(raw)
        data_path = spec.data.with_name(payload["data_file"])
        data = data_path.read_bytes()
        if sha256_bytes(data) != payload["data_sha256"]:
            raise ValueError("sensitivity data checksum mismatch")
        digest = sha256_bytes(raw + data)
        header = f"% Derived from {_relative(spec.data)} and {_relative(data_path)} (sha256: {digest}).\n"
        return header + render_lobeke_tuning(data_path), digest
    if spec.recipe == "directional_transfer":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        data_path = spec.data.with_suffix(".npz")
        digest = sha256_bytes(raw + data_path.read_bytes())
        header = f"% Derived from {_relative(spec.data)} and {_relative(data_path)} (combined sha256: {digest}).\n"
        return header + render_directional_transfer(json.loads(raw), spec.data), digest
    if spec.recipe == "lobeke_diagnostics":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        data_path = spec.data.with_name("data.npz")
        digest = sha256_bytes(raw + data_path.read_bytes())
        header = f"% Derived from {_relative(spec.data)} and {_relative(data_path)} (combined sha256: {digest}).\n"
        return header + render_lobeke_diagnostics(json.loads(raw), spec.data), digest
    if spec.recipe == "lobeke_behavioral_transfer":
        assert spec.data is not None and spec.variant is not None
        raw = spec.data.read_bytes()
        data_path = spec.data.with_name("data.npz")
        digest = sha256_bytes(raw + data_path.read_bytes())
        header = f"% Derived from {_relative(spec.data)} and {_relative(data_path)} (combined sha256: {digest}).\n"
        return (header + render_lobeke_behavioral_transfer(json.loads(raw), spec.data, metric=spec.variant), digest)
    if spec.recipe == "lobeke_environment":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        data_path = spec.data.with_name("data.npz")
        digest = sha256_bytes(raw + data_path.read_bytes())
        header = f"% Derived from {_relative(spec.data)} and {_relative(data_path)} (combined sha256: {digest}).\n"
        return header + render_lobeke_environment(json.loads(raw), spec.data), digest
    if spec.recipe == "zero_sum":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        digest = sha256_bytes(raw)
        header = f"% Derived from {_relative(spec.data)} (sha256: {digest}).\n"
        return header + render_zero_sum(json.loads(raw)), digest
    if spec.recipe == "lower_bound":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        digest = sha256_bytes(raw)
        header = f"% Derived from {_relative(spec.data)} (sha256: {digest}).\n"
        return header + render_lower_bound(json.loads(raw)), digest
    if spec.recipe == "sparse_information":
        assert spec.data is not None
        raw = spec.data.read_bytes()
        data_path = spec.data.with_suffix(".npz")
        digest = sha256_bytes(raw + data_path.read_bytes())
        header = f"% Derived from {_relative(spec.data)} and {_relative(data_path)} (combined sha256: {digest}).\n"
        return header + render_sparse_information(json.loads(raw), spec.data), digest
    raise ValueError(f"unknown figure recipe: {spec.recipe}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_publication_text(spec: FigureSpec, text: str) -> None:
    """Reject audit-only terminology from paper-facing generated fragments."""

    lowered = text.casefold()
    forbidden = [term for term in FORBIDDEN_PUBLICATION_TERMS if term in lowered]
    if forbidden:
        raise ValueError(
            f"generated publication figure {spec.id} contains forbidden audit terminology: {', '.join(forbidden)}"
        )


def render_figures(figures: tuple[FigureSpec, ...], *, check: bool = False) -> tuple[RenderedFigure, ...]:
    """Generate figures atomically, or verify that tracked outputs are current."""

    records: list[RenderedFigure] = []
    stale: list[str] = []
    for spec in figures:
        text, source_digest = render_text(spec)
        _validate_publication_text(spec, text)
        path = spec.output_path
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(spec.id)
        else:
            _atomic_write(path, text)
        records.append(
            RenderedFigure(
                id=spec.id,
                output=_relative(path),
                output_sha256=sha256_bytes(text.encode()),
                source=None if spec.source is None else _relative(spec.source),
                source_sha256=source_digest if spec.source is not None else None,
                data=None if spec.data is None else _relative(spec.data),
                data_sha256=source_digest if spec.data is not None else None,
            )
        )
    if stale:
        raise RuntimeError(f"generated figures are stale or missing: {', '.join(stale)}")
    return tuple(records)


def write_build_manifest(records: tuple[RenderedFigure, ...], *, check: bool = False) -> None:
    """Write or verify the deterministic build-provenance manifest."""

    payload = {
        "figures": [
            {
                "id": record.id,
                "output": record.output,
                "output_sha256": record.output_sha256,
                "source": record.source,
                "source_sha256": record.source_sha256,
                "data": record.data,
                "data_sha256": record.data_sha256,
            }
            for record in records
        ]
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if check:
        if not BUILD_MANIFEST.exists() or BUILD_MANIFEST.read_text() != text:
            raise RuntimeError("generated build manifest is stale or missing")
    else:
        _atomic_write(BUILD_MANIFEST, text)
