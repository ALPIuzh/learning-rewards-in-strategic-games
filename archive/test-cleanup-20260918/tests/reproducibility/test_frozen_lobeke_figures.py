"""Frozen numerical inputs must rebuild figures without pre-rendered sources."""

import json
import shutil

import pytest

from reproducibility.figures import manifest, render
from reproducibility.figures.recipes.lobeke_frozen import render_source, render_transfer

ROOT = manifest.REPOSITORY_ROOT
SOURCE = ROOT / "reproducibility/data/lobeke/source_convergence_unit_reward_v2/provenance.json"
TRANSFER = ROOT / "reproducibility/data/lobeke/downstream_figures.json"


def test_all_figures_rebuild_with_generated_trees_absent(tmp_path, monkeypatch):
    # A minimal copy excludes both historical producer snippets and current outputs.
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "reproducibility/data", root / "reproducibility/data")
    shutil.copyfile(ROOT / "reproducibility/figures.toml", root / "reproducibility/figures.toml")
    expected = {spec.id: render.render_text(spec)[0] for spec in manifest.load_manifest().figures}
    generated = root / "reproducibility/tex/generated"
    monkeypatch.setattr(manifest, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(manifest, "GENERATED_ROOT", generated)
    monkeypatch.setattr(render, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(render, "BUILD_MANIFEST", generated / "build_manifest.json")
    figures = manifest.load_manifest(root / "reproducibility/figures.toml").figures
    assert all(spec.recipe != "copy" for spec in figures)
    render.write_build_manifest(render.render_figures(figures))
    for spec in figures:
        assert spec.output_path.read_text() == expected[spec.id]
    render.write_build_manifest(render.render_figures(figures, check=True), check=True)


def test_source_export_rejects_tampered_data(tmp_path):
    shutil.copytree(SOURCE.parent, tmp_path / "source")
    data = tmp_path / "source/data.npz"
    data.write_bytes(data.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        render_source(tmp_path / "source/provenance.json")


@pytest.mark.parametrize("name", ["summary", "policy_grids", "environment"])
def test_transfer_rejects_tampered_input(tmp_path, name):
    descriptor = json.loads(TRANSFER.read_text())
    for key, record in descriptor["inputs"].items():
        destination = tmp_path / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / record["path"]).read_bytes())
        if key == name:
            destination.write_bytes(destination.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        render_transfer(TRANSFER, "lobeke_transfer", tmp_path)


def test_full_resolution_export_retains_seeds_and_checkpoints():
    provenance = json.loads(SOURCE.read_text())
    assert {source["run_id"] for source in provenance["sources"]} == {"349676e5922332fd", "cf7b0f494e850f2d"}
    assert sum(source["result_count"] for source in provenance["sources"]) == 450
    for protocol in provenance["protocols"]:
        assert len(protocol["seeds"]) == 50
        assert protocol["checkpoints"][-1] == 1600
        assert set(b - a for a, b in zip(protocol["checkpoints"][:-1], protocol["checkpoints"][1:], strict=True)) == {
            40
        }


def test_source_export_preserves_existing_frozen_data(tmp_path):
    from experiments.guard.paper.figures.export_source_convergence import export

    existing = tmp_path / "data.npz"
    existing.write_bytes(b"protected")
    with pytest.raises(ValueError, match="output directory must be empty"):
        export([], tmp_path)
    assert existing.read_bytes() == b"protected"
