from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.guard.paper.figures.export_final import METHOD_SLUGS, _array_sha256
from reproducibility.figures.manifest import (
    REPOSITORY_ROOT,
    load_manifest,
)
from reproducibility.figures.policy_grid import LAYOUT
from reproducibility.figures.render import render_text


def test_current_lobeke_export_is_clean_estimator_matched_and_complete() -> None:
    export_dir = REPOSITORY_ROOT / "reproducibility/data/lobeke/final_evaluation_unit_reward_v2"
    data_path = export_dir / "data.npz"
    provenance = json.loads((export_dir / "provenance.json").read_text())

    assert provenance["schema_version"] == 5
    source = provenance["source"]
    assert source["result_count"] == 450
    assert sum(run["result_count"] for run in source["runs"]) == 450
    assert {run["git_commit"] for run in source["runs"]} == {source["git_commit"]}
    assert all(run["git_dirty"] is False for run in source["runs"])
    assert {tuple(run["methods"]) for run in source["runs"]} == {
        ("active", "fixed_expert_mle", "fixed_random", "random_each_round"),
        ("follower_ols", "zero_sum_ols"),
    }
    assert provenance["protocol"]["active"] == {
        "warmup_budget": 200,
        "warm_lambda_information_multiplier": 0.003,
        "no_warm_lambda_information_multiplier": 0.03,
        "comparison": "separately tuned protocols, not a matched-lambda ablation",
    }
    assert provenance["protocol"]["fixed_expert_mle"]["ridge"] == 0.0
    assert provenance["protocol"]["ols"]["zero_sum_sampling"] == (
        "paired empirical leader and follower actions"
    )
    assert provenance["seed_policy"] == {
        "paired": True,
        "seeds": list(range(30_000, 30_050)),
        "count": 50,
    }
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == provenance["data_sha256"]

    with np.load(data_path, allow_pickle=False) as data:
        assert len(data.files) == 99
        assert set(data.files) == set(provenance["arrays"])
        assert data["active_warm__taskwise_transfer_suboptimality"].shape == (10, 50, 10)


def test_lobeke_warmup_factorial_is_paired_and_recomputable() -> None:
    export_dir = REPOSITORY_ROOT / "reproducibility/data/lobeke/warmup_factorial_unit_reward"
    provenance = json.loads((export_dir / "provenance.json").read_text())
    data_path = export_dir / "data.npz"

    assert provenance["design"]["paired_seeds"] == list(range(30_000, 30_050))
    assert provenance["design"]["warmup_budgets"] == [0, 200]
    assert provenance["design"]["lambda_information_multipliers"] == [3, 12]
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == provenance["data_sha256"]

    with np.load(data_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(provenance["arrays"])
        assert np.array_equal(arrays["seeds"], np.arange(30_000, 30_050))
        for name in arrays.files:
            assert list(arrays[name].shape) == provenance["arrays"][name]["shape"]
            assert str(arrays[name].dtype) == provenance["arrays"][name]["dtype"]
            assert np.all(np.isfinite(arrays[name]))
        for multiplier in (3, 12):
            effect = (
                arrays[f"warm_lambda{multiplier}__target_suboptimality"]
                - arrays[f"no_warm_lambda{multiplier}__target_suboptimality"]
            )
            recorded = provenance["effects"][f"warm_minus_no_warm_at_lambda{multiplier}"][
                "target_suboptimality"
            ]
            assert recorded["mean"] == pytest.approx(float(effect.mean()), abs=1e-15)


def test_behavioral_transfer_gap_heatmap_covers_every_method() -> None:
    manifest = load_manifest()
    figures = {
        figure.variant: figure
        for figure in manifest.figures
        if figure.recipe == "lobeke_behavioral_transfer"
    }

    assert set(figures) == {"gap"}
    for metric, figure in figures.items():
        rendered, digest = render_text(figure)
        assert digest is not None
        assert "group size=3 by 3" in rendered
        assert rendered.count("\\nextgroupplot[") == 9
        provenance = json.loads(figure.data.read_text())
        counts = provenance["distance_metrics"]["bucket_counts"][metric]
        assert sum(counts) == 180
        assert len(counts) == 3


def test_behavioral_transfer_export_is_complete_finite_and_checksum_verified() -> None:
    export_dir = REPOSITORY_ROOT / "reproducibility/data/lobeke/behavioral_transfer_v1"
    data_path = export_dir / "data.npz"
    provenance = json.loads((export_dir / "provenance.json").read_text())

    assert provenance["schema_version"] == 1
    task_policy = provenance["task_policy"]
    assert {
        "name": task_policy["name"],
        "count": task_policy["count"],
        "seed": task_policy["seed"],
        "candidate_pool": task_policy["candidate_pool"],
    } == {
        "name": "behavioral_transfer_dense_v1",
        "count": 180,
        "seed": 9600,
        "candidate_pool": 180,
    }
    assert provenance["evaluation"]["seed_count"] == 50
    assert provenance["evaluation"]["checkpoints"] == [200, 400, 800, 1200, 1600]
    assert set(provenance["evaluation"]["methods"]) == set(METHOD_SLUGS.values())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == provenance["data_sha256"]

    with np.load(data_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(provenance["arrays"])
        assert arrays["tasks"].shape == (180, 23, 23)
        assert arrays["oracle_policies"].shape == (180, 23)
        assert np.array_equal(arrays["seeds"], np.arange(30_000, 30_050))
        for name in arrays.files:
            values = arrays[name]
            recorded = provenance["arrays"][name]
            assert list(values.shape) == recorded["shape"]
            assert str(values.dtype) == recorded["dtype"]
            assert _array_sha256(values) == recorded["sha256"]
            assert np.all(np.isfinite(values))
        for slug in METHOD_SLUGS.values():
            assert arrays[f"{slug}__taskwise_transfer_suboptimality"].shape == (180, 50, 5)
            assert arrays[f"{slug}__taskwise_transfer_fisher_error"].shape == (180, 50, 5)


def test_lobeke_environment_uses_the_shared_map_and_verified_data() -> None:
    figure = next(figure for figure in load_manifest().figures if figure.id == "i1_environment")
    rendered, digest = render_text(figure)
    assert digest is not None
    cell_token = f"x={LAYOUT.cell_cm:.3f}cm, y={LAYOUT.cell_cm:.3f}cm"
    assert rendered.count(cell_token) == 3
    assert figure.data is not None
    provenance = json.loads(figure.data.read_text())
    data_path = figure.data.with_name("data.npz")
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == provenance["data_sha256"]
    assert provenance["source"]["dependency"] == "GUARD"
    assert provenance["source"]["target_count"] == 23
    assert provenance["model"]["source_game_is_zero_sum"] is True
    with np.load(data_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"target_value", "patrol_coverage", "poacher_attack"}
        for values in arrays.values():
            assert values.shape == (5, 5)
            assert np.all(np.isfinite(values))
        assert arrays["patrol_coverage"].sum() == pytest.approx(1.0)
        assert arrays["poacher_attack"].sum() == pytest.approx(1.0)
