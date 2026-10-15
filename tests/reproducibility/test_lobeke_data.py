from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.guard.paper.figures.common import _array_sha256
from experiments.guard.paper.figures.data import METHOD_SLUGS
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.execution.artifacts import sha256_file


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
    assert {method for run in source["runs"] for method in run["methods"]} == {key[0] for key in METHOD_SLUGS}
    active = provenance["protocol"]["active"]
    assert {
        key: active[key]
        for key in ("warmup_budget", "warm_lambda_information_multiplier", "no_warm_lambda_information_multiplier")
    } == {"warmup_budget": 50, "warm_lambda_information_multiplier": 0.03, "no_warm_lambda_information_multiplier": 0.1}
    with np.load(REPOSITORY_ROOT / "reproducibility/data/lobeke/sensitivity_32seeds/data.npz") as grid:
        scores = grid["transfer_suboptimality"].mean(axis=(2, 3))
        warm_rows = np.flatnonzero(grid["warmup_budgets"] > 0)
        row, column = np.unravel_index(np.argmin(scores[warm_rows]), scores[warm_rows].shape)
        assert active["warmup_budget"] == grid["warmup_budgets"][warm_rows[row]]
        assert active["warm_lambda_information_multiplier"] == grid["multipliers"][column]
        zero_row = np.flatnonzero(grid["warmup_budgets"] == 0).item()
        assert active["no_warm_lambda_information_multiplier"] == grid["multipliers"][scores[zero_row].argmin()]
        assert not set(grid["seeds"]) & set(provenance["seed_policy"]["seeds"])
        assert provenance["transfer_task_policy"]["seed"] != int(grid["task_seed"])
        assert provenance["protocol"]["fixed_expert_mle"]["ridge"] == float(grid["mle_ridge"])
    assert provenance["protocol"]["fixed_expert_mle"]["ridge"] == 1e-6
    assert provenance["protocol"]["ols"]["zero_sum_sampling"] == ("paired empirical leader and follower actions")
    assert provenance["seed_policy"] == {"paired": True, "seeds": list(range(2026097000, 2026097050)), "count": 50}
    assert sha256_file(data_path) == provenance["data_sha256"]

    with np.load(data_path, allow_pickle=False) as data:
        assert len(data.files) == 99
        assert set(data.files) == set(provenance["arrays"])
        assert data["active_warm__taskwise_transfer_suboptimality"].shape == (
            10,
            50,
            len(data["active_warm__checkpoints"]),
        )


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
    } == {"name": "fresh_behavioral", "count": 180, "seed": 202609701, "candidate_pool": 180}
    assert provenance["evaluation"]["seed_count"] == 50
    assert provenance["evaluation"]["checkpoints"] == [200, 400, 800, 1200, 1600]
    assert set(provenance["evaluation"]["methods"]) == set(METHOD_SLUGS.values())
    assert sha256_file(data_path) == provenance["data_sha256"]

    with np.load(data_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(provenance["arrays"])
        assert arrays["tasks"].shape == (180, 23, 23)
        assert arrays["oracle_policies"].shape == (180, 23)
        assert np.array_equal(arrays["seeds"], np.arange(2026097000, 2026097050))
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


def test_lobeke_environment_has_verified_data_and_probability_maps(figure_spec) -> None:
    figure = figure_spec("i1_environment")
    assert figure.data is not None
    provenance = json.loads(figure.data.read_text())
    data_path = figure.data.with_name("data.npz")
    assert sha256_file(data_path) == provenance["data_sha256"]
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
