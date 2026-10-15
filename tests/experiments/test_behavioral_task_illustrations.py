from __future__ import annotations

import json

import numpy as np

from experiments.guard.paper.figures.task_sets import _quantile_representative_indices
from mairl_exp.config import REPOSITORY_ROOT

ASSET_DIR = REPOSITORY_ROOT / "reproducibility/data/lobeke/downstream_assets"
EXPORT_DIR = REPOSITORY_ROOT / "reproducibility/data/lobeke/final_evaluation_unit_reward_v2"


def test_behavioral_quantile_selection_is_deterministic_and_algorithm_blind() -> None:
    distances = np.asarray([1.0, 0.8, 0.2, 0.8, 0.4])

    assert _quantile_representative_indices(distances, (0.0, 0.5, 1.0)) == (2, 1, 0)


def test_behavioral_illustrations_match_the_fixed_transfer_suite() -> None:
    summary = json.loads((ASSET_DIR / "summary.json").read_text())
    selection = summary["task_selection"]

    assert summary["schema_version"] == 4
    assert selection["candidate_suite"] == {
        "candidate_pool": 500,
        "count": 10,
        "name": "fresh_primary",
        "scale": 1.0,
        "seed": 202609700,
    }
    assert selection["behavior_quantiles"] == [0.0, 0.5, 1.0]
    assert selection["selected_suite_indices_zero_based"] == [4, 5, 0]
    assert selection["uses_algorithm_performance"] is False

    source_distances = np.asarray(selection["source_policy_total_variation"])
    assert np.all(np.diff(source_distances) > 0.0)
    np.testing.assert_allclose(source_distances, [0.36631, 0.85677, 1.0], atol=5e-6)
    pairwise = np.asarray(selection["selected_pairwise_policy_total_variation"])
    np.testing.assert_array_equal(np.diag(pairwise), np.zeros(3))
    np.testing.assert_allclose(pairwise, pairwise.T, atol=1e-14)
    assert np.min(pairwise[np.triu_indices(3, k=1)]) > 0.92

    method_arrays = {
        "active": "active_warm",
        "active_nowarm": "active_no_warm",
        "expert": "fixed_expert_mle",
        "liao": "zero_sum_ols",
        "follower": "follower_ols",
        "fixed": "fixed_random_warm",
        "fixed_nowarm": "fixed_random_no_warm",
        "random": "online_random_warm",
        "random_nowarm": "online_random_no_warm",
    }
    selected = selection["selected_suite_indices_zero_based"]
    with np.load(EXPORT_DIR / "data.npz", allow_pickle=False) as exported:
        for method, array_slug in method_arrays.items():
            displayed = np.asarray([result["mean"] for result in summary["methods"][method]["tasks"]])
            protected = exported[f"{array_slug}__taskwise_transfer_suboptimality"][selected, :, -1].mean(axis=1)
            np.testing.assert_allclose(displayed, protected, rtol=0.0, atol=1e-14)
