from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from mairl_exp.execution.artifacts import sha256_file
from reproducibility.figures.recipes.lobeke_tuning import render_lobeke_tuning
from reproducibility.figures.render import render_text


def test_lobeke_tuning_figure_uses_ridge_zero_discrete_grid(figure_spec) -> None:
    tuning = figure_spec("i1_hyperparameters")
    _, digest = render_text(tuning)
    payload = json.loads(tuning.data.read_text())
    assert digest is not None
    assert payload["schema_version"] == 3
    assert payload["status"] == "complete"
    assert payload["sample_count"] == 1600
    assert payload["mle_ridge"] == 0.0
    assert payload["seed_policy"] == {
        "count": 8,
        "disjoint_from_final_evaluation": True,
        "paired_across_cells": True,
        "start": 94000,
    }
    assert payload["task_policy"]["count"] == 100
    assert payload["task_policy"]["seed"] == 9400
    assert payload["task_policy"]["candidate_pool"] == 2000
    assert payload["available_candidates"] == payload["planned_candidates"] == 15
    assert payload["missing_candidates"] == []
    data_path = tuning.data.with_name(payload["data_file"])
    assert sha256_file(data_path) == payload["data_sha256"]
    with np.load(data_path, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["seeds"], np.arange(94_000, 94_008))
        assert arrays["per_seed_downstream_suboptimality"].shape == (15, 8)
        assert np.all(np.isfinite(arrays["per_seed_downstream_suboptimality"]))
        assert np.array_equal(arrays["heldout_seeds"], np.arange(95_000, 95_020))
        assert arrays["heldout_protocols"].shape == (5,)
        assert arrays["heldout_per_seed_downstream_suboptimality"].shape == (5, 20)
        assert np.all(np.isfinite(arrays["heldout_per_seed_downstream_suboptimality"]))
        by_key = {
            (int(budget), float(multiplier)): values
            for budget, multiplier, values in zip(
                arrays["candidate_warmup_budgets"],
                arrays["candidate_lambda_information_multipliers"],
                arrays["per_seed_downstream_suboptimality"],
                strict=True,
            )
        }
    for candidate in payload["candidates"]:
        key = (candidate["warmup_budget"], candidate["lambda_information_multiplier"])
        values = np.asarray(candidate["downstream_suboptimality"]["per_seed"])
        assert np.array_equal(values, by_key[key])
        assert np.isclose(candidate["downstream_suboptimality"]["mean"], np.mean(values))
        difference = candidate["paired_difference_from_frozen"]
        reference = difference["reference"]
        reference_values = by_key[reference["warmup_budget"], reference["lambda_information_multiplier"]]
        paired = values - reference_values
        assert np.isclose(difference["mean"], np.mean(paired))
        assert difference["contains_zero"] == (difference["ci95_lower"] <= 0.0 <= difference["ci95_upper"])

    heldout = payload["heldout_confirmation"]
    assert heldout["seed_policy"] == {
        "count": 20,
        "disjoint_from_sensitivity_and_final_evaluation": True,
        "paired_across_protocols": True,
        "start": 95000,
    }
    assert heldout["task_policy"]["count"] == 100
    assert heldout["task_policy"]["seed"] == 9500


def test_lobeke_tuning_omits_missing_cells_without_placeholder_marks(figure_spec, figure_payload) -> None:
    payload = figure_payload("i1_hyperparameters")
    partial = copy.deepcopy(payload)
    removed = partial["candidates"].pop(6)
    partial["status"] = "partial"
    partial["available_candidates"] -= 1
    partial["missing_candidates"] = [[removed["warmup_budget"], removed["lambda_information_multiplier"]]]
    rendered = render_lobeke_tuning(partial)
    assert rendered.count("\\path[draw=white") == 14
    assert "pending" not in rendered
    assert "$\\times$" not in rendered


def test_lobeke_tuning_rejects_incomplete_schema_three_uncertainty(figure_spec, figure_payload) -> None:
    payload = figure_payload("i1_hyperparameters")

    missing_frozen = copy.deepcopy(payload)
    missing_frozen["frozen_protocols"].pop()
    with pytest.raises(ValueError, match="exactly two frozen protocols"):
        render_lobeke_tuning(missing_frozen)

    missing_interval = copy.deepcopy(payload)
    missing_interval["candidates"][0].pop("paired_difference_from_frozen")
    with pytest.raises(ValueError, match="no paired interval"):
        render_lobeke_tuning(missing_interval)


@pytest.mark.parametrize(
    "multipliers",
    (
        (0.0, 0.01, 0.03, 0.3, 1.0),
        (-0.003, 0.01, 0.03, 0.3, 1.0),
        (0.003, 0.003, 0.03, 0.3, 1.0),
        (0.01, 0.003, 0.03, 0.3, 1.0),
    ),
)
def test_lobeke_tuning_rejects_invalid_multiplier_grid(
    multipliers: tuple[float, ...], figure_spec, figure_payload
) -> None:
    payload = figure_payload("i1_hyperparameters")
    payload["lambda_information_multipliers"] = list(multipliers)
    with pytest.raises(ValueError, match="unique, positive, and increasing"):
        render_lobeke_tuning(payload)
