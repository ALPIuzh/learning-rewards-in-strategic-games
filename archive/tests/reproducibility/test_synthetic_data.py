from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reproducibility.figures.manifest import (
    FigureSpec,
    load_manifest,
)
from reproducibility.figures.render import render_text


def test_sparse_information_figure_uses_protected_action_count_sweep() -> None:
    figure = next(
        figure for figure in load_manifest().figures if figure.id == "i2_sparse_information"
    )
    _, digest = render_text(figure)
    assert digest is not None

    payload = json.loads(figure.data.read_text())
    assert payload["run_id"] == "219b99056804830c"
    assert payload["seed_policy"] == {
        "paired": True,
        "seeds": list(range(72_000, 72_100)),
        "count": 100,
    }
    assert payload["analytical_ratio_validated"] is True


def test_lower_bound_figure_uses_complete_theorem_valid_grid() -> None:
    figure = next(figure for figure in load_manifest().figures if figure.id == "i3_lower_bound")
    _, digest = render_text(figure)
    assert digest is not None

    payload = json.loads(figure.data.read_text())
    assert payload["source"]["run_id"] == "0ce1a906d05aa86d"
    assert payload["source"]["result_count"] == 160
    assert payload["grid"]["dimensions"] == [33, 64]
    for slope in payload["subopt_tail_slopes"].values():
        assert slope["confidence_interval"][0] <= -0.5 <= slope["confidence_interval"][1]


def _structured_sensitivity_inputs() -> tuple[FigureSpec, Path, Path]:
    figure = next(
        figure for figure in load_manifest().figures if figure.id == "i2_structured_sensitivity"
    )
    assert figure.data is not None
    descriptor = json.loads(figure.data.read_text())
    return (
        figure,
        figure.data.with_name(descriptor["warmup"]),
        figure.data.with_name(descriptor["temperature"]),
    )


def test_structured_warmup_uses_complete_protected_ablation() -> None:
    figure, warmup_path, _temperature_path = _structured_sensitivity_inputs()
    _, digest = render_text(figure)
    assert digest is not None

    payload = json.loads(warmup_path.read_text())
    assert payload["run_id"] == "232ed798be85db15"
    assert payload["warmup_budgets"] == [0, 24, 60, 120, 240, 500, 800, 1000]
    assert payload["maximum_mle_residual"] <= 1.0e-6
    assert payload["seed_policy"] == {
        "paired": True,
        "seeds": list(range(82_000, 82_100)),
        "count": 100,
    }
    with np.load(warmup_path.with_suffix(".npz"), allow_pickle=False) as arrays:
        for metric in (
            "stackelberg_suboptimality",
            "fisher_transfer_error",
            "coefficient_l2_error",
        ):
            values = arrays[f"final__{metric}"]
            assert values.shape == (8, 3, 100)
            assert np.all(np.isfinite(values))
            np.testing.assert_array_equal(
                values[-1],
                np.repeat(values[-1, 0][None, :], 3, axis=0),
            )


def test_structured_temperature_uses_complete_protected_sweep() -> None:
    figure, _warmup_path, temperature_path = _structured_sensitivity_inputs()
    _, digest = render_text(figure)
    assert digest is not None

    payload = json.loads(temperature_path.read_text())
    assert payload["run_id"] == "3d275636ae19ce64"
    assert payload["eta_values"] == [0.5, 0.75, 1.25, 2.0, 3.0]
    assert payload["lambda_information_rule"] == "lambda = eta^2"
    assert payload["warmup_budget"] == 120
    assert payload["total_samples"] == 1000
    assert payload["maximum_mle_residual"] <= 1.0e-6
    assert payload["seed_policy"] == {
        "paired": True,
        "seeds": list(range(83_000, 83_100)),
        "count": 100,
    }
    with np.load(temperature_path.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert arrays["evaluation_sample_counts"].shape == (37,)
        assert arrays["evaluation_sample_counts"][0] == 120
        assert arrays["evaluation_sample_counts"][-1] == 1000
        for metric in (
            "stackelberg_suboptimality",
            "fisher_transfer_error",
            "coefficient_l2_error",
        ):
            finals = arrays[f"final__{metric}"]
            assert finals.shape == (5, 3, 100)
            assert np.all(np.isfinite(finals))
        assert not any(name.startswith("trace__") for name in arrays.files)
    assert payload["compact_export"] == {
        "aggregation": "per-seed final values",
        "metrics": [
            "stackelberg_suboptimality",
            "fisher_transfer_error",
            "coefficient_l2_error",
        ],
        "raw_traces_retained_locally": True,
    }
