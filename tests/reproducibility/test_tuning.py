from __future__ import annotations

import json

import numpy as np
import pytest

from mairl_exp.execution.artifacts import sha256_file
from reproducibility.figures.recipes.lobeke_tuning import render_lobeke_tuning
from reproducibility.figures.render import render_text


def test_sensitivity_plot_preserves_all_seed_task_means(figure_spec):
    spec = figure_spec("i1_hyperparameters")
    metadata = json.loads(spec.data.read_text())
    path = spec.data.with_name(metadata["data_file"])
    assert sha256_file(path) == metadata["data_sha256"]
    with np.load(path, allow_pickle=False) as data:
        assert data["transfer_suboptimality"].shape == (5, 7, 32, 100)
        np.testing.assert_array_equal(data["seeds"], np.arange(2026094000, 2026094032))
        assert data["mle_ridge"] == 1e-6
        expected = data["transfer_suboptimality"].mean(axis=-1).mean(axis=-1) * 1e4
    rendered, _ = render_text(spec)
    for (row, column), value in np.ndenumerate(expected):
        assert f"(axis cs:{column},{row}) {{{value:.2f}}}" in rendered


@pytest.mark.parametrize("failure", ["missing_task", "nan", "negative", "duplicate_multiplier"])
def test_sensitivity_rejects_invalid_data(figure_spec, tmp_path, failure):
    spec = figure_spec("i1_hyperparameters")
    with np.load(spec.data.with_name("data.npz")) as source:
        data = dict(source)
    if failure == "missing_task":
        data["transfer_suboptimality"] = data["transfer_suboptimality"][..., :-1]
    elif failure == "duplicate_multiplier":
        data["multipliers"][1] = data["multipliers"][0]
    else:
        data["transfer_suboptimality"][0, 0, 0, 0] = np.nan if failure == "nan" else -1
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError):
        render_lobeke_tuning(path)
