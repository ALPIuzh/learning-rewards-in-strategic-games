from __future__ import annotations

import json

from reproducibility.figures.manifest import load_manifest
from reproducibility.figures.render import render_text


def test_sparse_information_figure_uses_protected_action_count_sweep() -> None:
    figure = next(figure for figure in load_manifest().figures if figure.id == "i2_sparse_information")
    _, digest = render_text(figure)
    assert digest is not None

    payload = json.loads(figure.data.read_text())
    assert payload["run_id"] == "219b99056804830c"
    assert payload["seed_policy"] == {"paired": True, "seeds": list(range(72_000, 72_100)), "count": 100}
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
