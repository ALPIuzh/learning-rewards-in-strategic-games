from __future__ import annotations

import json

from reproducibility.figures.render import render_text


def test_lower_bound_figure_uses_complete_theorem_valid_grid(figure_spec) -> None:
    figure = figure_spec("i3_lower_bound")
    _, digest = render_text(figure)
    assert digest is not None

    payload = json.loads(figure.data.read_text())
    for slope in payload["subopt_tail_slopes"].values():
        assert slope["confidence_interval"][0] <= -0.5 <= slope["confidence_interval"][1]
