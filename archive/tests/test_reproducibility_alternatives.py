from __future__ import annotations

from reproducibility.alternatives.render import (
    render_lobeke_relation,
    render_structured_whiskers,
    render_tuning_spacious,
)


def test_all_figure_alternatives_render_as_tikz_fragments() -> None:
    alternatives = (
        render_lobeke_relation(),
        render_tuning_spacious(),
        render_structured_whiskers(),
    )
    for rendered in alternatives:
        assert "\\begin{tikzpicture}" in rendered
        assert "\\end{tikzpicture}" in rendered
        assert "\\documentclass" not in rendered
        assert "\\caption" not in rendered


def test_alternatives_expose_the_intended_scientific_comparisons() -> None:
    assert "Target Fisher--SubOpt relation" in render_lobeke_relation()
    assert "Held-out confirmation" in render_tuning_spacious()
    assert "Follower response sharpness" in render_structured_whiskers()
