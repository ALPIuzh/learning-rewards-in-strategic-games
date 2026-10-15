from __future__ import annotations

import numpy as np

from experiments.guard.paper.figures.full_grid_tuning import _paired_equivalence


def test_paired_equivalence_accepts_difference_within_three_standard_errors() -> None:
    reference = np.zeros(20)
    duplicate = np.asarray([-3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3], dtype=np.float64)

    result = _paired_equivalence(reference, duplicate)

    assert result["equivalent"] is True
    assert result["seeds"] == 20


def test_paired_equivalence_rejects_systematic_shift() -> None:
    reference = np.zeros(20)
    duplicate = np.linspace(0.9, 1.1, 20)

    result = _paired_equivalence(reference, duplicate)

    assert result["equivalent"] is False
