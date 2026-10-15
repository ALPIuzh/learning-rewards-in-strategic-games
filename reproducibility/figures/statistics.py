"""Statistical summaries shared by publication figure recipes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mairl_exp.evaluation.statistics import mean_confidence_interval

FloatArray = NDArray[np.float64]


def mean_interval(values: FloatArray, confidence_level: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Columnwise mean and two-sided Student-t confidence interval across seeds.

    Figure recipes validate their inputs strictly: a malformed or non-finite array
    is a broken pipeline, not a band to draw.
    """

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must have shape (seeds, checkpoints) with at least one seed")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or infinite entries")
    return mean_confidence_interval(values, confidence_level)
