"""Seed-level summary statistics shared by plots, exports and figure recipes."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from mairl.core.problem import FloatArray


def mean_confidence_interval(values: FloatArray, confidence_level: float,
        *, clip_lower: float | None = None) -> tuple[FloatArray, FloatArray, FloatArray]:  # fmt: skip
    """Return mean and a two-sided Student-t interval across the leading seed axis.

    Fewer than two seeds yields a zero-width band at the mean.
    ``clip_lower`` floors the lower edge for
    quantities that cannot be negative; leaving it unset reports the raw edge.
    """

    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=0)
    seed_count = array.shape[0]
    if seed_count < 2:
        return mean, np.copy(mean), np.copy(mean)
    standard_error = np.std(array, axis=0, ddof=1) / np.sqrt(seed_count)
    critical = float(student_t.ppf(0.5 + confidence_level / 2.0, df=seed_count - 1))
    half_width = critical * standard_error
    lower = mean - half_width
    if clip_lower is not None:
        lower = np.maximum(clip_lower, lower)
    return mean, lower, mean + half_width
