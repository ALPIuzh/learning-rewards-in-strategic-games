"""Shared fail-closed validation for protected structured-toy exports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

CANONICAL_APPLICATION = {
    "problem_seed": 17,
    "eta": 1.25,
    "radius": 2.0,
    "true_omega": [0.55, -0.65, 0.35],
    "leader_reward": [
        [0.1, 0.9, 0.3, -0.2],
        [0.7, -0.1, 0.5, 0.2],
        [-0.2, 0.4, 0.8, 0.6],
    ],
}


def validate_protocol_fields(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Require exact values for every scientifically meaningful protocol field."""

    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} differs from the protected protocol: {mismatches}")


def validate_canonical_semantic_config(semantic: Mapping[str, object]) -> None:
    """Validate the shared game and estimator settings of protected toy studies."""

    application = semantic.get("application")
    numerics = semantic.get("numerics")
    if not isinstance(application, Mapping) or not isinstance(numerics, Mapping):
        raise ValueError("semantic configuration is missing application or numerics")
    validate_protocol_fields(
        application,
        CANONICAL_APPLICATION,
        label="structured-toy application",
    )
    mle = numerics.get("mle")
    if not isinstance(mle, Mapping):
        raise ValueError("semantic configuration is missing MLE numerics")
    validate_protocol_fields(
        mle,
        {
            "solver": "newton",
            "ridge": 1.0e-4,
            "projected_gradient_tolerance": 1.0e-6,
        },
        label="structured-toy MLE numerics",
    )


def validate_trace_arrays(
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    total_samples: int,
    checkpoints: NDArray[np.int64],
    leader_action_count: int = 3,
    follower_action_count: int = 4,
    coefficient_dimension: int = 3,
    coefficient_radius: float = 2.0,
    mle_tolerance: float = 1.0e-6,
) -> float | None:
    """Validate common invariants and return a stored online MLE residual, if any.

    A run whose warm-up consumes the entire budget has no online diagnostic
    array. Its endpoint must be certified separately from the retained data.
    """

    for name in arrays:
        values = np.asarray(arrays[name])
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite values in {name}")

    trace_start = int(checkpoints[0])
    expected_trace_counts = np.arange(trace_start, total_samples + 1)
    trace_counts = np.asarray(arrays["trace_sample_counts"])
    if not np.array_equal(trace_counts, expected_trace_counts):
        raise ValueError("trace sample accounting is invalid")
    if not np.array_equal(arrays["evaluation_sample_counts"], checkpoints):
        raise ValueError("evaluation sample accounting is invalid")

    policies = np.asarray(arrays["policies"])
    if policies.shape != (total_samples, leader_action_count):
        raise ValueError("leader-policy trajectory has an invalid shape")
    if np.any(policies < -1.0e-12) or np.any(policies > 1.0 + 1.0e-12):
        raise ValueError("leader-policy trajectory leaves the simplex")
    if not np.allclose(policies.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("leader-policy trajectory is not normalized")

    follower_actions = np.asarray(arrays["follower_actions"])
    if follower_actions.shape != (total_samples,) or not np.issubdtype(
        follower_actions.dtype, np.integer
    ):
        raise ValueError("follower-action trajectory has an invalid shape or dtype")
    if np.any(follower_actions < 0) or np.any(follower_actions >= follower_action_count):
        raise ValueError("follower-action trajectory contains an invalid action")

    omega_history = np.asarray(arrays["omega_history"])
    if omega_history.shape != (expected_trace_counts.size, coefficient_dimension):
        raise ValueError("coefficient trajectory has an invalid shape")
    if np.any(np.linalg.norm(omega_history, axis=1) > coefficient_radius + 1.0e-9):
        raise ValueError("coefficient trajectory leaves the configured ball")

    residuals = np.asarray(arrays["diagnostic__mle_residual"])
    if residuals.ndim != 1 or np.any(residuals < 0):
        raise ValueError("MLE residual diagnostics are invalid")
    if residuals.size == 0:
        if trace_start != total_samples:
            raise ValueError("MLE residual diagnostics are unexpectedly empty")
        return None
    maximum_residual = float(np.max(residuals))
    if maximum_residual > mle_tolerance:
        raise ValueError("an MLE residual exceeds the configured tolerance")
    return maximum_residual
