from __future__ import annotations

import numpy as np
import pytest

from experiments.toy.final_export import (
    CANONICAL_APPLICATION,
    validate_canonical_semantic_config,
    validate_protocol_fields,
    validate_trace_arrays,
)
from mairl_exp.batching import evaluation_checkpoints


def _valid_arrays(total_samples: int, checkpoints: np.ndarray) -> dict[str, np.ndarray]:
    policies = np.zeros((total_samples, 3), dtype=np.float64)
    policies[:, 0] = 1.0
    trace_counts = np.arange(int(checkpoints[0]), total_samples + 1)
    return {
        "trace_sample_counts": trace_counts,
        "evaluation_sample_counts": checkpoints,
        "policies": policies,
        "follower_actions": np.arange(total_samples, dtype=np.int64) % 4,
        "omega_history": np.zeros((trace_counts.size, 3), dtype=np.float64),
        "diagnostic__mle_residual": np.asarray([1.0e-8]),
    }


def test_evaluation_checkpoints_include_both_phase_boundary_and_final_budget() -> None:
    np.testing.assert_array_equal(
        evaluation_checkpoints(24, 100, 25),
        np.asarray([24, 49, 74, 99, 100]),
    )


def test_semantic_protocol_validation_fails_on_a_false_regularization_claim() -> None:
    with pytest.raises(ValueError, match="protected protocol"):
        validate_protocol_fields(
            {"lambda_information_multiplier": 3.0},
            {"lambda_information_multiplier": 1.0},
            label="experiment",
        )


def test_canonical_semantic_validation_checks_game_and_mle() -> None:
    semantic = {
        "application": {"_target_": "unused", **CANONICAL_APPLICATION},
        "numerics": {
            "mle": {
                "_target_": "unused",
                "solver": "newton",
                "ridge": 1.0e-4,
                "projected_gradient_tolerance": 1.0e-6,
            }
        },
    }
    validate_canonical_semantic_config(semantic)
    semantic["application"]["eta"] = 2.0
    with pytest.raises(ValueError, match="structured-toy application"):
        validate_canonical_semantic_config(semantic)
    np.testing.assert_array_equal(
        evaluation_checkpoints(0, 100, 25),
        np.asarray([0, 25, 50, 75, 100]),
    )


def test_trace_validation_accepts_complete_scientific_invariants() -> None:
    checkpoints = evaluation_checkpoints(2, 10, 4)
    arrays = _valid_arrays(10, checkpoints)
    assert validate_trace_arrays(arrays, total_samples=10, checkpoints=checkpoints) == 1.0e-8


def test_trace_validation_defers_certificate_only_for_full_warmup() -> None:
    checkpoints = np.asarray([10])
    arrays = _valid_arrays(10, checkpoints)
    arrays["diagnostic__mle_residual"] = np.empty(0)
    assert validate_trace_arrays(arrays, total_samples=10, checkpoints=checkpoints) is None

    checkpoints = evaluation_checkpoints(2, 10, 4)
    arrays = _valid_arrays(10, checkpoints)
    arrays["diagnostic__mle_residual"] = np.empty(0)
    with pytest.raises(ValueError, match="unexpectedly empty"):
        validate_trace_arrays(arrays, total_samples=10, checkpoints=checkpoints)


@pytest.mark.parametrize("failure", ["simplex", "action", "ball", "residual", "nonfinite"])
def test_trace_validation_fails_closed(failure: str) -> None:
    checkpoints = evaluation_checkpoints(2, 10, 4)
    arrays = _valid_arrays(10, checkpoints)
    if failure == "simplex":
        arrays["policies"][0, 0] = 0.5
    elif failure == "action":
        arrays["follower_actions"][0] = 4
    elif failure == "ball":
        arrays["omega_history"][0, 0] = 2.1
    elif failure == "residual":
        arrays["diagnostic__mle_residual"][0] = 1.1e-6
    else:
        arrays["omega_history"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_trace_arrays(arrays, total_samples=10, checkpoints=checkpoints)
