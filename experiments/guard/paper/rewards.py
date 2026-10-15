"""Canonical evaluation units for Lobeke leader rewards."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError

UNIT_INTERVAL_DEFINITION = "(r - min(r)) / (max(r) - min(r))"
UNIT_INTERVAL_R_MAX = 1.0


def _validated_reward(reward: ArrayLike) -> tuple[FloatArray, float, float, float]:
    array = np.asarray(reward, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ConfigurationError("leader reward must be a finite matrix")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0.0:
        raise ConfigurationError("leader reward must have strictly positive range")
    return array, minimum, maximum, span


def unit_interval_leader_reward(reward: ArrayLike) -> FloatArray:
    """Return the positive-affine representative with entries in ``[0, 1]``.

    Stackelberg policies and follower behavior are invariant to a global shift
    of the leader reward, while SubOpt scales linearly under a positive reward
    rescaling.  This canonical representative therefore preserves every policy
    comparison and gives all reported SubOpt values the common convention
    ``R_max = ||r^1||_infinity = 1``.
    """

    array, minimum, _maximum, span = _validated_reward(reward)
    normalized = (array - minimum) / span
    normalized[np.abs(normalized) < 1.0e-15] = 0.0
    normalized[np.abs(normalized - 1.0) < 1.0e-15] = 1.0
    return normalized


def leader_reward_normalization_metadata(reward: ArrayLike) -> dict[str, float | str]:
    """Describe the canonical source-reward transformation for provenance."""

    _array, minimum, maximum, span = _validated_reward(reward)
    return {
        "definition": UNIT_INTERVAL_DEFINITION,
        "raw_minimum": minimum,
        "raw_maximum": maximum,
        "raw_range": span,
        "normalized_r_max": UNIT_INTERVAL_R_MAX,
    }


def is_unit_interval_reward_metadata(metadata: Mapping[str, object]) -> bool:
    """Return whether metadata identifies the canonical unit-interval convention."""

    return (
        metadata.get("definition") == UNIT_INTERVAL_DEFINITION
        and metadata.get("normalized_r_max") == UNIT_INTERVAL_R_MAX
    )


def evaluation_reward_normalization_provenance(source_metadata: Mapping[str, object]) -> dict[str, object]:
    """Build the shared source-and-transfer normalization provenance block."""

    if not is_unit_interval_reward_metadata(source_metadata):
        raise ValueError("source reward does not use the canonical unit-interval convention")
    return {
        "definition": UNIT_INTERVAL_DEFINITION,
        "source": dict(source_metadata),
        "transfer_r_max": UNIT_INTERVAL_R_MAX,
        "policy_invariant": True,
    }


def validate_evaluation_reward_normalization(metadata: Mapping[str, object]) -> None:
    """Reject publication provenance that does not use canonical reward units."""

    source = metadata.get("source")
    if (
        metadata.get("definition") != UNIT_INTERVAL_DEFINITION
        or metadata.get("policy_invariant") is not True
        or metadata.get("transfer_r_max") != UNIT_INTERVAL_R_MAX
        or not isinstance(source, Mapping)
        or not is_unit_interval_reward_metadata(source)
    ):
        raise ValueError("evaluation rewards do not use the canonical unit-interval convention")


def suboptimality_in_unit_reward(values: ArrayLike,
        *, raw_reward_range: float, already_unit_reward: bool) -> FloatArray:  # fmt: skip
    """Convert ordinary SubOpt values once to the canonical reward units."""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("SubOpt values must be finite")
    if not np.isfinite(raw_reward_range) or raw_reward_range <= 0.0:
        raise ValueError("raw_reward_range must be finite and strictly positive")
    return array.copy() if already_unit_reward else array / raw_reward_range
