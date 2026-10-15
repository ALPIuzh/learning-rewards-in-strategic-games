from __future__ import annotations

import numpy as np
import pytest

from mairl.core.randomness import RandomPurpose, experiment_rng


def test_experiment_rng_is_reproducible_and_purpose_separated() -> None:
    expected = experiment_rng(17, RandomPurpose.ONLINE_FOLLOWER, 42).random(8)
    repeated = experiment_rng(17, RandomPurpose.ONLINE_FOLLOWER, 42).random(8)
    other_purpose = experiment_rng(17, RandomPurpose.ACTIVE_POLICY, 42).random(8)
    other_index = experiment_rng(17, RandomPurpose.ONLINE_FOLLOWER, 43).random(8)

    np.testing.assert_array_equal(expected, repeated)
    assert not np.array_equal(expected, other_purpose)
    assert not np.array_equal(expected, other_index)


def test_experiment_rng_rejects_negative_sample_index() -> None:
    with pytest.raises(ValueError, match="sample_index"):
        experiment_rng(17, RandomPurpose.ONLINE_FOLLOWER, -1)
