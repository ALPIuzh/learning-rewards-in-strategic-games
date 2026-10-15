"""Purpose-separated random streams for reproducible experiment comparisons."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class RandomPurpose(IntEnum):
    """Stable integer keys for independent experiment random streams."""

    WARMUP_FOLLOWER = 1
    ACTIVE_POLICY = 2
    ONLINE_FOLLOWER = 3
    FIXED_RANDOM_POLICY = 4
    ONLINE_RANDOM_POLICY = 5
    EXPERT_FOLLOWER = 6
    EXPERT_LEADER = 7


def experiment_rng(seed: int, purpose: RandomPurpose, sample_index: int = 0) -> np.random.Generator:
    """Return a stream keyed by seed, purpose, and absolute sample index.

    Keying online randomness by the absolute sample index makes paired protocol
    comparisons insensitive to whether an earlier warm-up phase consumed random
    numbers.  The integer purpose values are explicit so results do not depend on
    Python's process-randomized string hashing.
    """

    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    sequence = np.random.SeedSequence((int(seed), int(purpose), int(sample_index)))
    return np.random.default_rng(sequence)
