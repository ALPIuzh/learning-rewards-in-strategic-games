from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import CompiledProblem, compile_problem


@pytest.fixture
def small_features() -> np.ndarray:
    return np.array(
        [
            [[0.60, 0.10], [-0.20, 0.55], [0.15, -0.45]],
            [[-0.35, 0.25], [0.50, -0.20], [-0.10, 0.65]],
            [[0.20, -0.50], [0.10, 0.35], [0.55, 0.05]],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def small_problem(small_features: np.ndarray) -> CompiledProblem:
    return compile_problem(small_features, eta=1.2, radius=2.0)


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def small_true_omega() -> np.ndarray:
    """The reward coefficients the `small_problem` fixture is normally probed with."""

    return np.array([0.4, -0.7])


@pytest.fixture
def small_oracle(small_problem: CompiledProblem, small_true_omega: np.ndarray) -> SoftmaxFollowerOracle:
    return SoftmaxFollowerOracle(small_problem, small_true_omega)


@pytest.fixture
def small_leader_reward() -> np.ndarray:
    """Leader payoffs shaped to match `small_features` (3 leader x 3 follower actions)."""

    return np.array([[0.0, 0.4, 1.0], [0.8, -0.1, 0.2], [0.3, 0.7, -0.2]])


@pytest.fixture
def degenerate_problem() -> CompiledProblem:
    """An unidentifiable instance: every action shares one feature vector."""

    features = np.broadcast_to(np.array([[[0.2, -0.1]]]), (2, 3, 2)).copy()
    return compile_problem(features, eta=1.0, radius=1.0)
