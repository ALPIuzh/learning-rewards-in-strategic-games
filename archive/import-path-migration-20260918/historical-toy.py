"""Seeded normal-form game for fast verification and comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.problem import CompiledProblem, FloatArray, compile_problem
from mairl.exceptions import ConfigurationError


def make_toy_problem(
    seed: int,
    *,
    eta: float = 1.25,
    radius: float = 2.0,
    true_omega: ArrayLike | None = None,
    leader_reward: ArrayLike | None = None,
) -> tuple[CompiledProblem, FloatArray, FloatArray]:
    """Return the shared toy problem, true follower coefficients, and leader reward."""

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(3, 4, 3))
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features /= np.maximum(1.0, 1.1 * norms)
    omega = np.asarray(
        [0.55, -0.65, 0.35] if true_omega is None else true_omega,
        dtype=np.float64,
    )
    reward = np.asarray(
        (
            [
                [0.1, 0.9, 0.3, -0.2],
                [0.7, -0.1, 0.5, 0.2],
                [-0.2, 0.4, 0.8, 0.6],
            ]
            if leader_reward is None
            else leader_reward
        ),
        dtype=np.float64,
    )
    if omega.shape != (3,) or not np.all(np.isfinite(omega)):
        raise ConfigurationError("toy true_omega must be finite with shape (3,)")
    if reward.shape != (3, 4) or not np.all(np.isfinite(reward)):
        raise ConfigurationError("toy leader_reward must be finite with shape (3, 4)")
    return (
        compile_problem(features, eta=eta, radius=radius),
        omega,
        reward,
    )
