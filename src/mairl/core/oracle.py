"""Application boundary for one-sample follower interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.math import follower_probabilities, validate_policy
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError


@runtime_checkable
class FollowerOracle(Protocol):
    """An application supplies exactly one follower action per selected policy."""

    def sample(self, leader_policy: FloatArray, rng: np.random.Generator) -> int:
        """Return an integer follower-action index."""


def sample_follower_action(oracle: FollowerOracle, problem: CompiledProblem,
        leader_policy: FloatArray, rng: np.random.Generator) -> int:  # fmt: skip
    """Sample once and enforce the application-interface contract."""

    action = oracle.sample(leader_policy.copy(), rng)
    if not isinstance(action, (int, np.integer)):
        raise ConfigurationError(f"follower oracle must return an integer action index; got {type(action)!r}")
    action = int(action)
    if not 0 <= action < problem.n_follower_actions:
        raise ConfigurationError(f"follower oracle returned action {action}, outside [0, {problem.n_follower_actions})")
    return action


@dataclass(frozen=True, slots=True)
class SoftmaxFollowerOracle:
    """Exact model oracle for tests, simulations, and diagnostics."""

    problem: CompiledProblem
    true_omega: FloatArray

    def __init__(self, problem: CompiledProblem, true_omega: ArrayLike) -> None:
        omega = np.asarray(true_omega, dtype=np.float64)
        if omega.shape != (problem.feature_dimension,):
            raise ConfigurationError(f"true_omega must have shape ({problem.feature_dimension},); got {omega.shape}")
        if not np.all(np.isfinite(omega)):
            raise ConfigurationError("true_omega contains NaN or infinite values")
        if np.linalg.norm(omega) > problem.radius + 1e-12:
            raise ConfigurationError("true_omega lies outside the compiled radius")
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "true_omega", omega.copy())

    def probabilities(self, leader_policy: ArrayLike) -> FloatArray:
        return follower_probabilities(self.problem, leader_policy, self.true_omega)

    def sample(self, leader_policy: FloatArray, rng: np.random.Generator) -> int:
        policy = validate_policy(leader_policy, self.problem.n_leader_actions)
        probabilities = self.probabilities(policy)
        return int(rng.choice(self.problem.n_follower_actions, p=probabilities))
