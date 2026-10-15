"""Validation certificates for fixed-policy follower responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.math import follower_probabilities, validate_policy
from mairl.core.problem import CompiledProblem, FloatArray, _readonly
from mairl.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class BestResponseValidationConfig:
    """Numerical tolerances for validating a simulated follower response."""

    response_tolerance: float = 1e-9
    parameter_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.response_tolerance <= 0 or self.parameter_tolerance < 0:
            raise ConfigurationError("best-response validation tolerances must be positive/nonnegative")


@dataclass(frozen=True, slots=True)
class ValidatedFollowerBestResponse:
    """Simulation-only certificate for one fixed leader and its true follower response."""

    problem: CompiledProblem
    leader_policy: FloatArray
    follower_policy: FloatArray
    residual: float

    def validate_for(self, problem: CompiledProblem) -> None:
        if self.problem is not problem:
            raise ConfigurationError("validated follower best response belongs to a different compiled problem")


def validate_follower_best_response(problem: CompiledProblem,
        true_omega: ArrayLike, leader_policy: ArrayLike, follower_policy: ArrayLike,
        config: BestResponseValidationConfig | None = None) -> ValidatedFollowerBestResponse:  # fmt: skip
    """Certify the entropy-regularized best response used by a simulation."""

    rules = config or BestResponseValidationConfig()
    omega = np.asarray(true_omega, dtype=np.float64)
    if omega.shape != (problem.feature_dimension,) or not np.all(np.isfinite(omega)):
        raise ConfigurationError(f"true_omega must be finite with shape ({problem.feature_dimension},)")
    omega_norm = float(np.linalg.norm(omega))
    if omega_norm > problem.radius + rules.parameter_tolerance:
        raise ConfigurationError(f"true_omega norm {omega_norm:.6g} exceeds radius {problem.radius:.6g}")
    leader = validate_policy(leader_policy, problem.n_leader_actions)
    follower = validate_policy(follower_policy, problem.n_follower_actions)
    if np.any(follower <= 0):
        raise ConfigurationError("an entropy-regularized follower best response must have full support")

    expected = follower_probabilities(problem, leader, omega)
    residual = float(np.max(np.abs(follower - expected)))
    if residual > rules.response_tolerance:
        raise ConfigurationError(
            "provided follower policy is not the entropy-regularized best response "
            "to the fixed leader policy: "
            f"residual={residual:.3e}, tolerance={rules.response_tolerance:.3e}"
        )
    return ValidatedFollowerBestResponse(
        problem=problem, leader_policy=_readonly(leader), follower_policy=_readonly(follower), residual=residual
    )
