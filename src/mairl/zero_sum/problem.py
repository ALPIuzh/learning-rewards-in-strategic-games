"""Compilation boundary for linear two-player zero-sum matrix games."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from mairl.core.problem import CompilationConfig, CompiledProblem, FloatArray, compile_problem
from mairl.exceptions import CompilationError


@dataclass(frozen=True, slots=True)
class ZeroSumCompilationConfig:
    """Validation rules specific to the zero-sum payoff pair."""

    zero_sum_tolerance: float = 1e-12
    problem: CompilationConfig = field(default_factory=CompilationConfig)

    def __post_init__(self) -> None:
        if self.zero_sum_tolerance < 0:
            raise CompilationError("zero_sum_tolerance must be nonnegative")


@dataclass(frozen=True, slots=True)
class CompiledZeroSumGame:
    """A follower problem augmented with its negated zero-sum leader payoff."""

    problem: CompiledProblem
    leader_features: FloatArray

    @property
    def max_player_features(self) -> FloatArray:
        return self.leader_features

    @property
    def min_player_features(self) -> FloatArray:
        return self.problem.features

    @property
    def n_max_actions(self) -> int:
        return self.problem.n_leader_actions

    @property
    def n_min_actions(self) -> int:
        return self.problem.n_follower_actions

    @property
    def feature_dimension(self) -> int:
        return self.problem.feature_dimension

    @property
    def eta(self) -> float:
        return self.problem.eta

    @property
    def radius(self) -> float:
        return self.problem.radius


def compile_zero_sum_game(max_player_features: ArrayLike, min_player_features: ArrayLike, *, eta: float,
        radius: float, config: ZeroSumCompilationConfig | None = None) -> CompiledZeroSumGame:  # fmt: skip
    """Validate both payoff feature maps and reject non-zero-sum games."""

    rules = config or ZeroSumCompilationConfig()
    problem = compile_problem(min_player_features, eta=eta, radius=radius, config=rules.problem)
    max_features = np.asarray(max_player_features, dtype=np.float64)
    if max_features.shape != problem.features.shape:
        raise CompilationError(
            "max-player features must match min-player feature shape; "
            f"got {max_features.shape} and {problem.features.shape}"
        )
    if not np.all(np.isfinite(max_features)):
        raise CompilationError("max-player features contain NaN or infinite values")
    violation = float(np.max(np.abs(max_features + problem.features)))
    if violation > rules.zero_sum_tolerance:
        raise CompilationError(
            "payoff feature maps are not zero-sum: "
            f"max |phi_max + phi_min|={violation:.3e} exceeds "
            f"{rules.zero_sum_tolerance:.3e}"
        )
    stored = np.ascontiguousarray(max_features)
    stored.setflags(write=False)
    return CompiledZeroSumGame(problem=problem, leader_features=stored)
