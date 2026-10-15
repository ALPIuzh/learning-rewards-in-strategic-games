"""Plug-in OLS reward estimation for one fixed leader policy."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from mairl.core.math import validate_policy
from mairl.core.problem import CompiledProblem, FloatArray, _readonly
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.inference.best_response import ValidatedFollowerBestResponse

IdentifiabilityMode = Literal["require_unique", "minimum_norm"]
ConstraintMode = Literal["unconstrained", "ball"]
EquationMode = Literal["follower_only", "zero_sum"]


class FrequencySmoothingWarning(UserWarning):
    """The configured pseudocount materially affects an empirical policy."""


class EstimateOutsideBallWarning(UserWarning):
    """The unconstrained OLS solution lies outside the known parameter ball."""


@dataclass(frozen=True, slots=True)
class FixedLeaderOLSConfig:
    """All statistical and numerical choices for fixed-leader logit OLS."""

    pseudocount: float
    smoothing_warning_ratio: float
    identifiability: IdentifiabilityMode = "require_unique"
    constraint: ConstraintMode = "ball"
    follower_reference_action: int = 0
    leader_reference_action: int = 0
    rank_relative_tolerance: float = 1e-10
    rank_absolute_tolerance: float = 1e-12
    ball_tolerance: float = 1e-10
    ball_root_tolerance: float = 1e-12
    ball_max_iterations: int = 200

    def __post_init__(self) -> None:
        if self.pseudocount <= 0:
            raise ConfigurationError("pseudocount must be strictly positive")
        if self.smoothing_warning_ratio <= 0:
            raise ConfigurationError("smoothing_warning_ratio must be strictly positive")
        if self.identifiability not in ("require_unique", "minimum_norm"):
            raise ConfigurationError("identifiability must be 'require_unique' or 'minimum_norm'")
        if self.constraint not in ("unconstrained", "ball"):
            raise ConfigurationError("constraint must be 'unconstrained' or 'ball'")
        if self.follower_reference_action < 0 or self.leader_reference_action < 0:
            raise ConfigurationError("reference action indices must be nonnegative")
        if self.rank_relative_tolerance < 0 or self.rank_absolute_tolerance < 0:
            raise ConfigurationError("rank tolerances must be nonnegative")
        if self.ball_tolerance < 0 or self.ball_root_tolerance <= 0:
            raise ConfigurationError("ball tolerances must be nonnegative/positive")
        if self.ball_max_iterations < 1:
            raise ConfigurationError("ball_max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class FrequencyDiagnostics:
    sample_count: int
    zero_count_actions: int
    minimum_positive_count: int
    pseudocount_to_minimum_count: float


@dataclass(frozen=True, slots=True)
class FixedLeaderOLSResult:
    """Estimate and complete diagnostics for auditing the plug-in estimator."""

    omega: FloatArray
    unconstrained_omega: FloatArray
    follower_payoff: FloatArray
    leader_policy: FloatArray
    follower_policy: FloatArray
    design_matrix: FloatArray
    target: FloatArray
    singular_values: FloatArray
    rank: int
    residual_norm: float
    unconstrained_residual_norm: float
    constrained: bool
    identifiable: bool
    equation_mode: EquationMode
    follower_sample_count: int
    leader_frequency_diagnostics: FrequencyDiagnostics | None
    follower_frequency_diagnostics: FrequencyDiagnostics | None


def _validate_actions(actions: ArrayLike, n_actions: int, name: str) -> np.ndarray:
    values = np.asarray(actions)
    if values.ndim != 1:
        raise ConfigurationError(f"{name} must be a one-dimensional action array")
    if values.size < 1:
        raise ConfigurationError(f"{name} must contain at least one action")
    if not np.issubdtype(values.dtype, np.integer):
        raise ConfigurationError(f"{name} must contain integer action indices")
    indices = np.asarray(values, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= n_actions):
        raise ConfigurationError(
            f"{name} contains an index outside [0, {n_actions}); min={indices.min()}, max={indices.max()}"
        )
    return indices


def _frequency_policy(actions: ArrayLike, n_actions: int, config: FixedLeaderOLSConfig,
        *, name: str) -> tuple[FloatArray, FrequencyDiagnostics]:  # fmt: skip
    indices = _validate_actions(actions, n_actions, name)
    counts = np.bincount(indices, minlength=n_actions)
    positive = counts[counts > 0]
    minimum_positive = int(np.min(positive))
    zero_count = int(np.count_nonzero(counts == 0))
    ratio = float(config.pseudocount / minimum_positive)
    if zero_count or ratio >= config.smoothing_warning_ratio:
        reasons = []
        if zero_count:
            reasons.append(f"{zero_count} unobserved actions receive only pseudocount mass")
        if ratio >= config.smoothing_warning_ratio:
            reasons.append(
                f"pseudocount/minimum-positive-count={ratio:.3g} >= warning ratio {config.smoothing_warning_ratio:.3g}"
            )
        warnings.warn(f"{name} smoothing is material: " + "; ".join(reasons), FrequencySmoothingWarning, stacklevel=3)
    smoothed = counts.astype(np.float64) + config.pseudocount
    smoothed /= np.sum(smoothed)
    return _readonly(smoothed), FrequencyDiagnostics(
        sample_count=int(indices.size),
        zero_count_actions=zero_count,
        minimum_positive_count=minimum_positive,
        pseudocount_to_minimum_count=ratio,
    )


def _build_follower_system(problem: CompiledProblem, leader_policy: ArrayLike, follower_policy: ArrayLike,
        config: FixedLeaderOLSConfig) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:  # fmt: skip
    leader = validate_policy(leader_policy, problem.n_leader_actions)
    follower = validate_policy(follower_policy, problem.n_follower_actions)
    if np.any(follower <= 0):
        raise ConfigurationError("log-ratio equations require a full-support follower policy")
    reference = config.follower_reference_action
    if reference >= problem.n_follower_actions:
        raise ConfigurationError("follower_reference_action is outside the follower action space")

    follower_actions = np.delete(np.arange(problem.n_follower_actions), reference)
    expected_features = np.einsum("a,abd->bd", leader, problem.features, optimize=True)
    matrix = expected_features[follower_actions] - expected_features[reference]
    target = np.log(follower[follower_actions] / follower[reference]) / problem.eta
    return (_readonly(leader), _readonly(follower), _readonly(matrix), _readonly(target))


def _minimum_norm_solution(matrix: FloatArray, target: FloatArray,
        config: FixedLeaderOLSConfig) -> tuple[FloatArray, FloatArray, int]:  # fmt: skip
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(config.rank_absolute_tolerance, config.rank_relative_tolerance * largest)
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank < matrix.shape[1] and config.identifiability == "require_unique":
        raise ConfigurationError(
            "log-ratio OLS system is not strongly identifiable: "
            f"rank={rank}, feature_dimension={matrix.shape[1]}, "
            f"rank_threshold={threshold:.3e}"
        )
    if rank == 0:
        solution = np.zeros(matrix.shape[1], dtype=np.float64)
    else:
        solution = vh[:rank].T @ ((u[:, :rank].T @ target) / singular_values[:rank])
    return solution, singular_values, rank


def _ball_constrained_solution(matrix: FloatArray, target: FloatArray,
        radius: float, config: FixedLeaderOLSConfig) -> FloatArray:  # fmt: skip
    hessian = matrix.T @ matrix
    gradient = matrix.T @ target
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated = eigenvectors.T @ gradient
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    threshold = max(config.rank_absolute_tolerance**2, config.rank_relative_tolerance**2 * largest)
    active = eigenvalues > threshold

    def solution(multiplier: float) -> FloatArray:
        coefficients = np.zeros_like(rotated)
        coefficients[active] = rotated[active] / (eigenvalues[active] + multiplier)
        return eigenvectors @ coefficients

    def secular(multiplier: float) -> float:
        return float(np.linalg.norm(solution(multiplier)) - radius)

    upper = max(1.0, float(np.linalg.norm(gradient)) / radius)
    for _ in range(config.ball_max_iterations):
        if secular(upper) <= 0:
            break
        upper *= 2.0
    else:
        raise ConvergenceError("failed to bracket the ball-constrained OLS multiplier")

    def solve_multiplier(root_tolerance: float) -> FloatArray:
        multiplier = brentq(secular, 0.0, upper, xtol=root_tolerance, maxiter=config.ball_max_iterations)
        return solution(float(multiplier))

    constrained = solve_multiplier(config.ball_root_tolerance)
    norm_error = abs(float(np.linalg.norm(constrained)) - radius)
    if norm_error > config.ball_tolerance:
        # An absolute multiplier tolerance can be too coarse when the design is
        # severely ill-conditioned and the secular root is close to zero. Retry
        # at machine precision before declaring a postcondition failure.
        constrained = solve_multiplier(np.finfo(np.float64).tiny)
        norm_error = abs(float(np.linalg.norm(constrained)) - radius)
    if norm_error > config.ball_tolerance:
        raise ConvergenceError("ball-constrained OLS solution failed its norm postcondition")
    return constrained


def _solve_ols_system(problem: CompiledProblem, leader_policy: ArrayLike, follower_policy: ArrayLike,
        matrix: FloatArray, target: FloatArray, config: FixedLeaderOLSConfig, *, equation_mode: EquationMode,
        follower_sample_count: int = 0, leader_frequency_diagnostics: FrequencyDiagnostics | None = None,
        follower_frequency_diagnostics: FrequencyDiagnostics | None = None) -> FixedLeaderOLSResult:  # fmt: skip
    leader = validate_policy(leader_policy, problem.n_leader_actions)
    follower = validate_policy(follower_policy, problem.n_follower_actions)
    raw, singular_values, rank = _minimum_norm_solution(matrix, target, config)
    raw_norm = float(np.linalg.norm(raw))
    outside = raw_norm > problem.radius + config.ball_tolerance
    if outside:
        warnings.warn(
            f"unconstrained log-ratio OLS norm {raw_norm:.6g} exceeds radius {problem.radius:.6g}",
            EstimateOutsideBallWarning,
            stacklevel=3,
        )
    constrained = outside and config.constraint == "ball"
    omega = _ball_constrained_solution(matrix, target, problem.radius, config) if constrained else raw.copy()
    residual = float(np.linalg.norm(matrix @ omega - target))
    raw_residual = float(np.linalg.norm(matrix @ raw - target))
    follower_payoff = np.einsum("abd,d->ab", problem.features, omega, optimize=True)
    return FixedLeaderOLSResult(
        omega=_readonly(omega),
        unconstrained_omega=_readonly(raw),
        follower_payoff=_readonly(follower_payoff),
        leader_policy=_readonly(leader),
        follower_policy=_readonly(follower),
        design_matrix=_readonly(matrix),
        target=_readonly(target),
        singular_values=_readonly(singular_values),
        rank=rank,
        residual_norm=residual,
        unconstrained_residual_norm=raw_residual,
        constrained=constrained,
        identifiable=rank == problem.feature_dimension,
        equation_mode=equation_mode,
        follower_sample_count=follower_sample_count,
        leader_frequency_diagnostics=leader_frequency_diagnostics,
        follower_frequency_diagnostics=follower_frequency_diagnostics,
    )


def solve_fixed_leader_ols(problem: CompiledProblem, response: ValidatedFollowerBestResponse,
        follower_policy: ArrayLike, config: FixedLeaderOLSConfig, *, follower_sample_count: int = 0,
        follower_frequency_diagnostics: FrequencyDiagnostics | None = None) -> FixedLeaderOLSResult:  # fmt: skip
    """Solve the follower log-ratio equations for a validated fixed leader."""

    response.validate_for(problem)
    leader, follower, matrix, target = _build_follower_system(problem, response.leader_policy, follower_policy, config)
    return _solve_ols_system(
        problem,
        leader,
        follower,
        matrix,
        target,
        config,
        equation_mode="follower_only",
        follower_sample_count=follower_sample_count,
        follower_frequency_diagnostics=follower_frequency_diagnostics,
    )


def fit_fixed_leader_ols(problem: CompiledProblem, response: ValidatedFollowerBestResponse,
        follower_actions: ArrayLike, config: FixedLeaderOLSConfig) -> FixedLeaderOLSResult:  # fmt: skip
    """Estimate follower reward from responses to one fixed, known leader policy."""

    response.validate_for(problem)
    follower, diagnostics = _frequency_policy(
        follower_actions, problem.n_follower_actions, config, name="follower_actions"
    )
    return solve_fixed_leader_ols(
        problem,
        response,
        follower,
        config,
        follower_sample_count=diagnostics.sample_count,
        follower_frequency_diagnostics=diagnostics,
    )
