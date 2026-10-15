"""Offline grouped D-optimal design, support reduction, and allocation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize_scalar

from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError, ConvergenceError


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    """Offline grouped D-optimal-design and allocation settings."""

    budget: int
    design_tolerance: float
    enabled: bool = True
    max_iterations: int = 10_000
    line_search_tolerance: float = 1e-12
    line_search_max_iterations: int = 500
    support_tolerance: float = 1e-12
    matrix_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ConfigurationError("budget must be nonnegative")
        if not 0 < self.design_tolerance <= 1:
            raise ConfigurationError("design_tolerance must lie in (0, 1]")
        if self.max_iterations < 1 or self.line_search_max_iterations < 1:
            raise ConfigurationError("iteration limits must be at least 1")
        values = {
            "line_search_tolerance": self.line_search_tolerance,
            "support_tolerance": self.support_tolerance,
            "matrix_tolerance": self.matrix_tolerance,
        }
        for name, value in values.items():
            if not value > 0:
                raise ConfigurationError(f"{name} must be strictly positive; got {value!r}")

    @property
    def effective_budget(self) -> int:
        """Number of interactions collected when this configuration is used."""

        return self.budget if self.enabled else 0


@dataclass(frozen=True, slots=True)
class FrankWolfeDiagnostics:
    iterations: int
    final_gap: float
    objective: float
    max_directional_value: float
    gap_history: FloatArray
    objective_history: FloatArray


@dataclass(frozen=True, slots=True)
class WarmupDesign:
    weights: FloatArray
    support_actions: np.ndarray
    support_weights: FloatArray
    allocation_counts: np.ndarray
    coverage_matrix: FloatArray
    diagnostics: FrankWolfeDiagnostics

    @property
    def support_size(self) -> int:
        return int(self.support_actions.size)


def _coverage(problem: CompiledProblem, weights: FloatArray) -> FloatArray:
    return np.einsum("a,agh->gh", weights, problem.pure_coverage_matrices, optimize=True)


def _logdet(matrix: FloatArray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(value):
        return -np.inf
    return float(value)


def _initial_design(problem: CompiledProblem, matrix_tolerance: float) -> FloatArray:
    dimension = problem.identifiable_dimension
    n_follower = problem.n_follower_actions
    vectors = problem.reduced_centered_pure_features.reshape(-1, dimension)
    _, _, pivots = qr(vectors.T, mode="economic", pivoting=True)
    basis_rows = pivots[:dimension]
    actions = np.unique(basis_rows // n_follower)
    weights = np.zeros(problem.n_leader_actions, dtype=np.float64)
    weights[actions] = 1.0 / actions.size
    eigenvalues = np.linalg.eigvalsh(_coverage(problem, weights))
    if eigenvalues[0] <= matrix_tolerance:
        raise ConvergenceError(
            f"failed to construct an admissible warm-up design: smallest reduced eigenvalue={eigenvalues[0]:.3e}"
        )
    return weights


def _support_reduce(matrices: FloatArray, weights: FloatArray,
        *, support_tolerance: float, matrix_tolerance: float) -> FloatArray:  # fmt: skip
    """Constructive Caratheodory reduction preserving the coverage matrix."""

    original = np.einsum("a,agh->gh", weights, matrices, optimize=True)
    reduced = weights.copy()
    dimension = matrices.shape[1]
    coordinate_count = dimension * (dimension + 1) // 2
    target_support = coordinate_count + 1
    upper = np.triu_indices(dimension)

    while np.count_nonzero(reduced > support_tolerance) > target_support:
        support = np.flatnonzero(reduced > support_tolerance)
        point_coordinates = matrices[support][:, upper[0], upper[1]].T
        affine = np.vstack((np.ones(support.size), point_coordinates))
        _, singular_values, vh = np.linalg.svd(affine, full_matrices=True)
        direction = vh[-1]
        residual = np.linalg.norm(affine @ direction)
        scale = max(1.0, float(singular_values[0]) if singular_values.size else 1.0)
        if residual > matrix_tolerance * scale:
            raise ConvergenceError(f"support-reduction null-space residual {residual:.3e} is too large")
        positive = direction > support_tolerance
        if not np.any(positive):
            direction = -direction
            positive = direction > support_tolerance
        step = float(np.min(reduced[support][positive] / direction[positive]))
        updated = reduced[support] - step * direction
        updated[np.abs(updated) <= support_tolerance] = 0.0
        if np.min(updated) < -matrix_tolerance:
            raise ConvergenceError("support reduction produced a negative weight")
        reduced[support] = np.maximum(updated, 0.0)
        reduced /= np.sum(reduced)

    preserved = np.einsum("a,agh->gh", reduced, matrices, optimize=True)
    error = float(np.linalg.norm(preserved - original, ord="fro"))
    if error > matrix_tolerance * max(1.0, float(np.linalg.norm(original, ord="fro"))):
        raise ConvergenceError(f"support reduction changed coverage by Frobenius error {error:.3e}")
    return reduced


def _round_allocation(support_weights: FloatArray, budget: int, design_tolerance: float) -> np.ndarray:
    support_size = support_weights.size
    minimum = ceil((1.0 + design_tolerance) * support_size / design_tolerance)
    if budget < minimum:
        raise ConfigurationError(
            "warm-up budget is too small for the paper's rounding condition: "
            f"budget={budget}, required at least {minimum} for support={support_size} "
            f"and tolerance={design_tolerance}"
        )
    order = np.argsort(-support_weights, kind="stable")
    base = np.ceil((budget - support_size) * support_weights).astype(np.int64)
    residual = int(budget - np.sum(base))
    if not 0 <= residual <= support_size:
        raise ConvergenceError(f"warm-up rounding residual must lie in [0, {support_size}]; got {residual}")
    counts = base.copy()
    counts[order[:residual]] += 1
    if int(np.sum(counts)) != budget or np.any(counts < 1):
        raise ConvergenceError("warm-up allocation failed its integer postconditions")
    return counts


def compute_warmup_design(problem: CompiledProblem, config: WarmupConfig) -> WarmupDesign:
    """Run Algorithm 4, reduce its support, and round to the requested budget."""

    if not config.enabled or problem.is_degenerate:
        empty = np.empty(0, dtype=np.float64)
        diagnostics = FrankWolfeDiagnostics(0, 0.0, 0.0, 0.0, empty, empty)
        dimension = problem.identifiable_dimension
        return WarmupDesign(
            weights=np.zeros(problem.n_leader_actions),
            support_actions=np.empty(0, dtype=np.int64),
            support_weights=empty,
            allocation_counts=np.empty(0, dtype=np.int64),
            coverage_matrix=np.zeros((dimension, dimension), dtype=np.float64),
            diagnostics=diagnostics,
        )

    dimension = problem.identifiable_dimension
    weights = _initial_design(problem, config.matrix_tolerance)
    gap_history: list[float] = []
    objective_history: list[float] = []

    converged = False
    for iteration in range(config.max_iterations + 1):
        coverage = _coverage(problem, weights)
        objective = _logdet(coverage)
        if not np.isfinite(objective):
            raise ConvergenceError("warm-up coverage lost positive definiteness")
        inverse = np.linalg.inv(coverage)
        directional_values = np.einsum("gh,agh->a", inverse, problem.pure_coverage_matrices, optimize=True)
        selected = int(np.argmax(directional_values))
        max_directional = float(directional_values[selected])
        gap = max_directional - dimension
        gap_history.append(gap)
        objective_history.append(objective)
        if gap <= config.design_tolerance * dimension:
            converged = True
            break
        if iteration == config.max_iterations:
            break

        selected_matrix = problem.pure_coverage_matrices[selected]

        def negative_logdet(step: float, current: FloatArray = coverage, arm: FloatArray = selected_matrix) -> float:
            candidate = (1.0 - step) * current + step * arm
            return -_logdet(candidate)

        line_search = minimize_scalar(
            negative_logdet,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": config.line_search_tolerance, "maxiter": config.line_search_max_iterations},
        )
        if not line_search.success:
            raise ConvergenceError(f"warm-up line search failed: {line_search.message}")
        step = float(line_search.x)
        if step <= config.line_search_tolerance:
            raise ConvergenceError(f"warm-up line search stalled with gap {gap:.3e} and step {step:.3e}")
        weights *= 1.0 - step
        weights[selected] += step

    if not converged:
        raise ConvergenceError(
            "warm-up Frank-Wolfe did not converge within "
            f"{config.max_iterations} iterations; final gap={gap_history[-1]:.3e}"
        )

    reduced = _support_reduce(
        problem.pure_coverage_matrices,
        weights,
        support_tolerance=config.support_tolerance,
        matrix_tolerance=config.matrix_tolerance,
    )
    support = np.flatnonzero(reduced > config.support_tolerance)
    support_weights = reduced[support]
    support_weights /= np.sum(support_weights)
    counts = _round_allocation(support_weights, config.budget, config.design_tolerance)
    allocation_coverage = np.einsum(
        "a,agh->gh",
        np.bincount(support, weights=counts, minlength=problem.n_leader_actions),
        problem.pure_coverage_matrices,
        optimize=True,
    )
    diagnostics = FrankWolfeDiagnostics(
        iterations=iteration,
        final_gap=float(gap_history[-1]),
        objective=float(objective_history[-1]),
        max_directional_value=max_directional,
        gap_history=np.asarray(gap_history),
        objective_history=np.asarray(objective_history),
    )
    return WarmupDesign(
        weights=reduced,
        support_actions=support,
        support_weights=support_weights,
        allocation_counts=counts,
        coverage_matrix=allocation_coverage,
        diagnostics=diagnostics,
    )
