"""Reusable, strictly checked optimization over a finite-dimensional simplex."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from mairl.core.math import project_simplex, project_simplex_rows, projected_simplex_gradient_residual
from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError, ConvergenceError


@dataclass(frozen=True, slots=True)
class PolicyOptimizationConfig:
    """Approximate global maximization over the leader simplex."""

    method: Literal["multistart", "grid"] = "multistart"
    tolerance: float = 1e-9
    projected_gradient_tolerance: float = 1e-5
    max_iterations: int = 1_000
    random_starts: int = 16
    batch_prefilter_iterations: int = 4
    batch_refine_starts: int = 1
    grid_resolution: int = 80
    grid_max_points: int = 250_000
    grid_batch_size: int = 4_096
    refine_grid_solution: bool = True

    def __post_init__(self) -> None:
        if not self.tolerance > 0:
            raise ConfigurationError("tolerance must be strictly positive")
        if not self.projected_gradient_tolerance > 0:
            raise ConfigurationError("projected_gradient_tolerance must be strictly positive")
        if self.max_iterations < 1:
            raise ConfigurationError("max_iterations must be at least 1")
        if self.random_starts < 0:
            raise ConfigurationError("random_starts must be nonnegative")
        if self.batch_prefilter_iterations < 0:
            raise ConfigurationError("batch_prefilter_iterations must be nonnegative")
        if self.batch_refine_starts < 1:
            raise ConfigurationError("batch_refine_starts must be positive")
        if self.grid_resolution < 1 or self.grid_max_points < 1 or self.grid_batch_size < 1:
            raise ConfigurationError("grid limits must be positive")


ObjectiveGradient = Callable[[FloatArray], tuple[float, FloatArray]]
BatchObjective = Callable[[FloatArray], FloatArray]
BatchObjectiveGradient = Callable[..., tuple[FloatArray, FloatArray | None]]
_GRID_CACHE_ENTRIES = 4
_ARMIJO_SUFFICIENT_INCREASE = 1e-4
_BACKTRACKING_FACTOR = 0.5
_MAX_BACKTRACKING_STEPS = 30

# Above this many leader actions, materializing one multistart vertex per pure action
# (np.eye(n_actions)) costs O(n_actions^2) memory -- 4.7GB at 24,389 actions. Comfortably
# above every n_actions used anywhere in this repository's test suite or validated
# applications (largest today: 169), so no currently-tested behavior changes; it exists
# to bound cost once problems get large enough for the dense construction to matter.
_DENSE_VERTEX_LIMIT = 512


@dataclass(frozen=True, slots=True)
class SimplexOptimizationResult:
    policy: FloatArray
    objective: float
    gradient: FloatArray
    projected_gradient_residual: float
    method: str
    attempted_starts: int
    converged_starts: int
    failed_start_messages: tuple[str, ...]
    grid_points: int


@lru_cache(maxsize=_GRID_CACHE_ENTRIES)
def _cached_simplex_grid(n_actions: int, resolution: int, max_points: int) -> FloatArray:
    count = comb(resolution + n_actions - 1, n_actions - 1)
    if count > max_points:
        raise ConfigurationError(
            f"simplex grid would contain {count:,} points, above grid_max_points="
            f"{max_points:,}; reduce resolution or use multistart"
        )
    points = np.empty((count, n_actions), dtype=np.float64)
    last = resolution + n_actions - 1
    for row, bars in enumerate(combinations(range(last), n_actions - 1)):
        cuts = (-1, *bars, last)
        points[row] = [cuts[i + 1] - cuts[i] - 1 for i in range(n_actions)]
    points /= resolution
    points.setflags(write=False)
    return points


def simplex_grid(n_actions: int, resolution: int, max_points: int) -> FloatArray:
    """Return a cached, read-only regular simplex lattice."""

    return _cached_simplex_grid(n_actions, resolution, max_points)


def _evaluate_grid(points: FloatArray, objective_gradient: ObjectiveGradient,
        batch_objective: BatchObjective | None, batch_size: int) -> FloatArray:  # fmt: skip
    if batch_objective is None:
        values = np.fromiter(
            (objective_gradient(point)[0] for point in points), dtype=np.float64, count=points.shape[0]
        )
    else:
        values = np.empty(points.shape[0], dtype=np.float64)
        for start in range(0, points.shape[0], batch_size):
            stop = min(start + batch_size, points.shape[0])
            batch_values = np.asarray(batch_objective(points[start:stop]), dtype=np.float64)
            if batch_values.shape != (stop - start,):
                raise ConfigurationError(
                    "batch objective must return one value per policy; "
                    f"expected {(stop - start,)}, got {batch_values.shape}"
                )
            values[start:stop] = batch_values
    if not np.all(np.isfinite(values)):
        raise ConvergenceError("simplex-grid objective returned a non-finite value")
    return values


def _local_solve(objective_gradient: ObjectiveGradient, start: FloatArray,
        config: PolicyOptimizationConfig) -> tuple[FloatArray, float, FloatArray, float, str | None]:  # fmt: skip
    n_actions = start.size

    def minimization_objective(policy: FloatArray) -> tuple[float, FloatArray]:
        value, gradient = objective_gradient(policy)
        return -value, -gradient

    result = minimize(
        minimization_objective,
        start,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * n_actions,
        constraints=({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0), "jac": lambda x: np.ones_like(x)},),
        options={"ftol": config.tolerance, "maxiter": config.max_iterations, "disp": False},
    )
    policy = project_simplex(np.asarray(result.x, dtype=np.float64))
    value, gradient = objective_gradient(policy)
    scale = max(1.0, float(np.linalg.norm(gradient)))
    residual = projected_simplex_gradient_residual(policy, gradient / scale)

    failure: str | None = None
    if not result.success:
        failure = str(result.message)
    elif residual > config.projected_gradient_tolerance:
        failure = f"projected-gradient residual {residual:.3e} exceeds {config.projected_gradient_tolerance:.3e}"
    return policy, value, gradient, residual, failure


def _batch_starts(n_actions: int, config: PolicyOptimizationConfig,
        rngs: tuple[np.random.Generator, ...], vertex_scores: FloatArray | None) -> FloatArray:  # fmt: skip
    """Build uniform, pure-action, and random starts for each independent problem."""

    batch_size = len(rngs)
    n_vertices = min(n_actions, _DENSE_VERTEX_LIMIT)
    if n_actions > _DENSE_VERTEX_LIMIT:
        top_indices = np.argpartition(vertex_scores, -n_vertices, axis=1)[:, -n_vertices:]
        vertices = np.zeros((batch_size, n_vertices, n_actions), dtype=np.float64)
        batch_indices = np.arange(batch_size)[:, None]
        start_indices = np.arange(n_vertices)[None, :]
        vertices[batch_indices, start_indices, top_indices] = 1.0
    else:
        vertices = np.broadcast_to(np.eye(n_actions, dtype=np.float64), (batch_size, n_vertices, n_actions))

    policies = np.empty((batch_size, 1 + n_vertices + config.random_starts, n_actions), dtype=np.float64)
    policies[:, 0, :] = 1.0 / n_actions
    policies[:, 1 : 1 + n_vertices, :] = vertices
    if config.random_starts:
        for index, rng in enumerate(rngs):
            policies[index, 1 + n_vertices :] = rng.dirichlet(np.ones(n_actions), size=config.random_starts)
    return policies


def _prefilter_batch(policies: FloatArray, objective_gradient: BatchObjectiveGradient,
        config: PolicyOptimizationConfig) -> FloatArray:  # fmt: skip
    """Improve starts in place with projected ascent, then return their objective values."""

    for _ in range(config.batch_prefilter_iterations):
        values, gradients = objective_gradient(policies)
        scales = np.maximum(1.0, np.linalg.norm(gradients, axis=2))
        projected = project_simplex_rows(policies + gradients / scales[:, :, None])
        direction = projected - policies
        derivative = np.einsum("ska,ska->sk", gradients, direction, optimize=True)
        searching = np.isfinite(derivative) & (derivative > 0)
        if not np.any(searching):
            break
        steps = np.ones(policies.shape[:2], dtype=np.float64)
        for _ in range(_MAX_BACKTRACKING_STEPS + 1):
            candidate = policies + steps[:, :, None] * direction
            candidate_values, _ = objective_gradient(candidate, with_gradient=False)
            threshold = values + _ARMIJO_SUFFICIENT_INCREASE * steps * derivative
            newly_accepted = searching & np.isfinite(candidate_values) & (candidate_values >= threshold)
            policies[newly_accepted] = candidate[newly_accepted]
            searching &= ~newly_accepted
            if not np.any(searching):
                break
            steps[searching] *= _BACKTRACKING_FACTOR
        if np.any(searching):
            break
    values, _ = objective_gradient(policies, with_gradient=False)
    return values


def maximize_on_simplex(objective_gradient: ObjectiveGradient, *, n_actions: int,
        config: PolicyOptimizationConfig, rng: np.random.Generator, batch_objective: BatchObjective | None = None,
        vertex_scores: FloatArray | None = None) -> SimplexOptimizationResult:  # fmt: skip
    """Maximize using the configured approximation and expose its diagnostics.

    ``batch_objective`` is used only for grid evaluation. It preserves the exact grid
    search while moving objective evaluation out of the Python point-by-point loop.

    ``vertex_scores``, if given, is the exact objective value at every pure-action
    vertex e_a, computed cheaply by the caller in closed form (no dense (n_actions,
    n_actions) matrix). Below ``_DENSE_VERTEX_LIMIT`` it is ignored and every vertex is
    still tried, exactly as without this parameter. Above that limit, only the top
    ``_DENSE_VERTEX_LIMIT`` vertices by exact score are tried -- not an approximation of
    which vertices are good, since the score is exact, only of how many are worth the
    cost of a full local refinement.
    """

    if n_actions < 1:
        raise ConfigurationError("n_actions must be positive")
    if n_actions == 1:
        policy = np.ones(1)
        objective, gradient = objective_gradient(policy)
        return SimplexOptimizationResult(policy, objective, gradient, 0.0, config.method, 1, 1, (), 1)

    if config.method == "grid":
        points = simplex_grid(n_actions, config.grid_resolution, config.grid_max_points)
        values = _evaluate_grid(points, objective_gradient, batch_objective, config.grid_batch_size)
        best_grid = points[int(np.argmax(values))]
        if not config.refine_grid_solution:
            objective, gradient = objective_gradient(best_grid)
            return SimplexOptimizationResult(
                policy=best_grid,
                objective=objective,
                gradient=gradient,
                projected_gradient_residual=float("nan"),
                method="grid",
                attempted_starts=0,
                converged_starts=0,
                failed_start_messages=(),
                grid_points=int(points.shape[0]),
            )
        policy, value, gradient, residual, failure = _local_solve(objective_gradient, best_grid, config)
        if failure is not None:
            raise ConvergenceError(f"grid refinement failed: {failure}")
        return SimplexOptimizationResult(
            policy=policy,
            objective=value,
            gradient=gradient,
            projected_gradient_residual=residual,
            method="grid+refine",
            attempted_starts=1,
            converged_starts=1,
            failed_start_messages=(),
            grid_points=int(points.shape[0]),
        )

    starts = [np.full(n_actions, 1.0 / n_actions)]
    if vertex_scores is not None and n_actions > _DENSE_VERTEX_LIMIT:
        if vertex_scores.shape != (n_actions,):
            raise ConfigurationError(f"vertex_scores must have shape ({n_actions},); got {vertex_scores.shape}")
        top_indices = np.argpartition(vertex_scores, -_DENSE_VERTEX_LIMIT)[-_DENSE_VERTEX_LIMIT:]
        vertex_starts = np.zeros((_DENSE_VERTEX_LIMIT, n_actions), dtype=np.float64)
        vertex_starts[np.arange(_DENSE_VERTEX_LIMIT), top_indices] = 1.0
        starts.extend(vertex_starts)
    else:
        starts.extend(np.eye(n_actions, dtype=np.float64))
    if config.random_starts:
        starts.extend(rng.dirichlet(np.ones(n_actions), size=config.random_starts))

    successful: list[tuple[FloatArray, float, FloatArray, float]] = []
    failures: list[str] = []
    for index, start in enumerate(starts):
        policy, value, gradient, residual, failure = _local_solve(objective_gradient, np.asarray(start), config)
        if failure is None:
            successful.append((policy, value, gradient, residual))
        else:
            failures.append(f"start {index}: {failure}")
    if not successful:
        raise ConvergenceError("all multistart simplex optimizations failed: " + " | ".join(failures))
    policy, value, gradient, residual = max(successful, key=lambda item: item[1])
    return SimplexOptimizationResult(
        policy=policy,
        objective=value,
        gradient=gradient,
        projected_gradient_residual=residual,
        method="multistart",
        attempted_starts=len(starts),
        converged_starts=len(successful),
        failed_start_messages=tuple(failures),
        grid_points=0,
    )
