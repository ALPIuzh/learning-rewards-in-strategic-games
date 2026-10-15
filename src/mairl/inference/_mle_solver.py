"""Exact-Newton and constrained-ball machinery behind the public MLE flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize, root

from mairl.core.math import project_ball
from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConvergenceError
from mairl.inference._mle_likelihood import (
    PreparedLikelihood,
    evaluate_prepared,
    evaluate_prepared_batch,
    likelihood_and_gradient_prepared,
)
from mairl.inference._mle_types import MLEConfig, MLEResult

_ARMIJO_SUFFICIENT_INCREASE = 1e-4
_BACKTRACKING_FACTOR = 0.5
_MAX_BACKTRACKING_STEPS = 30


@dataclass(frozen=True, slots=True)
class NewtonResult:
    omega: FloatArray
    converged: bool
    iterations: int
    function_evaluations: int


def unconstrained_newton(problem: CompiledProblem, prepared: PreparedLikelihood,
        config: MLEConfig, initial: FloatArray) -> NewtonResult:  # fmt: skip
    """Solve the unconstrained problem; the caller handles the ball exactly."""

    omega = initial.copy()
    evaluations = 0
    for iteration in range(config.max_iterations + 1):
        _, objective, gradient, hessian = evaluate_prepared(problem, prepared, omega, config.ridge, with_hessian=True)
        evaluations += 1
        if np.linalg.norm(gradient) <= config.projected_gradient_tolerance:
            return NewtonResult(omega, True, iteration, evaluations)
        if iteration == config.max_iterations:
            break
        if hessian is None:
            raise RuntimeError("internal error: Hessian was not computed")

        try:
            direction = np.linalg.solve(-hessian, gradient)
        except np.linalg.LinAlgError:
            break
        directional_derivative = float(gradient @ direction)
        if not np.isfinite(directional_derivative) or directional_derivative <= 0:
            break

        step = 1.0
        for _ in range(_MAX_BACKTRACKING_STEPS + 1):
            candidate = omega + step * direction
            _, candidate_objective, candidate_gradient, _ = evaluate_prepared(
                problem, prepared, candidate, config.ridge, with_hessian=False
            )
            evaluations += 1
            sufficient_increase = objective + _ARMIJO_SUFFICIENT_INCREASE * step * directional_derivative
            objective_allowance = config.objective_tolerance * max(1.0, abs(objective))
            numerical_refinement = candidate_objective >= objective - objective_allowance and np.linalg.norm(
                candidate_gradient
            ) < np.linalg.norm(gradient)
            if np.isfinite(candidate_objective) and (
                candidate_objective >= sufficient_increase or numerical_refinement
            ):
                omega = candidate
                break
            step *= _BACKTRACKING_FACTOR
        else:
            break

    return NewtonResult(omega, False, iteration, evaluations)


def unconstrained_newton_batch(problem: CompiledProblem, prepared: PreparedLikelihood, config: MLEConfig,
        initial: FloatArray) -> tuple[FloatArray, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Run masked exact-Newton iterations for many independent MLEs."""

    omega = initial.copy()
    seed_count = omega.shape[0]
    active = np.ones(seed_count, dtype=bool)
    converged = np.zeros(seed_count, dtype=bool)
    iterations = np.zeros(seed_count, dtype=np.int64)
    evaluations = np.zeros(seed_count, dtype=np.int64)

    for _ in range(config.max_iterations + 1):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        subset = prepared.select(indices)
        _, objective, gradient, hessian = evaluate_prepared_batch(
            problem, subset, omega[indices], config.ridge, with_hessian=True
        )
        evaluations[indices] += 1
        finished = np.linalg.norm(gradient, axis=1) <= config.projected_gradient_tolerance
        if np.any(finished):
            finished_indices = indices[finished]
            converged[finished_indices] = True
            active[finished_indices] = False
        indices = indices[~finished]
        if indices.size == 0:
            continue
        if np.max(iterations[indices], initial=0) >= config.max_iterations:
            active[indices] = False
            continue
        gradient = gradient[~finished]
        objective = objective[~finished]
        if hessian is None:
            raise RuntimeError("internal error: Hessian was not computed")
        hessian = hessian[~finished]
        try:
            direction = np.linalg.solve(-hessian, gradient[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            active[indices] = False
            continue
        derivative = np.einsum("sd,sd->s", gradient, direction, optimize=True)
        valid = np.isfinite(derivative) & (derivative > 0)
        if np.any(~valid):
            active[indices[~valid]] = False
        indices = indices[valid]
        if indices.size == 0:
            continue
        direction = direction[valid]
        derivative = derivative[valid]
        objective = objective[valid]
        iterations[indices] += 1

        searching = np.ones(indices.size, dtype=bool)
        steps = np.ones(indices.size, dtype=np.float64)
        for _ in range(_MAX_BACKTRACKING_STEPS + 1):
            local = np.flatnonzero(searching)
            if local.size == 0:
                break
            candidate = omega[indices[local]] + steps[local, None] * direction[local]
            candidate_rows = indices[local]
            candidate_subset = prepared.select(candidate_rows)
            _, candidate_objective, candidate_gradient, _ = evaluate_prepared_batch(
                problem, candidate_subset, candidate, config.ridge, with_hessian=False
            )
            evaluations[indices[local]] += 1
            threshold = objective[local] + _ARMIJO_SUFFICIENT_INCREASE * steps[local] * derivative[local]
            objective_allowance = config.objective_tolerance * np.maximum(1.0, np.abs(objective[local]))
            numerical_refinement = (candidate_objective >= objective[local] - objective_allowance) & (
                np.linalg.norm(candidate_gradient, axis=1) < np.linalg.norm(gradient[local], axis=1)
            )
            accepted = np.isfinite(candidate_objective) & ((candidate_objective >= threshold) | numerical_refinement)
            if np.any(accepted):
                accepted_local = local[accepted]
                omega[indices[accepted_local]] = candidate[accepted]
                searching[accepted_local] = False
            rejected_local = local[~accepted]
            steps[rejected_local] *= _BACKTRACKING_FACTOR
        if np.any(searching):
            active[indices[searching]] = False

    return omega, converged, iterations, evaluations


def projected_gradient_residual(omega: FloatArray, gradient: FloatArray, radius: float) -> float:
    """Return a sample-count-stable stationarity residual on the ball."""

    scale = max(1.0, float(np.linalg.norm(gradient)))
    projected = project_ball(omega + gradient / scale, radius)
    return float(np.linalg.norm(projected - omega))


def solution_failures(problem: CompiledProblem, config: MLEConfig, omega: FloatArray,
        gradient: FloatArray) -> tuple[list[str], float]:  # fmt: skip
    """List every failed postcondition instead of silently accepting a result."""

    failures: list[str] = []
    if not np.all(np.isfinite(omega)):
        failures.append("non-finite coefficients")
        return failures, np.inf

    violation = max(0.0, float(np.linalg.norm(omega)) - problem.radius)
    residual = projected_gradient_residual(omega, gradient, problem.radius)
    if violation > config.constraint_tolerance:
        failures.append(f"ball violation {violation:.3e} > {config.constraint_tolerance:.3e}")
    if residual > config.projected_gradient_tolerance:
        failures.append(f"projected-gradient residual {residual:.3e} > {config.projected_gradient_tolerance:.3e}")
    return failures, residual


def solve_constrained(problem: CompiledProblem, prepared: PreparedLikelihood, config: MLEConfig,
        initial: FloatArray, *, prior_iterations: int = 0, prior_evaluations: int = 0) -> MLEResult:  # fmt: skip
    """Run the general constrained solver and enforce strict postconditions."""

    def objective(coefficients: FloatArray) -> tuple[float, FloatArray]:
        _, value, gradient = likelihood_and_gradient_prepared(problem, prepared, coefficients, config.ridge)
        return -value, -gradient

    def objective_hessian(coefficients: FloatArray) -> FloatArray:
        _, _, _, hessian = evaluate_prepared(problem, prepared, coefficients, config.ridge, with_hessian=True)
        if hessian is None:
            raise RuntimeError("internal error: Hessian was not computed")
        return -hessian

    slsqp_constraint = {"type": "ineq", "fun": lambda x: problem.radius**2 - float(x @ x), "jac": lambda x: -2.0 * x}
    trust_constraint = NonlinearConstraint(
        lambda x: problem.radius**2 - float(x @ x),
        0.0,
        np.inf,
        jac=lambda x: -2.0 * x,
        hess=lambda x, multiplier: -2.0 * float(multiplier[0]) * np.eye(x.size),
    )
    slsqp_starts = [initial]
    if np.linalg.norm(initial) > config.constraint_tolerance:
        # Boundary starts sometimes make SLSQP report a line-search failure on
        # tiny datasets. The interior restart remains deterministic and checked.
        slsqp_starts.append(np.zeros(problem.feature_dimension, dtype=np.float64))

    attempts = [("SLSQP", start) for start in slsqp_starts]
    attempts.append(("trust-constr", initial))

    attempt_failures: list[str] = []
    total_iterations = prior_iterations
    total_evaluations = prior_evaluations
    best_omega = initial
    best_residual = np.inf
    for attempt, (method, start) in enumerate(attempts, start=1):
        if method == "SLSQP":
            options = {"ftol": config.objective_tolerance, "maxiter": config.max_iterations, "disp": False}
            solver_arguments = {"constraints": (slsqp_constraint,)}
        else:
            options = {
                "gtol": config.trust_region_gradient_tolerance,
                "xtol": config.trust_region_step_tolerance,
                "barrier_tol": config.trust_region_barrier_tolerance,
                "maxiter": config.max_iterations,
                "verbose": 0,
            }
            solver_arguments = {"hess": objective_hessian, "constraints": (trust_constraint,)}
        result = minimize(objective, start, method=method, jac=True, options=options, **solver_arguments)
        total_iterations += int(result.nit)
        total_evaluations += int(result.nfev)
        omega = np.asarray(result.x, dtype=np.float64)
        log_likelihood, penalized, gradient = likelihood_and_gradient_prepared(problem, prepared, omega, config.ridge)
        failures, residual = solution_failures(problem, config, omega, gradient)
        if residual < best_residual:
            best_omega = omega
            best_residual = residual
        if not result.success:
            failures.insert(0, f"optimizer: {result.message}")
        if not failures:
            break
        attempt_failures.append(f"attempt {attempt} ({method}): {'; '.join(failures)}")
    else:
        fallback_newton = unconstrained_newton(problem, prepared, config, best_omega)
        total_iterations += fallback_newton.iterations
        total_evaluations += fallback_newton.function_evaluations
        newton_log_likelihood, newton_penalized, newton_gradient = likelihood_and_gradient_prepared(
            problem, prepared, fallback_newton.omega, config.ridge
        )
        newton_failures, newton_residual = solution_failures(problem, config, fallback_newton.omega, newton_gradient)
        if fallback_newton.converged and not newton_failures:
            return MLEResult(
                omega=fallback_newton.omega,
                log_likelihood=newton_log_likelihood,
                penalized_objective=newton_penalized,
                gradient=newton_gradient,
                projected_gradient_residual=newton_residual,
                iterations=total_iterations,
                function_evaluations=total_evaluations,
            )
        if np.linalg.norm(fallback_newton.omega) > problem.radius:
            best_omega = project_ball(fallback_newton.omega, problem.radius)
        attempt_failures.append(
            "exact unconstrained Newton: "
            + ("; ".join(newton_failures) if newton_failures else "optimizer did not converge")
        )
        try:
            return solve_boundary_kkt(
                problem,
                prepared,
                config,
                best_omega,
                prior_iterations=total_iterations,
                prior_evaluations=total_evaluations,
            )
        except ConvergenceError as error:
            attempt_failures.append(f"exact boundary KKT: {error}")
            raise ConvergenceError("MLE failed post-solve checks: " + " | ".join(attempt_failures)) from error

    return MLEResult(
        omega=omega,
        log_likelihood=log_likelihood,
        penalized_objective=penalized,
        gradient=gradient,
        projected_gradient_residual=residual,
        iterations=total_iterations,
        function_evaluations=total_evaluations,
    )


def solve_boundary_kkt(problem: CompiledProblem, prepared: PreparedLikelihood, config: MLEConfig,
        initial: FloatArray, *, prior_iterations: int, prior_evaluations: int) -> MLEResult:  # fmt: skip
    """Refine a boundary optimum by solving its exact KKT equations."""

    norm = float(np.linalg.norm(initial))
    if norm > config.constraint_tolerance:
        omega = problem.radius * initial / norm
    else:
        _, _, initial_gradient, _ = evaluate_prepared(problem, prepared, initial, config.ridge, with_hessian=False)
        gradient_norm = float(np.linalg.norm(initial_gradient))
        if gradient_norm <= config.projected_gradient_tolerance:
            raise ConvergenceError("cannot initialize boundary direction from a zero gradient")
        omega = problem.radius * initial_gradient / gradient_norm

    _, _, gradient, _ = evaluate_prepared(problem, prepared, omega, config.ridge, with_hessian=False)
    multiplier = max(0.0, float(omega @ gradient) / problem.radius**2)
    start = np.append(omega, multiplier)

    def equations(state: FloatArray) -> FloatArray:
        coefficients = state[:-1]
        lagrange_multiplier = float(state[-1])
        _, _, current_gradient, _ = evaluate_prepared(problem, prepared, coefficients, config.ridge, with_hessian=False)
        return np.append(
            current_gradient - lagrange_multiplier * coefficients, coefficients @ coefficients - problem.radius**2
        )

    def jacobian(state: FloatArray) -> FloatArray:
        coefficients = state[:-1]
        lagrange_multiplier = float(state[-1])
        _, _, _, hessian = evaluate_prepared(problem, prepared, coefficients, config.ridge, with_hessian=True)
        if hessian is None:
            raise RuntimeError("internal error: Hessian was not computed")
        dimension = coefficients.size
        result = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
        result[:dimension, :dimension] = hessian - lagrange_multiplier * np.eye(dimension)
        result[:dimension, -1] = -coefficients
        result[-1, :dimension] = 2.0 * coefficients
        result[-1, -1] = 0.0
        return result

    result = root(
        equations,
        start,
        jac=jacobian,
        method="hybr",
        options={
            "xtol": config.trust_region_step_tolerance,
            "maxfev": config.max_iterations * (problem.feature_dimension + 1),
        },
    )
    coefficients = np.asarray(result.x[:-1], dtype=np.float64)
    lagrange_multiplier = float(result.x[-1])
    log_likelihood, penalized, final_gradient = likelihood_and_gradient_prepared(
        problem, prepared, coefficients, config.ridge
    )
    failures, residual = solution_failures(problem, config, coefficients, final_gradient)
    kkt_residual = float(np.linalg.norm(equations(result.x), ord=np.inf))
    if kkt_residual > config.kkt_tolerance:
        failures.append(f"KKT equation residual {kkt_residual:.3e} > {config.kkt_tolerance:.3e}")
    if lagrange_multiplier < -config.constraint_tolerance:
        failures.append(f"negative KKT multiplier {lagrange_multiplier:.3e}")
    if failures:
        if not result.success:
            failures.insert(0, f"root solver: {result.message}")
        raise ConvergenceError("; ".join(failures))
    evaluations = int(result.nfev)
    return MLEResult(
        omega=coefficients,
        log_likelihood=log_likelihood,
        penalized_objective=penalized,
        gradient=final_gradient,
        projected_gradient_residual=residual,
        iterations=prior_iterations + evaluations,
        function_evaluations=prior_evaluations + evaluations,
    )
