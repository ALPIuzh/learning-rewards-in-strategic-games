from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import mairl.inference._mle_solver as mle_solver
from mairl.core.math import project_ball
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import compile_problem
from mairl.exceptions import ConvergenceError
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, likelihood_and_gradient, solve_mle, solve_mle_batch


def test_batched_mle_matches_scalar_solutions(small_problem) -> None:
    seed_count = 5
    sample_count = 80
    true_omega = np.array([0.45, -0.65])
    oracle = SoftmaxFollowerOracle(small_problem, true_omega)
    policies = np.empty((seed_count, sample_count, 3))
    actions = np.empty((seed_count, sample_count), dtype=np.int64)
    for seed_index in range(seed_count):
        rng = np.random.default_rng(30 + seed_index)
        policies[seed_index] = rng.dirichlet(np.ones(3), size=sample_count)
        actions[seed_index] = [oracle.sample(policy, rng) for policy in policies[seed_index]]
    config = MLEConfig(ridge=1e-4)
    scalar = tuple(
        solve_mle(small_problem, InteractionDataset(policies[index], actions[index]), config)
        for index in range(seed_count)
    )
    batched = solve_mle_batch(small_problem, policies, actions, config)
    for scalar_result, batched_result in zip(scalar, batched, strict=True):
        np.testing.assert_allclose(batched_result.omega, scalar_result.omega, atol=1e-8)
        np.testing.assert_allclose(batched_result.gradient, scalar_result.gradient, atol=1e-8)
        assert batched_result.log_likelihood == pytest.approx(scalar_result.log_likelihood, abs=1e-10)
        assert batched_result.penalized_objective == pytest.approx(scalar_result.penalized_objective, abs=1e-10)
        assert batched_result.projected_gradient_residual <= (config.projected_gradient_tolerance)
        assert np.linalg.norm(batched_result.omega) <= (small_problem.radius + config.constraint_tolerance)


def test_batched_growing_mle_matches_independent_scalar_refits(small_problem) -> None:
    """Warm-started prefix refits reproduce independent scalar trajectories."""

    seed_count = 4
    sample_count = 24
    true_omega = np.array([0.45, -0.65])
    oracle = SoftmaxFollowerOracle(small_problem, true_omega)
    policies = np.empty((seed_count, sample_count, small_problem.n_leader_actions))
    actions = np.empty((seed_count, sample_count), dtype=np.int64)
    for seed_index in range(seed_count):
        rng = np.random.default_rng(300 + seed_index)
        policies[seed_index] = rng.dirichlet(np.ones(small_problem.n_leader_actions), size=sample_count)
        actions[seed_index] = [oracle.sample(policy, rng) for policy in policies[seed_index]]

    config = MLEConfig(ridge=1e-4)
    scalar_initials = np.zeros((seed_count, small_problem.feature_dimension))
    batch_initials = scalar_initials.copy()
    for prefix_size in (1, 2, 5, 11, sample_count):
        scalar = tuple(
            solve_mle(
                small_problem,
                InteractionDataset(policies[index, :prefix_size], actions[index, :prefix_size]),
                config,
                initial=scalar_initials[index],
            )
            for index in range(seed_count)
        )
        batched = solve_mle_batch(
            small_problem, policies[:, :prefix_size], actions[:, :prefix_size], config, initial=batch_initials
        )

        for scalar_result, batched_result in zip(scalar, batched, strict=True):
            np.testing.assert_allclose(
                batched_result.omega, scalar_result.omega, atol=2 * config.projected_gradient_tolerance
            )
            assert batched_result.penalized_objective == pytest.approx(scalar_result.penalized_objective, abs=1e-9)
        scalar_initials = np.stack([result.omega for result in scalar])
        batch_initials = np.stack([result.omega for result in batched])


def test_mle_does_not_silently_accept_nonconvergence(small_problem) -> None:
    policies = np.repeat(np.eye(3), repeats=40, axis=0)
    actions = np.tile(np.array([0, 1, 2]), 40)
    dataset = InteractionDataset(policies, actions)

    with pytest.raises(ConvergenceError, match="post-solve checks"):
        solve_mle(small_problem, dataset, MLEConfig(ridge=1e-4, max_iterations=1, projected_gradient_tolerance=1e-12))


def test_newton_accepts_gradient_improving_roundoff_step(small_problem, monkeypatch: pytest.MonkeyPatch) -> None:
    objective = -453.0
    initial_gradient = np.array([1.166e-6, 0.0])

    def evaluate(problem, prepared, coefficients, ridge, *, with_hessian):
        del problem, prepared, ridge
        if np.array_equal(coefficients, np.zeros(2)):
            gradient = initial_gradient
            value = objective
        else:
            gradient = np.zeros(2)
            value = objective - 2e-13
        hessian = -np.eye(2) if with_hessian else None
        return value, value, gradient, hessian

    monkeypatch.setattr(mle_solver, "evaluate_prepared", evaluate)
    result = mle_solver.unconstrained_newton(small_problem, SimpleNamespace(), MLEConfig(ridge=1e-4), np.zeros(2))

    assert result.converged
    assert result.iterations == 1
    assert np.array_equal(result.omega, initial_gradient)


@pytest.mark.parametrize("ridge", [0.0, 1.0e-4])
def test_newton_matches_slsqp_and_uses_exact_boundary_solution(small_problem, ridge: float) -> None:
    rng = np.random.default_rng(22)
    oracle = SoftmaxFollowerOracle(small_problem, np.array([0.45, -0.65]))
    policies = np.repeat(np.eye(3), repeats=150, axis=0)
    actions = np.array([oracle.sample(policy, rng) for policy in policies])
    dataset = InteractionDataset(policies, actions)
    unconstrained = solve_mle(small_problem, dataset, MLEConfig(ridge=ridge, solver="newton"))
    boundary_problem = compile_problem(small_problem.features, eta=small_problem.eta, radius=0.1)
    newton = solve_mle(boundary_problem, dataset, MLEConfig(ridge=ridge, solver="newton"))
    slsqp = solve_mle(boundary_problem, dataset, MLEConfig(ridge=ridge, solver="slsqp"))
    projected = project_ball(unconstrained.omega, boundary_problem.radius)
    projected_objective = likelihood_and_gradient(boundary_problem, dataset, projected, ridge=ridge)[1]

    assert np.isclose(np.linalg.norm(newton.omega), boundary_problem.radius, atol=1e-8)
    assert np.allclose(newton.omega, slsqp.omega, atol=1e-7)
    assert np.isclose(newton.penalized_objective, slsqp.penalized_objective, atol=1e-11)
    assert newton.penalized_objective > projected_objective + 1e-6


def test_exact_hessian_fallback_meets_strict_boundary_residual(small_problem) -> None:
    policies = np.repeat(np.eye(3), repeats=6, axis=0)
    actions = np.zeros(policies.shape[0], dtype=np.int64)
    dataset = InteractionDataset(policies, actions)
    result = solve_mle(
        small_problem,
        dataset,
        MLEConfig(
            ridge=1e-4,
            solver="slsqp",
            projected_gradient_tolerance=1e-6,
            trust_region_gradient_tolerance=1e-12,
            trust_region_step_tolerance=1e-14,
            trust_region_barrier_tolerance=1e-12,
        ),
    )

    assert np.linalg.norm(result.omega) <= small_problem.radius + 1e-9
    assert result.projected_gradient_residual <= 1e-6


def test_exact_kkt_refinement_recovers_from_generic_solver_failures(small_problem, monkeypatch) -> None:
    policies = np.repeat(np.eye(3), repeats=8, axis=0)
    actions = np.zeros(policies.shape[0], dtype=np.int64)
    dataset = InteractionDataset(policies, actions)
    boundary_problem = compile_problem(small_problem.features, eta=small_problem.eta, radius=0.1)

    def fail_minimize(_objective, initial, **_kwargs):
        return SimpleNamespace(
            x=np.asarray(initial), success=False, message="deliberate generic-solver failure", nit=1, nfev=1
        )

    monkeypatch.setattr(mle_solver, "minimize", fail_minimize)
    result = solve_mle(
        boundary_problem,
        dataset,
        MLEConfig(ridge=1e-4, solver="slsqp", projected_gradient_tolerance=1e-6, trust_region_step_tolerance=1e-12),
    )

    assert np.isclose(np.linalg.norm(result.omega), boundary_problem.radius, atol=1e-9)
    assert result.projected_gradient_residual <= 1e-6
