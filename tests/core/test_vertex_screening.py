"""Vertex-screening correctness: does bounding dense (n_actions, n_actions) multistart
construction to a closed-form-scored top-K subset ever silently degrade optimization
quality?

The screened start set is a strict subset of the brute-force (np.eye(n_actions)) start
set -- same uniform start, same random starts (identical rng draws), fewer vertex
starts. So brute_force_value >= screened_value always, by construction; the empirical
question is how much smaller, across many random instances, once screening actually
activates. _DENSE_VERTEX_LIMIT is patched low so this is testable on small, fast
problems instead of requiring an actually-huge n_actions.
"""

from __future__ import annotations

import numpy as np
import pytest

import mairl.exploration.policy as policy_module
import mairl.exploration.simplex as simplex_module
import mairl.planning.leader as leader_module
from mairl.core.problem import compile_problem
from mairl.exploration.policy import (
    information_gain_and_gradient,
    select_information_policy,
    select_information_policy_batch,
)
from mairl.exploration.simplex import PolicyOptimizationConfig, maximize_on_simplex
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)

_PATCHED_LIMIT = 4
_N_ACTIONS = 30
_N_TRIALS = 30


@pytest.fixture
def patched_limit(monkeypatch):
    """Lower _DENSE_VERTEX_LIMIT everywhere it was imported, so the screened code path
    activates on small, fast-to-brute-force problems instead of requiring huge ones."""

    monkeypatch.setattr(simplex_module, "_DENSE_VERTEX_LIMIT", _PATCHED_LIMIT)
    monkeypatch.setattr(leader_module, "_DENSE_VERTEX_LIMIT", _PATCHED_LIMIT)
    monkeypatch.setattr(policy_module, "_DENSE_VERTEX_LIMIT", _PATCHED_LIMIT)


def _random_problem(seed: int, n_actions: int = _N_ACTIONS, n_follower: int = 6, d: int = 4):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_actions, n_follower, d))
    features /= np.maximum(1.0, 1.1 * np.linalg.norm(features, axis=2, keepdims=True))
    problem = compile_problem(features, eta=1.25, radius=2.0)
    omega = rng.normal(size=d)
    omega /= max(1.0, float(np.linalg.norm(omega)) / 2.0)
    return problem, omega, rng


def _assert_gaps_negligible(gaps: list[float], label: str) -> None:
    """Screening may only cost a negligible amount of objective value.

    The distribution is printed so a failure shows how far off it was, not just
    that a threshold tripped.
    """

    values = np.array(gaps)
    nonzero = np.sum(values > 1e-9)
    print(f"\n{label}: mean={values.mean():.2e} max={values.max():.2e} nonzero={nonzero}/{_N_TRIALS}")
    assert values.max() < 1e-3, f"worst-case quality loss from screening is too large: {values.max():.3e}"
    assert values.mean() < 1e-4, f"average quality loss from screening is too large: {values.mean():.3e}"


@pytest.mark.usefixtures("patched_limit")
def test_leader_vertex_screening_gap_is_negligible() -> None:
    gaps = []
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=6, batch_refine_starts=2)
    )
    for seed in range(_N_TRIALS):
        problem, omega, _ = _random_problem(seed)
        leader_reward = np.random.default_rng(seed + 10_000).normal(size=(_N_ACTIONS, 6))

        def objective(policy, _omega=omega, _reward=leader_reward, _problem=problem):
            return leader_value_and_gradient(_problem, _omega, _reward, policy)

        brute_force = maximize_on_simplex(
            objective,
            n_actions=_N_ACTIONS,
            config=config.policy,
            rng=np.random.default_rng(seed + 20_000),
            vertex_scores=None,
        )
        screened_result = optimize_leader_policy(problem, omega, leader_reward, config, seed=seed + 20_000)
        assert screened_result.optimization.attempted_starts < 1 + _N_ACTIONS, (
            "screening should have actually reduced the number of attempted starts"
        )
        gap = brute_force.objective - screened_result.value
        assert gap >= -1e-9, "screened value exceeded brute force -- not a subset anymore"
        gaps.append(gap)

    _assert_gaps_negligible(gaps, "leader gaps")


@pytest.mark.usefixtures("patched_limit")
def test_information_gain_vertex_screening_gap_is_negligible() -> None:
    gaps = []
    config = PolicyOptimizationConfig(method="multistart", random_starts=6, batch_refine_starts=2)
    for seed in range(_N_TRIALS):
        problem, omega, _ = _random_problem(seed)
        d = problem.feature_dimension
        information_matrix = np.eye(d) * 3.0 + 0.1 * np.abs(np.random.default_rng(seed + 30_000).normal(size=(d, d)))
        information_matrix = (information_matrix + information_matrix.T) / 2.0
        inverse = np.linalg.inv(information_matrix)

        def objective(policy, _problem=problem, _omega=omega, _inv=inverse):
            return information_gain_and_gradient(_problem, _omega, _inv, policy)

        brute_force = maximize_on_simplex(
            objective, n_actions=_N_ACTIONS, config=config, rng=np.random.default_rng(seed + 40_000), vertex_scores=None
        )
        screened_result = select_information_policy(
            problem, omega, information_matrix, config, np.random.default_rng(seed + 40_000)
        )
        assert screened_result.optimization.attempted_starts < 1 + _N_ACTIONS
        gap = brute_force.objective - screened_result.information_gain
        assert gap >= -1e-9
        gaps.append(gap)

    _assert_gaps_negligible(gaps, "info-gain gaps")


@pytest.mark.usefixtures("patched_limit")
def test_leader_batch_matches_sequential_under_screening() -> None:
    """The batched leader optimizer's per-problem screening must match calling the
    (already-verified) sequential screened optimizer once per problem."""

    n_problems = 5
    problem, _, _ = _random_problem(0)
    rng = np.random.default_rng(999)
    omegas = rng.normal(size=(n_problems, problem.feature_dimension))
    leader_reward = rng.normal(size=(_N_ACTIONS, 6))
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=6, batch_refine_starts=2)
    )
    seeds = tuple(5000 + index for index in range(n_problems))

    sequential = [
        optimize_leader_policy(problem, omegas[index], leader_reward, config, seed=seeds[index])
        for index in range(n_problems)
    ]
    batched = optimize_leader_policy_batch(problem, omegas, leader_reward, config, seeds)

    assert np.allclose([r.value for r in batched], [r.value for r in sequential], atol=1e-8)


@pytest.mark.usefixtures("patched_limit")
def test_information_gain_batch_matches_sequential_under_screening() -> None:
    n_problems = 5
    problem, _, _ = _random_problem(1)
    d = problem.feature_dimension
    rng = np.random.default_rng(998)
    omegas = rng.normal(size=(n_problems, d))
    base = np.eye(d) * 3.0
    information_matrices = np.broadcast_to(base, (n_problems, d, d)).copy()
    config = PolicyOptimizationConfig(method="multistart", random_starts=6, batch_refine_starts=2)
    seeds = tuple(6000 + index for index in range(n_problems))
    rngs = tuple(np.random.default_rng(seed) for seed in seeds)
    rngs_for_sequential = tuple(np.random.default_rng(seed) for seed in seeds)

    sequential = [
        select_information_policy(
            problem, omegas[index], information_matrices[index], config, rngs_for_sequential[index]
        )
        for index in range(n_problems)
    ]
    batched = select_information_policy_batch(problem, omegas, information_matrices, config, rngs)

    assert np.allclose([r.information_gain for r in batched], [r.information_gain for r in sequential], atol=1e-8)


def test_default_threshold_unaffected_for_small_problems() -> None:
    """Without patching, _DENSE_VERTEX_LIMIT (512) is far above every n_actions used
    here, so passing vertex_scores must be a complete no-op vs. not passing it."""

    problem, omega, rng = _random_problem(0, n_actions=40)
    leader_reward = rng.normal(size=(40, 6))
    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=4, batch_refine_starts=2)
    )
    with_scores = optimize_leader_policy(problem, omega, leader_reward, config, seed=42)

    def objective(policy, _omega=omega, _reward=leader_reward, _problem=problem):
        return leader_value_and_gradient(_problem, _omega, _reward, policy)

    without_scores = maximize_on_simplex(
        objective, n_actions=40, config=config.policy, rng=np.random.default_rng(42), vertex_scores=None
    )
    assert with_scores.optimization.attempted_starts == without_scores.attempted_starts
    assert np.isclose(with_scores.value, without_scores.objective, atol=1e-12)
