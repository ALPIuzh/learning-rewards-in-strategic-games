from __future__ import annotations

import numpy as np
import pytest

from mairl.core.math import (
    cumulative_fisher_information,
    cumulative_fisher_information_batch,
    fisher_information,
    follower_features,
    follower_probabilities,
    stable_softmax,
)
from mairl.core.problem import compile_problem
from mairl.exceptions import CompilationError


def test_compile_precomputes_identifiable_geometry(small_features: np.ndarray) -> None:
    problem = compile_problem(small_features, eta=1.2, radius=2.0)

    assert problem.identifiable_dimension == 2
    assert problem.pure_coverage_matrices.shape == (3, 2, 2)
    assert not problem.features.flags.writeable
    assert np.allclose(np.mean(problem.centered_pure_features, axis=1), 0.0)


def test_compile_rejects_unnormalized_features() -> None:
    features = np.zeros((2, 2, 2))
    features[0, 0] = [1.0, 1.0]
    with pytest.raises(CompilationError, match="unit-norm"):
        compile_problem(features, eta=1.0, radius=1.0)


def test_zero_identifiable_dimension_is_explicit() -> None:
    features = np.broadcast_to(np.array([[[0.2, -0.1]]]), (2, 3, 2)).copy()
    problem = compile_problem(features, eta=1.0, radius=1.0)
    assert problem.is_degenerate
    assert problem.identifiable_dimension == 0


def test_fisher_is_eta_squared_covariance(small_problem) -> None:
    policy = np.array([0.2, 0.3, 0.5])
    omega = np.array([0.4, -0.6])
    psi = follower_features(small_problem, policy)
    probabilities = follower_probabilities(small_problem, policy, omega)
    centered = psi - probabilities @ psi
    direct = small_problem.eta**2 * np.einsum("b,bd,be->de", probabilities, centered, centered)
    computed = fisher_information(small_problem, policy, omega)

    assert np.allclose(computed, direct, atol=1e-13)
    assert np.allclose(computed, computed.T, atol=1e-13)
    assert np.linalg.eigvalsh(computed)[0] >= -1e-13


def test_batched_cumulative_fisher_matches_scalar_and_direct_covariances(small_problem) -> None:
    """The seeded fast path must preserve the Fisher matrix exactly.

    This checks both an independent scalar implementation and the defining
    covariance formula.  It protects the scientific quantity while allowing its
    feature-projection implementation to be optimized.
    """

    rng = np.random.default_rng(2026)
    seed_count = 4
    sample_count = 11
    policies = rng.dirichlet(np.ones(small_problem.n_leader_actions), size=(seed_count, sample_count))
    omegas = rng.normal(size=(seed_count, small_problem.feature_dimension))

    batched = cumulative_fisher_information_batch(small_problem, policies, omegas)
    scalar = np.stack(
        [cumulative_fisher_information(small_problem, policies[index], omegas[index]) for index in range(seed_count)]
    )

    projected = np.einsum("sna,abd->snbd", policies, small_problem.features, optimize=True)
    probabilities = stable_softmax(small_problem.eta * np.einsum("snbd,sd->snb", projected, omegas, optimize=True))
    means = np.einsum("snb,snbd->snd", probabilities, projected, optimize=True)
    centered = projected - means[:, :, None, :]
    direct = small_problem.eta**2 * np.einsum("snb,snbd,snbe->sde", probabilities, centered, centered, optimize=True)

    np.testing.assert_allclose(batched, scalar, atol=1e-13)
    np.testing.assert_allclose(batched, direct, atol=1e-13)
    np.testing.assert_allclose(batched, np.swapaxes(batched, 1, 2), atol=1e-13)
    assert np.min(np.linalg.eigvalsh(batched)) >= -1e-12


def test_empty_batched_cumulative_fisher_has_one_zero_matrix_per_seed(small_problem) -> None:
    seed_count = 3
    policies = np.empty((seed_count, 0, small_problem.n_leader_actions), dtype=np.float64)
    omegas = np.zeros((seed_count, small_problem.feature_dimension))

    result = cumulative_fisher_information_batch(small_problem, policies, omegas)

    assert result.shape == (seed_count, small_problem.feature_dimension, small_problem.feature_dimension)
    np.testing.assert_array_equal(result, np.zeros_like(result))
