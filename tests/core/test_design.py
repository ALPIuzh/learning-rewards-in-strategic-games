from __future__ import annotations

import numpy as np
import pytest

from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig, _support_reduce, compute_warmup_design


def test_warmup_design_certifies_and_rounds(small_problem) -> None:
    config = WarmupConfig(budget=80, design_tolerance=0.1)
    result = compute_warmup_design(small_problem, config)

    dimension = small_problem.identifiable_dimension
    assert result.diagnostics.final_gap <= config.design_tolerance * dimension
    assert result.support_size <= dimension * (dimension + 1) // 2 + 1
    assert result.allocation_counts.sum() == config.budget
    assert np.all(result.allocation_counts >= 1)
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.linalg.eigvalsh(result.coverage_matrix)[0] > 0


def test_warmup_budget_violation_raises(small_problem) -> None:
    with pytest.raises(ConfigurationError, match="budget is too small"):
        compute_warmup_design(small_problem, WarmupConfig(budget=2, design_tolerance=0.1))


def test_constructive_support_reduction_preserves_coverage() -> None:
    rng = np.random.default_rng(8)
    factors = rng.normal(size=(7, 2, 2))
    matrices = np.einsum("aij,akj->aik", factors, factors)
    weights = rng.dirichlet(np.ones(7))
    before = np.einsum("a,aij->ij", weights, matrices)

    reduced = _support_reduce(matrices, weights, support_tolerance=1e-12, matrix_tolerance=1e-9)
    after = np.einsum("a,aij->ij", reduced, matrices)

    assert np.count_nonzero(reduced > 1e-12) <= 4
    assert np.allclose(before, after, atol=1e-9)
    assert np.isclose(reduced.sum(), 1.0)
