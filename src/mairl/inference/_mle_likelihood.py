"""Prepared multinomial-logit likelihood algebra for the MLE solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from mairl.core.problem import CompiledProblem, FloatArray
from mairl.exceptions import ConfigurationError
from mairl.inference.data import InteractionDataset


@dataclass(frozen=True, slots=True)
class PreparedLikelihood:
    """Policy-projected features; carries a leading seed axis when prepared in batch."""

    follower_features: FloatArray
    selected_features: FloatArray

    def select(self, seeds: int | np.ndarray) -> PreparedLikelihood:
        """Select one seed, or a subset of seeds, along the leading axis."""

        return PreparedLikelihood(self.follower_features[seeds], self.selected_features[seeds])


def validate_dataset(problem: CompiledProblem, dataset: InteractionDataset) -> None:
    if dataset.policies.shape[1] != problem.n_leader_actions:
        raise ConfigurationError("dataset leader-action dimension does not match problem")
    if dataset.size and (
        np.min(dataset.follower_actions) < 0 or np.max(dataset.follower_actions) >= problem.n_follower_actions
    ):
        raise ConfigurationError("dataset contains an invalid follower-action index")


def prepare_likelihood(problem: CompiledProblem, dataset: InteractionDataset) -> PreparedLikelihood:
    follower_features = np.einsum("na,abd->nbd", dataset.policies, problem.features, optimize=True)
    rows = np.arange(dataset.size)
    return PreparedLikelihood(
        follower_features=follower_features, selected_features=follower_features[rows, dataset.follower_actions]
    )


def prepare_likelihood_batch(problem: CompiledProblem, policies: FloatArray,
        follower_actions: np.ndarray) -> PreparedLikelihood:  # fmt: skip
    follower_features = np.einsum("sna,abd->snbd", policies, problem.features, optimize=True)
    seed_rows = np.arange(policies.shape[0])[:, None]
    sample_rows = np.arange(policies.shape[1])[None, :]
    return PreparedLikelihood(
        follower_features=follower_features,
        selected_features=follower_features[seed_rows, sample_rows, follower_actions],
    )


def evaluate_prepared(problem: CompiledProblem, prepared: PreparedLikelihood, coefficients: FloatArray,
        ridge: float, *, with_hessian: bool) -> tuple[float, float, FloatArray, FloatArray | None]:  # fmt: skip
    """Evaluate the scalar objective, gradient, and optionally exact Hessian."""

    logits = problem.eta * np.einsum("nbd,d->nb", prepared.follower_features, coefficients, optimize=True)
    log_normalizers = logsumexp(logits, axis=1)
    log_likelihood = float(np.sum(problem.eta * (prepared.selected_features @ coefficients) - log_normalizers))
    probabilities = np.exp(logits - log_normalizers[:, None])
    expected = np.einsum("nb,nbd->nd", probabilities, prepared.follower_features)
    gradient = problem.eta * np.sum(prepared.selected_features - expected, axis=0) - ridge * coefficients
    penalized = log_likelihood - 0.5 * ridge * float(coefficients @ coefficients)
    if not with_hessian:
        return log_likelihood, penalized, gradient, None

    centered = prepared.follower_features - expected[:, None, :]
    information = problem.eta**2 * np.einsum("nb,nbd,nbe->de", probabilities, centered, centered, optimize=True)
    hessian = -information - ridge * np.eye(coefficients.size)
    return log_likelihood, penalized, gradient, hessian


def evaluate_prepared_batch(problem: CompiledProblem,
        prepared: PreparedLikelihood, coefficients: FloatArray, ridge: float,
        *, with_hessian: bool) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:  # fmt: skip
    """Evaluate independent seeded objectives through one vectorized contraction."""

    logits = problem.eta * np.einsum("snbd,sd->snb", prepared.follower_features, coefficients, optimize=True)
    log_normalizers = logsumexp(logits, axis=2)
    selected_values = np.einsum("snd,sd->sn", prepared.selected_features, coefficients, optimize=True)
    log_likelihood = np.sum(problem.eta * selected_values - log_normalizers, axis=1)
    probabilities = np.exp(logits - log_normalizers[:, :, None])
    expected = np.einsum("snb,snbd->snd", probabilities, prepared.follower_features, optimize=True)
    gradient = problem.eta * np.sum(prepared.selected_features - expected, axis=1) - ridge * coefficients
    penalized = log_likelihood - 0.5 * ridge * np.einsum("sd,sd->s", coefficients, coefficients, optimize=True)
    if not with_hessian:
        return log_likelihood, penalized, gradient, None

    centered = prepared.follower_features - expected[:, :, None, :]
    information = problem.eta**2 * np.einsum("snb,snbd,snbe->sde", probabilities, centered, centered, optimize=True)
    hessian = -information - ridge * np.eye(coefficients.shape[1])[None, :, :]
    return log_likelihood, penalized, gradient, hessian


def likelihood_and_gradient_prepared(problem: CompiledProblem, prepared: PreparedLikelihood,
        coefficients: FloatArray, ridge: float) -> tuple[float, float, FloatArray]:  # fmt: skip
    log_likelihood, penalized, gradient, _ = evaluate_prepared(
        problem, prepared, coefficients, ridge, with_hessian=False
    )
    return log_likelihood, penalized, gradient
