from __future__ import annotations

import numpy as np

from mairl.exploration.policy import information_gain_and_gradient
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import likelihood_and_gradient, likelihood_gradient_and_hessian
from mairl.planning.leader import leader_value_and_gradient


def _finite_difference(function, point: np.ndarray, step: float = 1e-6) -> np.ndarray:
    basis = np.eye(point.size)
    return np.array(
        [
            (function(point + step * direction) - function(point - step * direction)) / (2.0 * step)
            for direction in basis
        ]
    )


def test_likelihood_gradient(small_problem) -> None:
    policies = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 1.0, 0.0]])
    dataset = InteractionDataset(policies, np.array([0, 2, 1]))
    point = np.array([0.3, -0.4])
    ridge = 2e-4
    _, _, gradient = likelihood_and_gradient(small_problem, dataset, point, ridge=ridge)
    finite = _finite_difference(
        lambda omega: likelihood_and_gradient(small_problem, dataset, omega, ridge=ridge)[1], point
    )
    assert np.allclose(gradient, finite, atol=2e-8)


def test_likelihood_hessian(small_problem) -> None:
    policies = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 1.0, 0.0]])
    dataset = InteractionDataset(policies, np.array([0, 2, 1]))
    point = np.array([0.3, -0.4])
    ridge = 2e-4
    _, _, _, hessian = likelihood_gradient_and_hessian(small_problem, dataset, point, ridge=ridge)
    finite = _finite_difference(
        lambda omega: likelihood_and_gradient(small_problem, dataset, omega, ridge=ridge)[2], point
    ).T
    assert np.allclose(hessian, finite, atol=2e-8)


def test_information_gain_gradient(small_problem) -> None:
    omega = np.array([0.3, -0.5])
    policy = np.array([0.25, 0.35, 0.40])
    inverse_information = np.linalg.inv(np.array([[2.0, 0.2], [0.2, 1.3]]))
    _, gradient = information_gain_and_gradient(small_problem, omega, inverse_information, policy)
    finite = _finite_difference(
        lambda mu: information_gain_and_gradient(small_problem, omega, inverse_information, mu)[0], policy
    )
    assert np.allclose(gradient, finite, atol=2e-8)


def test_leader_gradient(small_problem, small_leader_reward) -> None:
    omega = np.array([0.3, -0.5])
    policy = np.array([0.25, 0.35, 0.40])
    reward = small_leader_reward
    _, gradient = leader_value_and_gradient(small_problem, omega, reward, policy)
    finite = _finite_difference(lambda mu: leader_value_and_gradient(small_problem, omega, reward, mu)[0], policy)
    assert np.allclose(gradient, finite, atol=2e-8)
