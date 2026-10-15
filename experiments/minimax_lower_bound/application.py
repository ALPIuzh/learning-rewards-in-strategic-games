"""The hard Stackelberg family used by the minimax lower-bound experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from mairl.core.math import stable_softmax
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import CompiledProblem, FloatArray, compile_problem
from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics


@dataclass(frozen=True, slots=True)
class MinimaxInstance:
    """One fully specified hard instance and its scientific parameters."""

    problem: CompiledProblem
    true_omega: FloatArray
    leader_reward: FloatArray
    active_coordinate: int
    epsilon: float
    total_samples: int


def coordinate_for_seed(seed: int, dimension: int) -> int:
    """Select a reproducible coordinate for one-off validation instances.

    Final experiments use the balanced design in :mod:`experiments.minimax_lower_bound.study`.
    This domain-separated stream prevents validation-coordinate selection from reusing the
    algorithm's random-number stream.
    """

    if seed < 0:
        raise ConfigurationError("seed must be nonnegative")
    if dimension < 1:
        raise ConfigurationError("dimension must be positive")
    sequence = np.random.SeedSequence([0x4D414952, seed, dimension])
    return int(np.random.default_rng(sequence).integers(dimension))


def make_minimax_instance(dimension: int, total_samples: int, seed: int,
        *, active_coordinate: int | None = None) -> MinimaxInstance:  # fmt: skip
    """Construct ``A=[d]``, ``B={b1,b2}`` exactly as specified in I.3."""

    if dimension < 32:
        raise ConfigurationError("the theorem-valid instance requires dimension >= 32")
    if total_samples < 4 * dimension:
        raise ConfigurationError("the theorem-valid instance requires K >= 4d")
    coordinate = coordinate_for_seed(seed, dimension) if active_coordinate is None else int(active_coordinate)
    if not 0 <= coordinate < dimension:
        raise ConfigurationError("active_coordinate must lie in [0, dimension)")

    features = np.zeros((dimension, 2, dimension), dtype=np.float64)
    features[:, 1, :] = np.eye(dimension, dtype=np.float64)
    leader_reward = np.zeros((dimension, 2), dtype=np.float64)
    leader_reward[:, 1] = 1.0
    epsilon = 0.5 * np.sqrt(dimension / total_samples)
    true_omega = np.zeros(dimension, dtype=np.float64)
    true_omega[coordinate] = epsilon
    return MinimaxInstance(
        problem=compile_problem(features, eta=1.0, radius=1.0),
        true_omega=true_omega,
        leader_reward=leader_reward,
        active_coordinate=coordinate,
        epsilon=float(epsilon),
        total_samples=total_samples,
    )


def exact_hard_instance_metrics(instance: MinimaxInstance, estimates: FloatArray) -> dict[str, FloatArray]:
    """Evaluate deployment and transfer metrics from the hard instance's closed form.

    For an estimate ``omega_hat``, an optimal deployed leader policy is the vertex
    ``e_argmax(omega_hat)``. The true optimum is ``e_i`` for the active coordinate
    ``i``. For a vertex ``e_j``, the Fisher matrix at ``omega_hat`` is
    ``sigma(omega_hat_j) (1-sigma(omega_hat_j)) e_j e_j^T``.
    """

    values = np.asarray(estimates, dtype=np.float64)
    dimension = instance.problem.feature_dimension
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ConfigurationError(f"estimates must have shape (checkpoints, {dimension})")
    if not np.all(np.isfinite(values)):
        raise ConfigurationError("estimates contain non-finite values")

    rows = np.arange(values.shape[0])
    deployed_coordinates = np.argmax(values, axis=1)
    deployed_is_optimal = deployed_coordinates == instance.active_coordinate
    deployed_true_logits = instance.epsilon * deployed_is_optimal.astype(np.float64)
    deployed_values = stable_softmax(np.column_stack((np.zeros(values.shape[0]), deployed_true_logits)))[:, 1]
    optimal_value = float(stable_softmax(np.asarray([0.0, instance.epsilon]))[1])

    errors = instance.true_omega[None, :] - values
    true_logits = values[:, instance.active_coordinate]
    deployed_logits = values[rows, deployed_coordinates]
    true_probabilities = stable_softmax(np.column_stack((np.zeros(values.shape[0]), true_logits)))[:, 1]
    deployed_probabilities = stable_softmax(np.column_stack((np.zeros(values.shape[0]), deployed_logits)))[:, 1]
    true_policy_norm = np.abs(errors[:, instance.active_coordinate]) * np.sqrt(
        true_probabilities * (1.0 - true_probabilities)
    )
    deployed_policy_norm = np.abs(errors[rows, deployed_coordinates]) * np.sqrt(
        deployed_probabilities * (1.0 - deployed_probabilities)
    )

    return {
        "stackelberg_suboptimality": np.maximum(0.0, optimal_value - deployed_values),
        "fisher_transfer_error": np.maximum(true_policy_norm, deployed_policy_norm),
        "coefficient_l2_error": np.linalg.norm(errors, axis=1),
    }


def build_instance_runtime(instance: MinimaxInstance, numerics: ExperimentNumerics) -> ApplicationRuntime:
    """Build a runtime for one fixed hard instance."""

    del numerics  # Estimation numerics are consumed by the generic experiment executor.

    def oracle_factory(_: int) -> SoftmaxFollowerOracle:
        return SoftmaxFollowerOracle(instance.problem, instance.true_omega)

    def evaluate_estimates(estimates: FloatArray, _: int) -> dict[str, FloatArray]:
        return exact_hard_instance_metrics(instance, estimates)

    return ApplicationRuntime(
        name="minimax-lower-bound",
        problem=instance.problem,
        oracle_factory=oracle_factory,
        evaluate_estimates=evaluate_estimates,
        metadata=_instance_metadata(instance),
    )


def build_family_runtime(dimension: int, total_samples: int, coordinates_by_seed: Mapping[int, int],
        numerics: ExperimentNumerics) -> ApplicationRuntime:  # fmt: skip
    """Build one seed-aware runtime shared by a vectorized ``(d, K)`` batch."""

    del numerics
    coordinates = {int(seed): int(coordinate) for seed, coordinate in coordinates_by_seed.items()}
    if not coordinates:
        raise ConfigurationError("coordinates_by_seed must be nonempty")
    if any(seed < 0 for seed in coordinates):
        raise ConfigurationError("all execution seeds must be nonnegative")
    if any(not 0 <= coordinate < dimension for coordinate in coordinates.values()):
        raise ConfigurationError("all active coordinates must lie in [0, dimension)")

    template = make_minimax_instance(dimension, total_samples, seed=0, active_coordinate=0)

    def instance_for_seed(seed: int) -> MinimaxInstance:
        try:
            coordinate = coordinates[seed]
        except KeyError as error:
            raise ConfigurationError(f"seed {seed} is absent from the balanced design") from error
        true_omega = np.zeros(dimension, dtype=np.float64)
        true_omega[coordinate] = template.epsilon
        return MinimaxInstance(
            problem=template.problem,
            true_omega=true_omega,
            leader_reward=template.leader_reward,
            active_coordinate=coordinate,
            epsilon=template.epsilon,
            total_samples=total_samples,
        )

    def oracle_factory(seed: int) -> SoftmaxFollowerOracle:
        instance = instance_for_seed(seed)
        return SoftmaxFollowerOracle(instance.problem, instance.true_omega)

    def evaluate_estimates(estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
        return exact_hard_instance_metrics(instance_for_seed(seed), estimates)

    return ApplicationRuntime(
        name="minimax-lower-bound",
        problem=template.problem,
        oracle_factory=oracle_factory,
        evaluate_estimates=evaluate_estimates,
        metadata={
            "dimension": dimension,
            "total_samples": total_samples,
            "epsilon": template.epsilon,
            "eta": 1.0,
            "radius": 1.0,
            "leader_reward": "1{b=b2}",
            "coordinate_design": "balanced",
        },
    )


def _instance_metadata(instance: MinimaxInstance) -> dict[str, object]:
    return {
        "dimension": instance.problem.feature_dimension,
        "total_samples": instance.total_samples,
        "active_coordinate": instance.active_coordinate,
        "epsilon": instance.epsilon,
        "eta": 1.0,
        "radius": 1.0,
        "leader_reward": "1{b=b2}",
    }


@dataclass(frozen=True, slots=True)
class MinimaxLowerBoundApplication:
    """Hydra application adapter; the study creates each grid family per batch."""

    validation_dimension: int = 33
    validation_total_samples: int = 132
    validation_seed: int = 92000

    def build(self, numerics: object) -> ApplicationRuntime:
        if not isinstance(numerics, ExperimentNumerics):
            raise TypeError("MinimaxLowerBoundApplication requires ExperimentNumerics")
        instance = make_minimax_instance(self.validation_dimension, self.validation_total_samples, self.validation_seed)
        return build_instance_runtime(instance, numerics)
