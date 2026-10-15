"""Application boundary for the generic experiment layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from mairl.core.oracle import FollowerOracle
from mairl.core.problem import CompiledProblem, FloatArray

OracleFactory = Callable[[int], FollowerOracle]
EstimateEvaluator = Callable[[FloatArray, int], Mapping[str, FloatArray]]


@dataclass(frozen=True, slots=True)
class ApplicationRuntime:
    """Compiled application state consumed by a complete experiment."""

    name: str
    problem: CompiledProblem
    oracle_factory: OracleFactory
    evaluate_estimates: EstimateEvaluator
    metadata: Mapping[str, object]

    def evaluate(self, estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
        """Evaluate one estimate trace and validate application-produced arrays."""

        if estimates.ndim != 2 or estimates.shape[1] != self.problem.feature_dimension:
            raise ValueError(
                f"estimate trace must have shape (checkpoints, {self.problem.feature_dimension}); got {estimates.shape}"
            )
        metrics: dict[str, FloatArray] = {}
        for name, values in self.evaluate_estimates(estimates, seed).items():
            if not isinstance(name, str) or not name:
                raise ValueError("application metric names must be nonempty strings")
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (estimates.shape[0],):
                raise ValueError(
                    f"application metric {name!r} must have shape ({estimates.shape[0]},); got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"application metric {name!r} contains non-finite values")
            metrics[name] = array
        return metrics


@runtime_checkable
class ExperimentApplication(Protocol):
    """Factory implemented by each application outside the source packages."""

    def build(self, numerics: object) -> ApplicationRuntime:
        """Compile the problem and construct oracle and evaluation factories."""
