"""Configuration and result values shared by the MLE implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mairl.core.problem import FloatArray
from mairl.exceptions import ConfigurationError

MLESolver = Literal["newton", "slsqp"]


@dataclass(frozen=True, slots=True)
class MLEConfig:
    """Constrained maximum-likelihood solver settings.

    ``ridge`` is required deliberately: callers must choose and record its value.
    Set it to zero for the unregularized estimator used in the theory; positive
    values define a different, ridge-regularized estimator.
    The ``newton`` backend is an exact hybrid: an unconstrained interior optimum
    is accepted, while an exterior optimum only initializes the constrained solve.
    """

    ridge: float
    solver: MLESolver = "newton"
    objective_tolerance: float = 1e-13
    projected_gradient_tolerance: float = 1e-6
    constraint_tolerance: float = 1e-9
    max_iterations: int = 1_000
    trust_region_gradient_tolerance: float = 1e-8
    trust_region_step_tolerance: float = 1e-12
    trust_region_barrier_tolerance: float = 1e-12
    kkt_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.solver not in ("newton", "slsqp"):
            raise ConfigurationError("solver must be 'newton' or 'slsqp'")
        if self.ridge < 0:
            raise ConfigurationError(f"ridge must be nonnegative; got {self.ridge!r}")
        values = {
            "objective_tolerance": self.objective_tolerance,
            "projected_gradient_tolerance": self.projected_gradient_tolerance,
            "constraint_tolerance": self.constraint_tolerance,
            "trust_region_gradient_tolerance": self.trust_region_gradient_tolerance,
            "trust_region_step_tolerance": self.trust_region_step_tolerance,
            "trust_region_barrier_tolerance": self.trust_region_barrier_tolerance,
            "kkt_tolerance": self.kkt_tolerance,
        }
        for name, value in values.items():
            if not value > 0:
                raise ConfigurationError(f"{name} must be strictly positive; got {value!r}")
        if self.max_iterations < 1:
            raise ConfigurationError("max_iterations must be at least 1")


@dataclass(frozen=True, slots=True)
class MLEResult:
    """A verified optimum and the diagnostics needed to audit it."""

    omega: FloatArray
    log_likelihood: float
    penalized_objective: float
    gradient: FloatArray
    projected_gradient_residual: float
    iterations: int
    function_evaluations: int
