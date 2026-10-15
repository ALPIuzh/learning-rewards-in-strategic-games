"""Static problem validation and compilation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mairl.exceptions import CompilationError, ConfigurationError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CompilationConfig:
    """Validation and numerical-rank rules applied by :func:`compile_problem`."""

    rank_relative_tolerance: float = 1e-10
    rank_absolute_tolerance: float = 1e-12
    feature_norm_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        values = {
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "rank_absolute_tolerance": self.rank_absolute_tolerance,
            "feature_norm_tolerance": self.feature_norm_tolerance,
        }
        for name, value in values.items():
            if value < 0:
                raise ConfigurationError(f"{name} must be nonnegative; got {value!r}")


def _readonly(array: ArrayLike) -> FloatArray:
    array = np.ascontiguousarray(array, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CompiledProblem:
    """Validated model plus feature geometry cached for repeated experiments."""

    features: FloatArray
    eta: float
    radius: float
    identifiable_basis: FloatArray
    centered_pure_features: FloatArray
    reduced_centered_pure_features: FloatArray
    pure_coverage_matrices: FloatArray
    rank_threshold: float
    max_feature_norm: float

    @property
    def n_leader_actions(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_follower_actions(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[2])

    @property
    def identifiable_dimension(self) -> int:
        return int(self.identifiable_basis.shape[1])

    @property
    def is_degenerate(self) -> bool:
        return self.identifiable_dimension == 0


def compile_problem(features: ArrayLike, *, eta: float, radius: float,
        config: CompilationConfig | None = None) -> CompiledProblem:  # fmt: skip
    """Validate once and precompute the identifiable feature geometry.

    This is the package's compilation boundary. Runtime algorithms accept only a
    :class:`CompiledProblem`, so shape, finiteness, normalization, and numerical rank
    are not repeatedly checked in hot loops.
    """

    config = config or CompilationConfig()
    phi = np.asarray(features, dtype=np.float64)
    if phi.ndim != 3:
        raise CompilationError(
            f"features must have shape (leader_actions, follower_actions, dimension); got shape {phi.shape}"
        )
    if any(size < 1 for size in phi.shape):
        raise CompilationError(f"every feature axis must be nonempty; got shape {phi.shape}")
    if not np.all(np.isfinite(phi)):
        raise CompilationError("features contain NaN or infinite values")
    if not np.isfinite(eta) or eta <= 0:
        raise CompilationError(f"eta must be finite and strictly positive; got {eta!r}")
    if not np.isfinite(radius) or radius <= 0:
        raise CompilationError(f"radius must be finite and strictly positive; got {radius!r}")

    feature_norms = np.linalg.norm(phi, axis=2)
    max_feature_norm = float(np.max(feature_norms))
    if max_feature_norm > 1.0 + config.feature_norm_tolerance:
        index = tuple(int(i) for i in np.unravel_index(np.argmax(feature_norms), feature_norms.shape))
        raise CompilationError(
            "features violate the required unit-norm bound: "
            f"max norm={max_feature_norm:.17g} at (leader, follower)={index}; "
            "normalize explicitly before compilation"
        )

    centered = phi - np.mean(phi, axis=1, keepdims=True)
    rows = centered.reshape(-1, phi.shape[2])
    if np.all(rows == 0):
        basis = np.empty((phi.shape[2], 0), dtype=np.float64)
        rank_threshold = config.rank_absolute_tolerance
    else:
        _, singular_values, vh = np.linalg.svd(rows, full_matrices=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        rank_threshold = max(config.rank_absolute_tolerance, config.rank_relative_tolerance * largest)
        rank = int(np.count_nonzero(singular_values > rank_threshold))
        basis = vh[:rank].T

    reduced = np.einsum("abd,dg->abg", centered, basis, optimize=True)
    n_follower = phi.shape[1]
    coverage = np.einsum("abg,abh->agh", reduced, reduced, optimize=True) / n_follower

    return CompiledProblem(
        features=_readonly(phi.copy()),
        eta=float(eta),
        radius=float(radius),
        identifiable_basis=_readonly(basis),
        centered_pure_features=_readonly(centered),
        reduced_centered_pure_features=_readonly(reduced),
        pure_coverage_matrices=_readonly(coverage),
        rank_threshold=float(rank_threshold),
        max_feature_norm=max_feature_norm,
    )
