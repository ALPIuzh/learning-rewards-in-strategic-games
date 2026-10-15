"""Durable, vectorized study for the minimax lower-bound instance."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import asdict, dataclass, field

import numpy as np

from experiments.minimax_lower_bound.simulation import run_hard_instance_batch
from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.jobs import StudyJobResult


@dataclass(frozen=True, slots=True)
class MinimaxJob:
    """One durable replicate containing every active coordinate for one grid cell."""

    dimension: int
    samples_per_dimension: int
    replicate: int
    active_coordinates: tuple[int, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_coordinates", tuple(self.active_coordinates))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.dimension < 1 or self.samples_per_dimension < 1 or self.replicate < 0:
            raise ConfigurationError("lower-bound job dimensions and replicate are invalid")
        if len(self.active_coordinates) != self.dimension or set(self.active_coordinates) != set(range(self.dimension)):
            raise ConfigurationError("a compact lower-bound job must contain every coordinate")
        if len(self.seeds) != self.dimension or len(set(self.seeds)) != self.dimension:
            raise ConfigurationError("a compact lower-bound job needs one unique seed per coordinate")
        if any(seed < 0 for seed in self.seeds):
            raise ConfigurationError("lower-bound execution seeds must be nonnegative")

    @property
    def total_samples(self) -> int:
        return self.dimension * self.samples_per_dimension

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"total_samples": self.total_samples}

    @property
    def label(self) -> str:
        return f"d={self.dimension}/K={self.total_samples}/active/all-coordinates/replicate={self.replicate}"


def balanced_coordinate_schedule(dimension: int, replicates_per_coordinate: int,
        *, design_seed: int) -> tuple[tuple[int, int], ...]:  # fmt: skip
    """Return ``(replicate, coordinate)`` pairs balanced within every replicate."""

    if dimension < 1 or replicates_per_coordinate < 1 or design_seed < 0:
        raise ConfigurationError("dimension, replicates, and design_seed must be positive")
    schedule: list[tuple[int, int]] = []
    for replicate in range(replicates_per_coordinate):
        sequence = np.random.SeedSequence([0x4C4F5742, design_seed, dimension, replicate])
        coordinates = np.random.default_rng(sequence).permutation(dimension)
        schedule.extend((replicate, int(coordinate)) for coordinate in coordinates)
    return tuple(schedule)


@dataclass(slots=True)
class MinimaxLowerBoundStudy:
    """Prespecified I.3 grid with balanced coordinate replications."""

    name: str = "minimax-lower-bound"
    dimensions: tuple[int, ...] = (33, 64)
    samples_per_dimension: tuple[int, ...] = (4, 8, 16, 32, 64)
    seed_start: int = 92000
    replicates_per_coordinate: int = 16
    coordinate_design_seed: int = 7319
    confidence_delta: float = 0.05
    _numerics: ExperimentNumerics = field(init=False, repr=False, compare=False)
    _jobs: frozenset[MinimaxJob] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.dimensions = tuple(self.dimensions)
        self.samples_per_dimension = tuple(self.samples_per_dimension)
        if not self.name:
            raise ConfigurationError("study name must be nonempty")
        if self.seed_start < 0 or self.replicates_per_coordinate < 1:
            raise ConfigurationError("seed_start must be nonnegative and replicates positive")
        if self.coordinate_design_seed < 0:
            raise ConfigurationError("coordinate_design_seed must be nonnegative")
        if not self.dimensions or len(set(self.dimensions)) != len(self.dimensions):
            raise ConfigurationError("dimensions must be nonempty and unique")
        if any(dimension < 32 for dimension in self.dimensions):
            raise ConfigurationError("all dimensions must satisfy d >= 32")
        if not self.samples_per_dimension or len(set(self.samples_per_dimension)) != len(self.samples_per_dimension):
            raise ConfigurationError("K/d values must be nonempty and unique")
        if any(ratio < 4 for ratio in self.samples_per_dimension):
            raise ConfigurationError("all K/d values must be at least 4")
        if not 0 < self.confidence_delta < 1:
            raise ConfigurationError("confidence_delta must lie in (0, 1)")

    def jobs(self) -> tuple[MinimaxJob, ...]:
        jobs: list[MinimaxJob] = []
        for dimension in self.dimensions:
            schedule = balanced_coordinate_schedule(
                dimension, self.replicates_per_coordinate, design_seed=self.coordinate_design_seed
            )
            for ratio in self.samples_per_dimension:
                for replicate in range(self.replicates_per_coordinate):
                    coordinates = tuple(
                        coordinate for scheduled_replicate, coordinate in schedule if scheduled_replicate == replicate
                    )
                    seeds = tuple(self.seed_start + replicate * dimension + position for position in range(dimension))
                    jobs.append(
                        MinimaxJob(
                            dimension=dimension,
                            samples_per_dimension=ratio,
                            replicate=replicate,
                            active_coordinates=coordinates,
                            seeds=seeds,
                        )
                    )
        return tuple(jobs)

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> MinimaxLowerBoundStudy:
        if application.name != "minimax-lower-bound":
            raise ConfigurationError("minimax study requires the minimax application")
        prepared = copy(self)
        prepared._numerics = numerics
        prepared._jobs = frozenset(self.jobs())
        return prepared

    def run(self, job: MinimaxJob, *, on_progress: Callable[[int, int], None] | None = None) -> StudyJobResult:
        if job not in self._jobs:
            raise ConfigurationError(f"job {job.label!r} is not part of this study")

        def report(seed_index: int, samples: int, total: int) -> None:
            if on_progress is not None and seed_index == job.dimension - 1:
                on_progress(samples, total)

        results = run_hard_instance_batch(
            dimension=job.dimension,
            total_samples=job.total_samples,
            seeds=job.seeds,
            active_coordinates=job.active_coordinates,
            numerics=self._numerics,
            on_progress=report,
        )
        return self._pack(results, job)

    @staticmethod
    def _pack(results: tuple[StudyJobResult, ...], job: MinimaxJob) -> StudyJobResult:
        if len(results) != job.dimension:
            raise RuntimeError("lower-bound coordinate batch returned the wrong result count")
        epsilon = 0.5 * np.sqrt(job.dimension / job.total_samples)
        names = set(results[0].arrays)
        if any(set(result.arrays) != names for result in results):
            raise RuntimeError("lower-bound coordinate results have inconsistent arrays")
        arrays = {name: np.stack([result.arrays[name] for result in results]) for name in sorted(names)}
        deployed_coordinates = np.argmax(arrays["omega_history"][:, -1], axis=1)
        active_coordinates = np.asarray(job.active_coordinates, dtype=np.int64)
        arrays.update(
            {
                "deployment__active_coordinate_identified": np.asarray(
                    deployed_coordinates == active_coordinates, dtype=np.bool_
                ),
                "deployment__coordinate": np.asarray(deployed_coordinates, dtype=np.int64),
                "instance__active_coordinate": active_coordinates,
                "instance__dimension": np.asarray(job.dimension, dtype=np.int64),
                "instance__epsilon": np.asarray(epsilon, dtype=np.float64),
                "instance__replicate": np.asarray(job.replicate, dtype=np.int64),
                "instance__seeds": np.asarray(job.seeds, dtype=np.int64),
                "instance__total_samples": np.asarray(job.total_samples, dtype=np.int64),
            }
        )
        metrics = {
            name.removeprefix("metric__"): values[:, -1].tolist()
            for name, values in arrays.items()
            if name.startswith("metric__")
        }
        summary = {
            "job": job.as_dict(),
            "algorithm_seconds": float(sum(result.summary["algorithm_seconds"] for result in results)),
            "coordinate_count": job.dimension,
            "final_metrics": metrics,
            "deployment": {
                "coordinates": deployed_coordinates.tolist(),
                "active_coordinate_identified": (deployed_coordinates == active_coordinates).tolist(),
            },
            "instance": {
                "dimension": job.dimension,
                "total_samples": job.total_samples,
                "active_coordinates": list(job.active_coordinates),
                "replicate": job.replicate,
                "epsilon": float(epsilon),
                "eta": 1.0,
                "radius": 1.0,
                "leader_reward": "1{b=b2}",
                "coordinate_design": "balanced",
            },
            "lambda_information": 1.0,
            "mle_solver": "one-hot-sufficient-statistics",
            "mle_ridge": float(results[0].summary["mle_ridge"]),
            "samples": job.total_samples,
            "warmup": {"enabled": False, "samples": 0, "support_size": 0},
        }
        return StudyJobResult(arrays=arrays, summary=summary)
