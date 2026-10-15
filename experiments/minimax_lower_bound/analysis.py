"""Auditable aggregation of completed lower-bound jobs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from mairl.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class FinalMetricRecord:
    """One final metric from one coordinate/replicate execution."""

    dimension: int
    samples_per_dimension: int
    active_coordinate: int
    replicate: int
    value: float


def paired_bootstrap_loglog_slope(x: np.ndarray, values: np.ndarray, *, bootstrap_samples: int,
        seed: int, confidence_level: float = 0.95) -> dict[str, object]:  # fmt: skip
    """Estimate a log-log slope and paired nonparametric confidence interval.

    ``values`` has shape ``(x points, paired units)``. The same resampled unit
    indices are used at every x value, preserving the experiment's paired-seed
    construction.
    """

    x_values = np.asarray(x, dtype=np.float64)
    observations = np.asarray(values, dtype=np.float64)
    if (
        x_values.ndim != 1
        or x_values.size < 2
        or observations.ndim != 2
        or observations.shape[0] != x_values.size
        or observations.shape[1] < 2
        or not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(observations))
        or np.any(x_values <= 0)
        or np.any(observations < 0)
        or np.any(observations.mean(axis=1) <= 0)
    ):
        raise ConfigurationError("log-log slope inputs must be finite, positive, and paired")
    if bootstrap_samples < 100 or seed < 0 or not 0 < confidence_level < 1:
        raise ConfigurationError("invalid paired-bootstrap configuration")

    centered_x = np.log(x_values) - np.log(x_values).mean()
    denominator = float(centered_x @ centered_x)

    def slopes(means: np.ndarray) -> np.ndarray:
        if np.any(means <= 0):
            raise ConfigurationError("bootstrap mean vanished, so its log slope is undefined")
        log_means = np.log(means)
        return (log_means @ centered_x) / denominator

    point = float(slopes(observations.mean(axis=1)[None, :])[0])
    random = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 256):
        stop = min(start + 256, bootstrap_samples)
        indices = random.integers(0, observations.shape[1], size=(stop - start, observations.shape[1]))
        means = np.stack([observations[index][indices].mean(axis=1) for index in range(x_values.size)], axis=1)
        bootstrap[start:stop] = slopes(means)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap, (tail, 1.0 - tail))
    return {
        "estimate": point,
        "confidence_level": confidence_level,
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "paired_units": observations.shape[1],
    }


def aggregate_coordinate_risk(records: Iterable[FinalMetricRecord]) -> tuple[dict[str, object], ...]:
    """Return both uniform-coordinate and maximum-coordinate sample means.

    The maximum is deliberately named a *sample* mean: it estimates the theorem's
    maximum expected risk only when every coordinate has enough independent replicates.
    The replicate count and every coordinate mean are retained so this limitation is
    visible rather than hidden by the plotting layer.
    """

    grouped: dict[tuple[int, int], list[FinalMetricRecord]] = defaultdict(list)
    for record in records:
        if not np.isfinite(record.value):
            raise ConfigurationError("lower-bound metric records must be finite")
        grouped[(record.dimension, record.samples_per_dimension)].append(record)
    if not grouped:
        raise ConfigurationError("at least one lower-bound metric record is required")

    cells: list[dict[str, object]] = []
    for (dimension, ratio), cell_records in sorted(grouped.items()):
        by_coordinate: dict[int, list[FinalMetricRecord]] = defaultdict(list)
        for record in cell_records:
            if not 0 <= record.active_coordinate < dimension:
                raise ConfigurationError("active coordinate lies outside its dimension")
            by_coordinate[record.active_coordinate].append(record)
        if set(by_coordinate) != set(range(dimension)):
            raise ConfigurationError("every coordinate must be represented in each (d, K) cell")
        replicate_counts = {len(items) for items in by_coordinate.values()}
        if len(replicate_counts) != 1:
            raise ConfigurationError("coordinate replicate counts must be balanced")
        replicates = replicate_counts.pop()
        expected_replicates = set(range(replicates))
        if any({item.replicate for item in items} != expected_replicates for items in by_coordinate.values()):
            raise ConfigurationError("each coordinate must contain every replicate exactly once")

        coordinate_values = np.asarray(
            [
                [item.value for item in sorted(by_coordinate[index], key=lambda item: item.replicate)]
                for index in range(dimension)
            ],
            dtype=np.float64,
        )
        coordinate_means = coordinate_values.mean(axis=1)
        coordinate_standard_errors = (
            np.zeros(dimension) if replicates == 1 else coordinate_values.std(axis=1, ddof=1) / np.sqrt(replicates)
        )
        uniform_values = coordinate_values.reshape(-1)
        maximum = float(np.max(coordinate_means))
        cells.append(
            {
                "dimension": dimension,
                "samples_per_dimension": ratio,
                "total_samples": dimension * ratio,
                "replicates_per_coordinate": replicates,
                "uniform_coordinate_mean": float(np.mean(coordinate_means)),
                "uniform_coordinate_standard_error": float(uniform_values.std(ddof=1) / np.sqrt(uniform_values.size)),
                "maximum_coordinate_sample_mean": maximum,
                "maximizing_coordinates": np.flatnonzero(
                    np.isclose(coordinate_means, maximum, rtol=1e-12, atol=1e-15)
                ).tolist(),
                "coordinate_means": coordinate_means.tolist(),
                "coordinate_standard_errors": coordinate_standard_errors.tolist(),
                "coordinate_values": coordinate_values.tolist(),
            }
        )
    return tuple(cells)


def records_from_job_summaries(summaries: Iterable[Mapping[str, object]], metric: str) -> tuple[FinalMetricRecord, ...]:
    """Extract typed records from durable runner summaries."""

    records: list[FinalMetricRecord] = []
    for summary in summaries:
        job = summary.get("job")
        metrics = summary.get("final_metrics")
        if not isinstance(job, Mapping) or not isinstance(metrics, Mapping):
            raise ConfigurationError("job summaries must contain job and final_metrics mappings")
        if metric not in metrics:
            raise ConfigurationError(f"job summary is missing metric {metric!r}")
        coordinates = tuple(int(value) for value in job["active_coordinates"])
        values = tuple(float(value) for value in metrics[metric])
        if len(coordinates) != len(values):
            raise ConfigurationError("compact job summary metric length is inconsistent")
        records.extend(
            FinalMetricRecord(
                dimension=int(job["dimension"]),
                samples_per_dimension=int(job["samples_per_dimension"]),
                active_coordinate=coordinate,
                replicate=int(job["replicate"]),
                value=value,
            )
            for coordinate, value in zip(coordinates, values, strict=True)
        )
    return tuple(records)
