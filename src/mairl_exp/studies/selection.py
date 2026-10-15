"""Paired candidate comparisons and completed-run summaries for tuning."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mairl.exceptions import ConfigurationError

MINIMUM_SEEDS_MESSAGE = "protected tuning requires at least two seeds"


def not_worse(difference: dict[str, float | int], standard_errors: float) -> bool:
    """Guardrail test: the candidate is no worse than the reference within tolerance."""

    return bool(float(difference["mean"]) <= standard_errors * float(difference["standard_error"]))


def sample_statistics(values: Sequence[float], expected_count: int | None = None,
        *, minimum_seeds_message: str = MINIMUM_SEEDS_MESSAGE) -> dict[str, float | int]:  # fmt: skip
    """Summarize per-seed scores; ``expected_count`` also pins how many are required."""

    samples = np.asarray(values, dtype=np.float64)
    if expected_count is not None and (samples.size != expected_count or not np.all(np.isfinite(samples))):
        raise ConfigurationError(f"expected {expected_count} finite scores, received {samples.size}")
    if samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ConfigurationError(minimum_seeds_message)
    return {
        "mean": float(np.mean(samples)),
        "standard_error": float(np.std(samples, ddof=1) / np.sqrt(samples.size)),
        "median": float(np.median(samples)),
        "seeds": int(samples.size),
    }


def paired_statistics(candidate: dict[int, float], reference: dict[int, float],
        *, mismatch_message: str = "paired candidates do not contain identical seeds",
        minimum_seeds_message: str = MINIMUM_SEEDS_MESSAGE) -> dict[str, float | int]:  # fmt: skip
    """Summarize per-seed candidate-minus-reference differences on identical seeds."""

    if candidate.keys() != reference.keys():
        raise ConfigurationError(mismatch_message)
    seeds = sorted(candidate)
    return sample_statistics(
        [candidate[seed] - reference[seed] for seed in seeds], len(seeds), minimum_seeds_message=minimum_seeds_message
    )


def completed_summaries(run_dir: Path, *, require_nonempty: bool = False,
        empty_message: str = "finalization found no completed jobs") -> tuple[dict[str, object], ...]:  # fmt: skip
    """Load the summary of every job the run store marked complete."""

    summaries = []
    for path in sorted((run_dir / "jobs").glob("*/summary.json")):
        status_path = path.with_name("status.json")
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
            summaries.append(json.loads(path.read_text()))
    if require_nonempty and not summaries:
        raise ConfigurationError(empty_message)
    return tuple(summaries)


def actual_lambdas(summaries: tuple[dict[str, object], ...]) -> dict[str, float]:
    """Map each lambda multiplier to the single actual value it produced."""

    grouped: dict[str, set[float]] = {}
    for summary in summaries:
        job = summary["job"]
        multiplier = f"{float(job['lambda_information_multiplier']):g}"
        grouped.setdefault(multiplier, set()).add(float(summary["lambda_information"]))
    if any(len(values) != 1 for values in grouped.values()):
        raise ConfigurationError("a lambda multiplier produced inconsistent actual values")
    return {multiplier: values.pop() for multiplier, values in grouped.items()}
