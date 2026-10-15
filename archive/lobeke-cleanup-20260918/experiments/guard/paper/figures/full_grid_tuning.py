"""Incrementally export the complete N=1000 Lobeke tuning heatmap.

The existing 1,600-sample selection trajectories contain exact prefixes at
N=1,000.  This exporter combines those prefixes with completed cells from the
small completion run, evaluates every available estimate on a fixed collection
of held-out downstream rewards, and writes compact publication data.  Missing
cells remain NaN in the NPZ and are rendered as pending crosses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mairl_exp.config import REPOSITORY_ROOT, RUNS_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import numpy as np

from experiments.guard.paper.figures.downstream import load_runtime
from experiments.guard.paper.figures.task_sets import TUNING_TASKS, normalized_structured_tasks
from mairl.exceptions import ConfigurationError
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy_batch
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file

EXISTING_RUNS = (
    RUNS_ROOT / "guard-lobeke-warm-selection" / "2ce0376fe643084a",
    RUNS_ROOT / "guard-lobeke-no-warm-selection" / "9e985f76d0b8ed3a",
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "data" / "lobeke" / "tuning_grid.json"
DEFAULT_CACHE = REPOSITORY_ROOT / "artifacts" / "derived" / "guard-lobeke-full-grid-1000" / "task_scores"

TOTAL_SAMPLES = 1_000
SEEDS = np.arange(10_000, 10_020, dtype=np.int64)
WARMUP_BUDGETS = np.asarray((0, 50, 100, 200, 400), dtype=np.int64)
MULTIPLIERS = np.asarray((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), dtype=np.float64)
TASK_COUNT = TUNING_TASKS.count
TASK_SEED = TUNING_TASKS.seed
TASK_CANDIDATE_POOL = TUNING_TASKS.candidate_pool
PREFIX_EQUIVALENCE_STANDARD_ERRORS = 3.0

Candidate = tuple[int, float]
DuplicateReference = tuple[np.ndarray, np.ndarray]


def _job_results(run_dir: Path) -> dict[tuple[Candidate, int], Path]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    results: dict[tuple[Candidate, int], Path] = {}
    for expected in manifest.get("expected_jobs", []):
        spec = expected["spec"]
        seed = int(spec["seed"])
        if seed not in SEEDS:
            continue
        job_dir = run_dir / "jobs" / expected["job_id"]
        status_path = job_dir / "status.json"
        result_path = job_dir / "result.npz"
        if not status_path.exists() or not result_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete" or status.get("result_sha256") != sha256_file(result_path):
            continue
        key = (int(spec["warmup_budget"]), float(spec["lambda_information_multiplier"]))
        results[(key, seed)] = result_path
    return results


def _omega_at(path: Path, sample_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        counts = arrays["trace_sample_counts"]
        position = int(np.searchsorted(counts, sample_count))
        if position >= counts.size or int(counts[position]) != sample_count:
            raise ConfigurationError(f"{path} has no estimate at sample {sample_count}")
        return np.asarray(arrays["omega_history"][position], dtype=np.float64).copy()


def _available_estimates(completion_run_dir: Path | None,
) -> tuple[dict[Candidate, np.ndarray], dict[str, Any], DuplicateReference | None]:  # fmt: skip
    existing: dict[tuple[Candidate, int], Path] = {}
    for run_dir in EXISTING_RUNS:
        existing.update(_job_results(run_dir))
    completion = {} if completion_run_dir is None else _job_results(completion_run_dir)

    estimates: dict[Candidate, np.ndarray] = {}
    duplicate_reference: DuplicateReference | None = None
    reused_cells: list[list[float | int]] = []
    completed_cells: list[list[float | int]] = []
    for budget in WARMUP_BUDGETS:
        for multiplier in MULTIPLIERS:
            candidate = int(budget), float(multiplier)
            old_paths = [existing.get((candidate, int(seed))) for seed in SEEDS]
            new_paths = [completion.get((candidate, int(seed))) for seed in SEEDS]
            old_complete = all(path is not None for path in old_paths)
            new_complete = all(path is not None for path in new_paths)
            if old_complete:
                values = np.asarray([_omega_at(path, TOTAL_SAMPLES) for path in old_paths])
                estimates[candidate] = values
                reused_cells.append([candidate[0], candidate[1]])
            elif new_complete:
                values = np.asarray([_omega_at(path, TOTAL_SAMPLES) for path in new_paths])
                estimates[candidate] = values
                completed_cells.append([candidate[0], candidate[1]])

            # The duplicated reference checks that re-batching the same seeds does
            # not materially change the downstream result. Exact paths need not be
            # identical: a near-tied non-convex policy argmax can flip under roundoff
            # when the leading batch dimension changes.
            if candidate == (100, 1.0) and old_complete and new_complete:
                old_values = np.asarray([_omega_at(path, TOTAL_SAMPLES) for path in old_paths])
                new_values = np.asarray([_omega_at(path, TOTAL_SAMPLES) for path in new_paths])
                duplicate_reference = old_values, new_values

    provenance = {
        "existing_runs": [str(path.relative_to(REPOSITORY_ROOT)) for path in EXISTING_RUNS],
        "completion_run": (
            None if completion_run_dir is None else str(completion_run_dir.relative_to(REPOSITORY_ROOT))
        ),
        "reused_cells": reused_cells,
        "newly_completed_cells": completed_cells,
    }
    return estimates, provenance, duplicate_reference


def _paired_equivalence(reference: np.ndarray, duplicate: np.ndarray,
        *, standard_errors: float = PREFIX_EQUIVALENCE_STANDARD_ERRORS) -> dict[str, float | int | bool]:  # fmt: skip
    """Check whether two paired seed evaluations differ materially in their mean."""

    if reference.shape != duplicate.shape or reference.ndim != 1 or reference.size < 2:
        raise ConfigurationError("duplicate reference scores must be paired one-dimensional data")
    differences = duplicate - reference
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
    threshold = standard_errors * standard_error
    equivalent = abs(mean) <= threshold if standard_error > 0 else mean == 0
    return {
        "paired_mean_difference": mean,
        "paired_standard_error": standard_error,
        "standard_error_multiplier": standard_errors,
        "acceptance_threshold": threshold,
        "equivalent": equivalent,
        "seeds": int(differences.size),
    }


def _tuning_tasks(runtime: Any) -> np.ndarray:
    return normalized_structured_tasks(runtime, TUNING_TASKS)


def _digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _candidate_cache_path(cache_dir: Path, candidate: Candidate) -> Path:
    return cache_dir / f"warmup-{candidate[0]}_lambda-{candidate[1]:g}.npz"


def _task_scores(runtime: Any, tasks: np.ndarray, candidate: Candidate,
        omegas: np.ndarray, cache_dir: Path) -> np.ndarray:  # fmt: skip
    cache_path = _candidate_cache_path(cache_dir, candidate)
    omega_sha256 = _digest_array(omegas)
    task_sha256 = _digest_array(tasks)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["omega_sha256"].item()) == omega_sha256 and str(cached["task_sha256"].item()) == task_sha256:
                return cached["scores"].copy()

    config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=2)
    )
    scores = np.empty((SEEDS.size, TASK_COUNT), dtype=np.float64)
    for task_index, reward in enumerate(tasks):
        oracle = optimize_leader_policy_batch(
            runtime.problem, runtime.true_omega[None, :], reward, config, (TASK_SEED + task_index,)
        )[0]
        solutions = optimize_leader_policy_batch(
            runtime.problem, omegas, reward, config, tuple(int(seed) for seed in SEEDS)
        )
        for seed_index, solution in enumerate(solutions):
            value, _ = leader_value_and_gradient(runtime.problem, runtime.true_omega, reward, solution.policy)
            scores[seed_index, task_index] = max(0.0, oracle.value - value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(
        cache_path, {"scores": scores, "omega_sha256": np.asarray(omega_sha256), "task_sha256": np.asarray(task_sha256)}
    )
    return scores


def export(*, completion_run_dir: Path | None, output: Path, cache_dir: Path) -> Path:
    if completion_run_dir is not None:
        completion_run_dir = completion_run_dir.resolve()
    estimates, source, duplicate_reference = _available_estimates(completion_run_dir)
    runtime = load_runtime(EXISTING_RUNS[0])
    tasks = _tuning_tasks(runtime)
    if duplicate_reference is not None:
        old_omegas, new_omegas = duplicate_reference
        old_scores = _task_scores(runtime, tasks, (100, 1.0), old_omegas, cache_dir)
        new_scores = _task_scores(runtime, tasks, (100, 1.0), new_omegas, cache_dir / "duplicate_reference")
        equivalence = _paired_equivalence(np.mean(old_scores, axis=1), np.mean(new_scores, axis=1))
        equivalence["maximum_omega_absolute_difference"] = float(np.max(np.abs(old_omegas - new_omegas)))
        equivalence["exact_prefix_match"] = bool(np.array_equal(old_omegas, new_omegas))
        source["duplicate_reference_check"] = equivalence
        if not equivalence["equivalent"]:
            raise ConfigurationError(
                "N=1000 duplicate reference failed paired downstream equivalence: "
                f"mean difference {equivalence['paired_mean_difference']:.3e}, "
                f"SE {equivalence['paired_standard_error']:.3e}"
            )
    shape = (WARMUP_BUDGETS.size, MULTIPLIERS.size, SEEDS.size, TASK_COUNT)
    all_scores = np.full(shape, np.nan, dtype=np.float64)
    candidates: list[dict[str, Any]] = []

    for row, budget in enumerate(WARMUP_BUDGETS):
        for column, multiplier in enumerate(MULTIPLIERS):
            candidate = int(budget), float(multiplier)
            omegas = estimates.get(candidate)
            if omegas is None:
                continue
            scores = _task_scores(runtime, tasks, candidate, omegas, cache_dir)
            all_scores[row, column] = scores
            per_seed = np.mean(scores, axis=1)
            candidates.append(
                {
                    "warmup_budget": candidate[0],
                    "lambda_information_multiplier": candidate[1],
                    "downstream_suboptimality": {
                        "mean": float(np.mean(per_seed)),
                        "standard_error": float(np.std(per_seed, ddof=1) / np.sqrt(SEEDS.size)),
                        "median": float(np.median(per_seed)),
                        "seeds": int(SEEDS.size),
                        "tasks": TASK_COUNT,
                    },
                }
            )

    npz_path = output.with_suffix(".npz")
    atomic_write_npz(
        npz_path,
        {
            "warmup_budgets": WARMUP_BUDGETS,
            "lambda_information_multipliers": MULTIPLIERS,
            "seeds": SEEDS,
            "task_rewards": tasks,
            "task_scores": all_scores,
        },
    )
    total_cells = int(WARMUP_BUDGETS.size * MULTIPLIERS.size)
    complete = len(candidates) == total_cells
    selected: list[dict[str, float | int]] = []
    if complete:
        winner = min(candidates, key=lambda item: item["downstream_suboptimality"]["mean"])
        selected.append(
            {
                "warmup_budget": int(winner["warmup_budget"]),
                "lambda_information_multiplier": float(winner["lambda_information_multiplier"]),
            }
        )
    atomic_write_json(
        output,
        {
            "schema_version": 2,
            "status": "complete" if complete else "partial",
            "total_samples": TOTAL_SAMPLES,
            "warmup_budgets": WARMUP_BUDGETS.tolist(),
            "lambda_information_multipliers": MULTIPLIERS.tolist(),
            "seed_policy": {"start": int(SEEDS[0]), "count": int(SEEDS.size), "paired_across_cells": True},
            "task_policy": {
                "count": TASK_COUNT,
                "seed": TASK_SEED,
                "candidate_pool": TASK_CANDIDATE_POOL,
                "normalization": "unit Frobenius norm per reward matrix",
                "disjoint_from_main_figure": True,
                "same_tasks_for_every_cell_and_seed": True,
            },
            "score": "per-seed mean downstream SubOpt, then mean across seeds",
            "completed_cells": len(candidates),
            "total_cells": total_cells,
            "selected": selected,
            "candidates": candidates,
            "source": source,
            "data_file": npz_path.name,
            "data_sha256": sha256_file(npz_path),
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-run-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    output = export(completion_run_dir=args.completion_run_dir, output=args.output, cache_dir=args.cache_dir)
    payload = json.loads(output.read_text())
    print(f"wrote {output}: {payload['completed_cells']}/{payload['total_cells']} cells ({payload['status']})")


if __name__ == "__main__":
    main()
