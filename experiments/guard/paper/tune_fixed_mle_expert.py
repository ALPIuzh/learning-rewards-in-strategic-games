"""Tune the fixed Stackelberg-expert MLE ridge on protected source-task splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.stats import t as student_t

from experiments.guard.paper.application import LobekeZeroSumApplication
from experiments.guard.paper.study import LobekeComparisonStudy
from mairl.core.math import fisher_information
from mairl_exp.config import REPOSITORY_ROOT, ExperimentNumerics

DEFAULT_GRID = (1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 1.0, 2.0, 3.0, 10.0)
METRICS = ("stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error")


def _load() -> tuple[LobekeComparisonStudy, ExperimentNumerics]:
    with initialize_config_dir(config_dir=str((REPOSITORY_ROOT / "conf").resolve()), version_base=None):
        config = compose(config_name="runs/lobeke/comparison")
        study = instantiate(config.experiment)
        numerics = instantiate(config.numerics)
    if not isinstance(study, LobekeComparisonStudy):
        raise TypeError("runs/lobeke/comparison did not instantiate LobekeComparisonStudy")
    if not isinstance(numerics, ExperimentNumerics):
        raise TypeError("runs/lobeke/comparison did not instantiate ExperimentNumerics")
    return study, numerics


def _standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _summary(values: np.ndarray) -> dict[str, object]:
    return {"mean": float(np.mean(values)), "standard_error": _standard_error(values), "values": values.tolist()}


def _paired_summary(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    differences = candidate - reference
    standard_error = _standard_error(differences)
    critical = float(student_t.ppf(0.975, differences.size - 1))
    mean = float(np.mean(differences))
    return {
        "candidate_mean": float(np.mean(candidate)),
        "reference_mean": float(np.mean(reference)),
        "paired_difference_mean": mean,
        "paired_difference_standard_error": standard_error,
        "paired_difference_ci_95": [mean - critical * standard_error, mean + critical * standard_error],
        "candidate_better_count": int(np.count_nonzero(candidate < reference)),
        "values": differences.tolist(),
    }


def _evaluate(base_study: LobekeComparisonStudy, application: object, numerics: ExperimentNumerics,
        *, ridge: float, seed_start: int, seed_count: int, total_samples: int) -> dict[str, object]:  # fmt: skip
    study = replace(
        base_study,
        name="guard-lobeke-fixed-stackelberg-expert-mle-tuning",
        seed_start=seed_start,
        seed_count=seed_count,
        total_samples=total_samples,
        warmup_budget=0,
        evaluation_interval=total_samples,
        fixed_expert_mle_ridge=ridge,
        methods=("fixed_expert_mle",),
        warmup_modes=(False,),
    )
    results = study.executor(application, numerics).run_many(study.jobs())
    metrics = {name: np.asarray([result.summary["final_metrics"][name] for result in results]) for name in METRICS}
    estimates = np.stack([result.arrays["omega_history"][-1] for result in results])
    radius = application.problem.radius
    return {
        "ridge": ridge,
        "seeds": list(range(seed_start, seed_start + seed_count)),
        "metrics": {name: _summary(values) for name, values in metrics.items()},
        "boundary_fraction": float(np.mean(np.isclose(np.linalg.norm(estimates, axis=1), radius, atol=1e-7))),
        "mean_estimate": np.mean(estimates, axis=0).tolist(),
        "metrics_arrays": metrics,
    }


def _public(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key != "metrics_arrays"}


def build_record(*, grid: tuple[float, ...], total_samples: int, selection_seed_start: int, selection_seed_count: int,
        confirmation_seed_start: int, confirmation_seed_count: int) -> dict[str, object]:  # fmt: skip
    base_study, numerics = _load()
    application = LobekeZeroSumApplication().build(numerics)
    selection = [
        _evaluate(
            base_study,
            application,
            numerics,
            ridge=ridge,
            seed_start=selection_seed_start,
            seed_count=selection_seed_count,
            total_samples=total_samples,
        )
        for ridge in grid
    ]
    best = min(selection, key=lambda result: result["metrics"]["stackelberg_suboptimality"]["mean"])
    threshold = (
        best["metrics"]["stackelberg_suboptimality"]["mean"]
        + best["metrics"]["stackelberg_suboptimality"]["standard_error"]
    )
    eligible = [result for result in selection if result["metrics"]["stackelberg_suboptimality"]["mean"] <= threshold]
    selected_ridge = float(min(result["ridge"] for result in eligible))
    reference_ridge = float(grid[0])
    confirmation_candidate = _evaluate(
        base_study,
        application,
        numerics,
        ridge=selected_ridge,
        seed_start=confirmation_seed_start,
        seed_count=confirmation_seed_count,
        total_samples=total_samples,
    )
    confirmation_reference = _evaluate(
        base_study,
        application,
        numerics,
        ridge=reference_ridge,
        seed_start=confirmation_seed_start,
        seed_count=confirmation_seed_count,
        total_samples=total_samples,
    )
    policy = application.fixed_mle_leader_policy
    information = fisher_information(application.problem, policy, application.true_omega)
    eigenvalues = np.linalg.eigvalsh(information)
    return {
        "schema_version": 1,
        "criterion": "final source-task stackelberg_suboptimality",
        "expert": {
            "definition": application.fixed_mle_expert_definition,
            "leader_policy": policy.tolist(),
            "minimum_probability": float(np.min(policy)),
            "maximum_probability": float(np.max(policy)),
            "support_size": int(np.count_nonzero(policy > 1e-12)),
            "per_sample_fisher_eigenvalues": eigenvalues.tolist(),
            "per_sample_fisher_condition": float(eigenvalues[-1] / eigenvalues[0]),
        },
        "decision": {
            "grid": list(grid),
            "reference_ridge": reference_ridge,
            "best_mean_ridge": float(best["ridge"]),
            "one_standard_error_threshold": threshold,
            "eligible_ridges": [float(result["ridge"]) for result in eligible],
            "selected_ridge": selected_ridge,
            "selection_rule": (
                "minimum source SubOpt; retain candidates within one standard error "
                "of its mean; choose the smallest ridge"
            ),
            "selected_is_grid_boundary": selected_ridge in (min(grid), max(grid)),
            "downstream_tasks_used": False,
        },
        "seed_policy": {
            "selection": list(range(selection_seed_start, selection_seed_start + selection_seed_count)),
            "confirmation": list(range(confirmation_seed_start, confirmation_seed_start + confirmation_seed_count)),
            "final_evaluation": list(range(30_000, 30_050)),
        },
        "total_samples": total_samples,
        "selection": [_public(result) for result in selection],
        "confirmation": {
            "candidate": _public(confirmation_candidate),
            "reference": _public(confirmation_reference),
            "paired_metrics": {
                name: _paired_summary(
                    confirmation_candidate["metrics_arrays"][name], confirmation_reference["metrics_arrays"][name]
                )
                for name in METRICS
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed-start", type=int, default=10_000)
    parser.add_argument("--selection-seed-count", type=int, default=64)
    parser.add_argument("--confirmation-seed-start", type=int, default=20_000)
    parser.add_argument("--confirmation-seed-count", type=int, default=64)
    parser.add_argument("--total-samples", type=int, default=1_600)
    parser.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing tuning artifact: {args.output}")
    record = build_record(
        grid=tuple(args.grid),
        selection_seed_start=args.selection_seed_start,
        selection_seed_count=args.selection_seed_count,
        confirmation_seed_start=args.confirmation_seed_start,
        confirmation_seed_count=args.confirmation_seed_count,
        total_samples=args.total_samples,
    )
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "selected_ridge": record["decision"]["selected_ridge"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
