"""Numerical-profile calibration on explicit pilot seeds."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from mairl.core.math import cumulative_fisher_information
from mairl.exceptions import ConfigurationError, ConvergenceError
from mairl.exploration.design import WarmupConfig
from mairl.exploration.policy import select_information_policy
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.exploration.warmup import PreparedWarmup
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import MLEConfig, solve_mle
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics, WarmupNumerics
from mairl_exp.execution.artifacts import atomic_write_json
from mairl_exp.execution.jobs import ExperimentJob, StudyJob, StudyJobResult, StudyMethod
from mairl_exp.studies.comparison import BaselineComparisonExecutor, BaselineComparisonStudy

CalibrationSplit = Literal["calibration", "validation"]


@dataclass(frozen=True, slots=True)
class CalibrationGuardrails:
    """Reference accuracy requirements used before runtime can affect selection."""

    reference_mle: MLEConfig
    reference_policy: PolicyOptimizationConfig
    max_mle_reference_gap: float
    max_policy_reference_gap: float
    max_mle_residual: float
    max_policy_residual: float
    minimum_seeds_per_split: int = 5
    reject_numeric_boundary: bool = True

    def __post_init__(self) -> None:
        if self.reference_policy.method != "grid":
            raise ConfigurationError("calibration reference_policy must use the grid method")
        for name in ("max_mle_reference_gap", "max_policy_reference_gap", "max_mle_residual", "max_policy_residual"):
            if not getattr(self, name) > 0:
                raise ConfigurationError(f"{name} must be strictly positive")
        if self.minimum_seeds_per_split < 1:
            raise ConfigurationError("minimum_seeds_per_split must be positive")


@dataclass(frozen=True, slots=True)
class NumericalCandidate:
    """Complete candidate numerical profile evaluated on pilot seeds."""

    name: str
    mle: MLEConfig
    warmup: WarmupNumerics
    active_policy: PolicyOptimizationConfig

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("numerical candidate name must be nonempty")

    def create(self, evaluation_policy: PolicyOptimizationConfig) -> ExperimentNumerics:
        return ExperimentNumerics(
            mle=self.mle, warmup=self.warmup, active_policy=self.active_policy, evaluation_policy=evaluation_policy
        )


@dataclass(frozen=True, slots=True)
class CalibrationJob:
    """One numerical candidate evaluated on one pilot seed."""

    candidate: str
    split: CalibrationSplit
    method: StudyMethod
    warmup_enabled: bool
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        warmup = "warmup" if self.warmup_enabled else "no-warmup"
        return f"{self.split}/{warmup}/{self.method}/candidate={self.candidate}/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class NumericalCalibrationStudy:
    """Measure explicit numerical profiles without silently selecting a winner."""

    candidates: tuple[NumericalCandidate, ...]
    guardrails: CalibrationGuardrails
    name: str = "numerical-calibration"
    calibration_seed_start: int = 1_000
    calibration_seed_count: int = 5
    validation_seed_start: int = 2_000
    validation_seed_count: int = 5
    total_samples: int = 180
    warmup_budget: int = 120
    methods: tuple[StudyMethod, ...] = ("active", "fixed_random", "random_each_round")
    warmup_modes: tuple[bool, ...] = (True, False)
    evaluation_interval: int = 10
    lambda_information_multiplier: float = 1.0
    lambda_warm: float = 0.01
    confidence_delta: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "warmup_modes", tuple(self.warmup_modes))
        if not self.name:
            raise ConfigurationError("study name must be nonempty")
        if min(self.calibration_seed_count, self.validation_seed_count) < (self.guardrails.minimum_seeds_per_split):
            raise ConfigurationError(
                f"each calibration seed split must contain at least {self.guardrails.minimum_seeds_per_split} seeds"
            )
        calibration_seeds = set(
            range(self.calibration_seed_start, self.calibration_seed_start + self.calibration_seed_count)
        )
        validation_seeds = set(
            range(self.validation_seed_start, self.validation_seed_start + self.validation_seed_count)
        )
        if calibration_seeds & validation_seeds:
            raise ConfigurationError("calibration and validation seed splits must be disjoint")
        if self.total_samples < 0 or self.warmup_budget < 0:
            raise ConfigurationError("sample budgets must be nonnegative")
        if not self.warmup_modes or len(set(self.warmup_modes)) != len(self.warmup_modes):
            raise ConfigurationError("warmup_modes must be nonempty and unique")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ConfigurationError("methods must be nonempty and unique")
        invalid_methods = set(self.methods) - {"active", "fixed_random", "random_each_round"}
        if invalid_methods:
            raise ConfigurationError(f"unknown calibration methods: {sorted(invalid_methods)}")
        if True in self.warmup_modes and self.warmup_budget > self.total_samples:
            raise ConfigurationError("enabled warmup_budget cannot exceed total_samples")
        if self.evaluation_interval < 1:
            raise ConfigurationError("evaluation_interval must be positive")
        if not self.lambda_information_multiplier > 0:
            raise ConfigurationError("lambda_information_multiplier must be strictly positive")
        if not self.lambda_warm > 0:
            raise ConfigurationError("lambda_warm must be strictly positive")
        if not 0 < self.confidence_delta < 1:
            raise ConfigurationError("confidence_delta must lie in (0, 1)")
        names = [candidate.name for candidate in self.candidates]
        if not names or len(set(names)) != len(names):
            raise ConfigurationError("calibration candidate names must be nonempty and unique")
        if len({candidate.mle.ridge for candidate in self.candidates}) != 1:
            raise ConfigurationError("ridge is a scientific hyperparameter and cannot vary in numerical calibration")
        if self.candidates[0].mle.ridge != self.guardrails.reference_mle.ridge:
            raise ConfigurationError("reference and candidate MLEs must use the same scientific ridge")

    def jobs(self) -> tuple[CalibrationJob, ...]:
        return tuple(
            CalibrationJob(candidate.name, split, method, warmup_enabled, seed)
            for candidate in self.candidates
            for split, start, count in (
                ("calibration", self.calibration_seed_start, self.calibration_seed_count),
                ("validation", self.validation_seed_start, self.validation_seed_count),
            )
            for method in self.methods
            for warmup_enabled in self.warmup_modes
            for seed in range(start, start + count)
        )

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> NumericalCalibrationExecutor:
        return NumericalCalibrationExecutor(self, application, numerics)

    def finalize(self, run_dir: Path) -> None:
        """Select on calibration seeds, then confirm once on validation seeds."""

        summaries = _load_summaries(run_dir)
        by_candidate: dict[str, dict[str, list[dict[str, object]]]] = {
            candidate.name: {"calibration": [], "validation": []} for candidate in self.candidates
        }
        for summary in summaries:
            job = summary["job"]
            by_candidate[str(job["candidate"])][str(job["split"])].append(summary)

        feasible = []
        runtime_medians = {}
        for candidate, splits in by_candidate.items():
            calibration = splits["calibration"]
            if calibration and all(item["guardrails"]["passed"] for item in calibration):
                feasible.append(candidate)
                runtime_medians[candidate] = float(np.median([item["algorithm_seconds"] for item in calibration]))

        selected = min(feasible, key=runtime_medians.__getitem__) if feasible else None
        boundary_parameters = _numeric_boundary_parameters(selected, self.candidates) if selected else []
        validation_passed = bool(
            selected
            and by_candidate[selected]["validation"]
            and all(item["guardrails"]["passed"] for item in by_candidate[selected]["validation"])
        )
        if selected is None:
            status = "no_feasible_candidate"
        elif boundary_parameters and self.guardrails.reject_numeric_boundary:
            status = "expand_boundary_search"
        elif not validation_passed:
            status = "selected_candidate_failed_validation"
        else:
            status = "selected"

        atomic_write_json(
            run_dir / "calibration_selection.json",
            {
                "status": status,
                "selected_candidate": selected if status == "selected" else None,
                "provisional_candidate": selected,
                "selection_metric": "median_algorithm_seconds_on_calibration_split",
                "application_metrics_used_for_selection": False,
                "runtime_medians": runtime_medians,
                "feasible_on_calibration_split": feasible,
                "validation_passed": validation_passed,
                "numeric_boundary_parameters": boundary_parameters,
                "test_seeds_used": False,
            },
        )
        if status != "selected":
            raise ConfigurationError(f"numerical calibration did not produce a frozen profile: {status}")


class NumericalCalibrationExecutor:
    """Run candidate profiles while sharing deterministic warm-up preparation."""

    def __init__(self, study: NumericalCalibrationStudy, application: ApplicationRuntime,
            base_numerics: ExperimentNumerics) -> None:  # fmt: skip
        self.study = study
        self.application = application
        self.base_numerics = base_numerics
        self._candidates = {candidate.name: candidate for candidate in study.candidates}
        self._executors: dict[tuple[str, CalibrationSplit, StudyMethod, bool], BaselineComparisonExecutor] = {}
        self._warmups: dict[WarmupConfig, PreparedWarmup] = {}

    def run(self, job: ExperimentJob, *, on_progress: Callable[[int, int], None] | None = None) -> StudyJobResult:
        if not isinstance(job, CalibrationJob) or job not in self.study.jobs():
            raise ConfigurationError("job is not part of this numerical calibration study")
        candidate = self._candidates[job.candidate]
        executor_key = (candidate.name, job.split, job.method, job.warmup_enabled)
        executor = self._executors.get(executor_key)
        if executor is None:
            executor = self._make_executor(candidate, job.split, job.method, job.warmup_enabled)
            self._executors[executor_key] = executor

        started = perf_counter()
        try:
            result = executor.run(StudyJob(job.method, job.warmup_enabled, job.seed), on_progress=on_progress)
        except ConvergenceError as error:
            return self._rejected_nonconvergent_candidate(job, candidate, error, elapsed=perf_counter() - started)
        summary = dict(result.summary)
        summary["job"] = job.as_dict()
        summary["numerical_candidate"] = asdict(candidate)
        guardrail_metrics = self._guardrail_metrics(result)
        summary["guardrails"] = {
            "passed": all(
                guardrail_metrics[name] <= getattr(self.study.guardrails, f"max_{name}") for name in guardrail_metrics
            ),
            "metrics": guardrail_metrics,
            "limits": {name: getattr(self.study.guardrails, f"max_{name}") for name in guardrail_metrics},
        }
        return StudyJobResult(arrays=result.arrays, summary=summary)

    def _rejected_nonconvergent_candidate(self, job: CalibrationJob, candidate: NumericalCandidate,
            error: ConvergenceError, *, elapsed: float) -> StudyJobResult:  # fmt: skip
        """Record an expected candidate rejection without accepting a bad solution."""

        limits = {
            name: getattr(self.study.guardrails, f"max_{name}")
            for name in ("mle_reference_gap", "policy_reference_gap", "mle_residual", "policy_residual")
        }
        return StudyJobResult(
            arrays={},
            summary={
                "job": job.as_dict(),
                "algorithm_seconds": elapsed,
                "numerical_candidate": asdict(candidate),
                "candidate_execution": {"status": "rejected_nonconvergence", "error": str(error)},
                "guardrails": {"passed": False, "metrics": {}, "limits": limits},
            },
        )

    def _make_executor(self, candidate: NumericalCandidate, split: CalibrationSplit,
            method: StudyMethod, warmup_enabled: bool) -> BaselineComparisonExecutor:  # fmt: skip
        if split == "calibration":
            seed_start = self.study.calibration_seed_start
            seed_count = self.study.calibration_seed_count
        else:
            seed_start = self.study.validation_seed_start
            seed_count = self.study.validation_seed_count
        comparison = BaselineComparisonStudy(
            name=f"{self.study.name}-{candidate.name}",
            seed_start=seed_start,
            seed_count=seed_count,
            total_samples=self.study.total_samples,
            warmup_budget=self.study.warmup_budget,
            evaluation_interval=self.study.evaluation_interval,
            lambda_information_multiplier=self.study.lambda_information_multiplier,
            lambda_warm=self.study.lambda_warm,
            confidence_delta=self.study.confidence_delta,
            methods=(method,),
            warmup_modes=(warmup_enabled,),
        )
        return BaselineComparisonExecutor(
            comparison,
            self.application,
            candidate.create(self.base_numerics.evaluation_policy),
            warmup_cache=self._warmups,
        )

    def _guardrail_metrics(self, result: StudyJobResult) -> dict[str, float]:
        arrays = result.arrays
        dataset = InteractionDataset(arrays["policies"], arrays["follower_actions"].astype(np.int64))
        reference_mle = solve_mle(self.application.problem, dataset, self.study.guardrails.reference_mle)
        final_estimate = arrays["omega_history"][-1]
        mle_reference_gap = float(np.linalg.norm(final_estimate - reference_mle.omega))
        mle_residual = float(np.max(arrays["diagnostic__mle_residual"], initial=0.0))
        policy_residual = float(np.max(arrays["diagnostic__policy_residual"], initial=0.0))
        policy_reference_gap = self._policy_reference_gap(arrays)
        return {
            "mle_reference_gap": mle_reference_gap,
            "policy_reference_gap": policy_reference_gap,
            "mle_residual": mle_residual,
            "policy_residual": policy_residual,
        }

    def _policy_reference_gap(self, arrays: dict[str, np.ndarray]) -> float:
        gains = arrays["diagnostic__information_gain"]
        online_rounds = gains.size
        warmup_size = arrays["policies"].shape[0] - online_rounds
        regularizer = self.study.lambda_information_multiplier * self.application.problem.eta**2
        largest_gap = 0.0
        for offset, selected_gain in enumerate(gains):
            estimate = arrays["omega_history"][offset]
            information = regularizer * np.eye(
                self.application.problem.feature_dimension
            ) + cumulative_fisher_information(
                self.application.problem, arrays["policies"][: warmup_size + offset], estimate
            )
            reference = select_information_policy(
                self.application.problem,
                estimate,
                information,
                self.study.guardrails.reference_policy,
                np.random.default_rng(offset),
            )
            largest_gap = max(largest_gap, reference.information_gain - float(selected_gain))
        return max(0.0, largest_gap)


def _load_summaries(run_dir: Path) -> list[dict[str, object]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summaries = []
    for expected in manifest["expected_jobs"]:
        path = run_dir / "jobs" / expected["job_id"] / "summary.json"
        if not path.exists():
            raise ConfigurationError(f"calibration summary is missing: {path}")
        summaries.append(json.loads(path.read_text()))
    return summaries


def _numeric_boundary_parameters(selected_name: str, candidates: tuple[NumericalCandidate, ...]) -> list[str]:
    """Return varied numeric axes where the selected candidate is an endpoint."""

    flattened = {candidate.name: _flatten_numeric(asdict(candidate)) for candidate in candidates}
    selected = flattened[selected_name]
    boundaries = []
    for path, selected_value in selected.items():
        values = sorted({fields[path] for fields in flattened.values() if path in fields})
        if len(values) >= 3 and selected_value in {values[0], values[-1]}:
            boundaries.append(path)
    return boundaries


def _flatten_numeric(value: object, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, path))
        return flattened
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {}
    return {prefix: float(value)}
