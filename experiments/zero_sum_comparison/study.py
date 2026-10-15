"""Resumable studies for warm-up tuning and the final zero-sum comparison."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from experiments.zero_sum_comparison.application import ZeroSumApplicationRuntime
from mairl import (
    EstimateOutsideBallWarning,
    FixedLeaderOLSConfig,
    FrequencySmoothingWarning,
    fit_fixed_leader_ols,
    fit_liao_zero_sum_ols,
)
from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.artifacts import atomic_write_json
from mairl_exp.execution.batching import evaluation_checkpoints
from mairl_exp.execution.jobs import StudyJob, StudyJobResult
from mairl_exp.studies.comparison import BaselineComparisonExecutor, BaselineComparisonStudy, DelegatingSeedExecutor
from mairl_exp.studies.selection import completed_summaries, sample_statistics

_MINIMUM_SEEDS = "each tuning candidate requires at least two finite scores"

TuningSplit = Literal["selection", "confirmation"]
ComparisonMethod = Literal["active", "follower_ols", "zero_sum_ols"]


@dataclass(frozen=True, slots=True)
class WarmupTuningJob:
    """One Active IRL warm-up candidate on one protected seed split."""

    split: TuningSplit
    warmup_budget: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{self.split}/warmup={self.warmup_budget}/active/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class ActiveWarmupTuningStudy:
    """Select on one seed split and confirm the frozen choice on another."""

    name: str
    selection_seed_start: int
    selection_seed_count: int
    confirmation_seed_start: int
    confirmation_seed_count: int
    total_samples: int
    warmup_budgets: tuple[int, ...]
    evaluation_interval: int
    lambda_information_multiplier: float
    lambda_warm: float
    confidence_delta: float
    tuning_metric: str = "fixed_leader_fisher_error"
    confirmation_standard_errors: float = 2.0
    minimum_seeds_per_split: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_budgets", tuple(self.warmup_budgets))
        if not self.name or not self.tuning_metric:
            raise ConfigurationError("study name and tuning metric must be nonempty")
        if min(self.selection_seed_count, self.confirmation_seed_count) < (self.minimum_seeds_per_split):
            raise ConfigurationError("each warm-up tuning split has fewer seeds than the configured minimum")
        selection_seeds = set(range(self.selection_seed_start, self.selection_seed_start + self.selection_seed_count))
        confirmation_seeds = set(
            range(self.confirmation_seed_start, self.confirmation_seed_start + self.confirmation_seed_count)
        )
        if selection_seeds & confirmation_seeds:
            raise ConfigurationError("warm-up selection and confirmation seeds must be disjoint")
        if (
            not self.warmup_budgets
            or len(set(self.warmup_budgets)) != len(self.warmup_budgets)
            or tuple(sorted(self.warmup_budgets)) != self.warmup_budgets
        ):
            raise ConfigurationError("warmup_budgets must be nonempty and strictly increasing")
        if any(budget < 0 or budget > self.total_samples for budget in self.warmup_budgets):
            raise ConfigurationError("every warm-up budget must lie in [0, total_samples]")
        if 0 not in self.warmup_budgets:
            raise ConfigurationError("warm-up tuning must include the no-warm-up baseline")
        if self.confirmation_standard_errors <= 0:
            raise ConfigurationError("confirmation_standard_errors must be positive")
        self._baseline_study("selection", self.warmup_budgets[0])

    def jobs(self) -> tuple[WarmupTuningJob, ...]:
        return tuple(
            WarmupTuningJob(split, budget, seed)
            for split, start, count in (
                ("selection", self.selection_seed_start, self.selection_seed_count),
                ("confirmation", self.confirmation_seed_start, self.confirmation_seed_count),
            )
            for budget in self.warmup_budgets
            for seed in range(start, start + count)
        )

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> ActiveWarmupTuningExecutor:
        return ActiveWarmupTuningExecutor(self, application, numerics)

    def _baseline_study(self, split: TuningSplit, budget: int) -> BaselineComparisonStudy:
        if split == "selection":
            seed_start = self.selection_seed_start
            seed_count = self.selection_seed_count
        else:
            seed_start = self.confirmation_seed_start
            seed_count = self.confirmation_seed_count
        return BaselineComparisonStudy(
            name=f"{self.name}-{split}-warmup-{budget}",
            seed_start=seed_start,
            seed_count=seed_count,
            total_samples=self.total_samples,
            warmup_budget=budget,
            evaluation_interval=self.evaluation_interval,
            lambda_information_multiplier=self.lambda_information_multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=("active",),
            warmup_modes=(budget > 0,),
        )

    def finalize(self, run_dir: Path) -> None:
        """Freeze one candidate without using confirmation seeds for selection."""

        grouped: dict[TuningSplit, dict[int, list[float]]] = {
            "selection": {budget: [] for budget in self.warmup_budgets},
            "confirmation": {budget: [] for budget in self.warmup_budgets},
        }
        for summary in completed_summaries(
            run_dir, require_nonempty=True, empty_message="tuning finalization found no completed jobs"
        ):
            job = summary["job"]
            split = str(job["split"])
            if split not in grouped:
                raise ConfigurationError(f"unexpected tuning split {split!r}")
            metric = float(summary["final_metrics"][self.tuning_metric])
            grouped[split][int(job["warmup_budget"])].append(metric)

        statistics = {
            split: {
                str(budget): sample_statistics(values, minimum_seeds_message=_MINIMUM_SEEDS)
                for budget, values in budgets.items()
            }
            for split, budgets in grouped.items()
        }
        selection = statistics["selection"]
        raw_best_budget = min(self.warmup_budgets, key=lambda budget: float(selection[str(budget)]["mean"]))
        raw_best = selection[str(raw_best_budget)]
        one_se_threshold = float(raw_best["mean"]) + float(raw_best["standard_error"])
        eligible = tuple(
            budget for budget in self.warmup_budgets if float(selection[str(budget)]["mean"]) <= one_se_threshold
        )
        selected_budget = min(eligible)

        confirmation = statistics["confirmation"]
        selected_confirmation = confirmation[str(selected_budget)]
        raw_best_confirmation = confirmation[str(raw_best_budget)]
        tie_break_margin = self.confirmation_standard_errors * np.hypot(
            float(selected_confirmation["standard_error"]), float(raw_best_confirmation["standard_error"])
        )
        tie_break_confirmed = bool(
            float(selected_confirmation["mean"]) <= float(raw_best_confirmation["mean"]) + float(tie_break_margin)
        )
        no_warmup_confirmation = confirmation["0"]
        baseline_margin = self.confirmation_standard_errors * np.hypot(
            float(selected_confirmation["standard_error"]), float(no_warmup_confirmation["standard_error"])
        )
        no_harm_confirmed = bool(
            float(selected_confirmation["mean"]) <= float(no_warmup_confirmation["mean"]) + float(baseline_margin)
        )
        confirmation_passed = tie_break_confirmed and no_harm_confirmed
        status = "selected" if confirmation_passed else "selection_failed_confirmation"
        atomic_write_json(
            run_dir / "warmup_selection.json",
            {
                "status": status,
                "selected_warmup_budget": (selected_budget if confirmation_passed else None),
                "provisional_warmup_budget": selected_budget,
                "raw_selection_best_budget": raw_best_budget,
                "one_standard_error_threshold": one_se_threshold,
                "one_standard_error_eligible_budgets": list(eligible),
                "tie_break": "smallest warm-up budget among one-SE candidates",
                "tuning_metric": self.tuning_metric,
                "selection_statistics": statistics["selection"],
                "confirmation_statistics": statistics["confirmation"],
                "confirmation_rule": (
                    "provisional mean must be within "
                    f"{self.confirmation_standard_errors:g} combined standard errors "
                    "of both the raw selection winner and the predeclared no-warm-up baseline"
                ),
                "tie_break_confirmed": tie_break_confirmed,
                "no_harm_vs_no_warmup_confirmed": no_harm_confirmed,
                "confirmation_passed": confirmation_passed,
                "selected_is_physical_boundary": selected_budget in (0, self.total_samples),
                "selection_and_confirmation_seeds_disjoint": True,
                "test_seeds_used": False,
            },
        )
        if not confirmation_passed:
            raise ConfigurationError("warm-up selection failed its confirmation split")


class ActiveWarmupTuningExecutor(DelegatingSeedExecutor):
    """Reuse the vectorized Active IRL executor for each split and budget."""

    batch_mismatch_message = "tuning batches must share split and warm-up budget"

    def batch_key(self, job: WarmupTuningJob) -> tuple[TuningSplit, int]:
        self._validate(job)
        return job.split, job.warmup_budget

    def _baseline_study(self, key: tuple[TuningSplit, int]) -> BaselineComparisonStudy:
        return self.study._baseline_study(*key)

    @staticmethod
    def _delegate_job(job: WarmupTuningJob) -> StudyJob:
        return StudyJob("active", job.warmup_budget > 0, job.seed)

    def _validate(self, job: WarmupTuningJob) -> None:
        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this tuning study")


@dataclass(frozen=True, slots=True)
class ZeroSumComparisonJob:
    """One estimator and evaluation seed at the shared follower budget."""

    method: ComparisonMethod
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{self.method}/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class ZeroSumEstimatorComparisonStudy:
    """Compare Active IRL and both fixed-leader OLS equation sets."""

    ols: FixedLeaderOLSConfig
    name: str
    seed_start: int
    seed_count: int
    total_samples: int
    warmup_budget: int
    evaluation_interval: int
    lambda_information_multiplier: float
    lambda_warm: float
    confidence_delta: float
    methods: tuple[ComparisonMethod, ...] = ("active", "follower_ols", "zero_sum_ols")

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        if (
            not self.methods
            or len(set(self.methods)) != len(self.methods)
            or set(self.methods) != {"active", "follower_ols", "zero_sum_ols"}
        ):
            raise ConfigurationError("comparison must contain active, follower_ols, and zero_sum_ols exactly once")
        if self.seed_count < 2:
            raise ConfigurationError("comparison requires at least two evaluation seeds")
        self._active_study()

    def jobs(self) -> tuple[ZeroSumComparisonJob, ...]:
        return tuple(
            ZeroSumComparisonJob(method, seed)
            for method in self.methods
            for seed in range(self.seed_start, self.seed_start + self.seed_count)
        )

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> ZeroSumComparisonExecutor:
        if not isinstance(application, ZeroSumApplicationRuntime):
            raise ConfigurationError("zero-sum comparison requires ZeroSumApplicationRuntime")
        return ZeroSumComparisonExecutor(self, application, numerics)

    def _active_study(self) -> BaselineComparisonStudy:
        return BaselineComparisonStudy(
            name=f"{self.name}-active",
            seed_start=self.seed_start,
            seed_count=self.seed_count,
            total_samples=self.total_samples,
            warmup_budget=self.warmup_budget,
            evaluation_interval=self.evaluation_interval,
            lambda_information_multiplier=self.lambda_information_multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=("active",),
            warmup_modes=(self.warmup_budget > 0,),
        )


class ZeroSumComparisonExecutor:
    """Vectorize Active IRL and keep the two tiny OLS fits deterministic."""

    def __init__(self, study: ZeroSumEstimatorComparisonStudy, application: ZeroSumApplicationRuntime,
            numerics: ExperimentNumerics) -> None:  # fmt: skip
        self.study = study
        self.application = application
        self.numerics = numerics
        self._active = BaselineComparisonExecutor(study._active_study(), application, numerics)

    def batch_key(self, job: ZeroSumComparisonJob) -> ComparisonMethod:
        self._validate(job)
        return job.method

    def run(self, job: ZeroSumComparisonJob,
            *, on_progress: Callable[[int, int], None] | None = None) -> StudyJobResult:  # fmt: skip
        self._validate(job)
        if job.method == "active":
            result = self._active.run(
                StudyJob("active", self.study.warmup_budget > 0, job.seed), on_progress=on_progress
            )
            return self._active_result(result, job)
        return self._ols_result(job, on_progress=on_progress)

    def run_many(self, jobs: tuple[ZeroSumComparisonJob, ...],
            *, on_progress: Callable[[ZeroSumComparisonJob, int, int], None] | None = None,
    ) -> tuple[StudyJobResult, ...]:  # fmt: skip
        if not jobs:
            return ()
        method = self.batch_key(jobs[0])
        if any(self.batch_key(job) != method for job in jobs):
            raise ConfigurationError("comparison batches must share an estimator")
        if method != "active":
            return tuple(
                self._ols_result(
                    job,
                    on_progress=(
                        None
                        if on_progress is None
                        else lambda current, total, item=job: on_progress(item, current, total)
                    ),
                )
                for job in jobs
            )

        jobs_by_seed = {job.seed: job for job in jobs}

        def report(job: StudyJob, samples: int, total: int) -> None:
            if on_progress is not None:
                on_progress(jobs_by_seed[job.seed], samples, total)

        delegated = tuple(StudyJob("active", self.study.warmup_budget > 0, job.seed) for job in jobs)
        results = self._active.run_many(delegated, on_progress=report)
        return tuple(self._active_result(result, job) for result, job in zip(results, jobs, strict=True))

    def _ols_result(self, job: ZeroSumComparisonJob,
            *, on_progress: Callable[[int, int], None] | None) -> StudyJobResult:  # fmt: skip
        rng = np.random.default_rng(job.seed)
        follower_actions = rng.choice(
            self.application.problem.n_follower_actions,
            size=self.study.total_samples,
            p=self.application.expert.min_policy,
        )
        checkpoints = evaluation_checkpoints(
            self.study.evaluation_interval, self.study.total_samples, self.study.evaluation_interval
        )
        estimates = np.empty((checkpoints.size, self.application.problem.feature_dimension), dtype=np.float64)
        ranks = np.empty(checkpoints.size, dtype=np.int64)
        residuals = np.empty(checkpoints.size, dtype=np.float64)
        constrained = np.empty(checkpoints.size, dtype=np.bool_)
        warning_counts = {"frequency_smoothing": 0, "estimate_outside_ball": 0}

        started = perf_counter()
        for index, checkpoint in enumerate(checkpoints):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if job.method == "follower_ols":
                    result = fit_fixed_leader_ols(
                        self.application.problem,
                        self.application.fixed_response,
                        follower_actions[:checkpoint],
                        self.study.ols,
                    )
                else:
                    result = fit_liao_zero_sum_ols(
                        self.application.game, self.application.expert, follower_actions[:checkpoint], self.study.ols
                    )
            for item in caught:
                if issubclass(item.category, FrequencySmoothingWarning):
                    warning_counts["frequency_smoothing"] += 1
                elif issubclass(item.category, EstimateOutsideBallWarning):
                    warning_counts["estimate_outside_ball"] += 1
                else:
                    warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
            if result.follower_sample_count != checkpoint:
                raise RuntimeError("OLS result reported an incorrect follower sample count")
            estimates[index] = result.omega
            ranks[index] = result.rank
            residuals[index] = result.residual_norm
            constrained[index] = result.constrained
            if on_progress is not None:
                on_progress(int(checkpoint), self.study.total_samples)

        metrics = self.application.evaluate(estimates, job.seed)
        elapsed = perf_counter() - started
        fixed_policies = np.repeat(self.application.expert.max_policy[None, :], self.study.total_samples, axis=0)
        arrays: dict[str, np.ndarray] = {
            "trace_sample_counts": checkpoints,
            "omega_history": estimates,
            "evaluation_sample_counts": checkpoints,
            "policies": fixed_policies,
            "follower_actions": follower_actions,
            "diagnostic__rank": ranks,
            "diagnostic__residual": residuals,
            "diagnostic__constrained": constrained,
        }
        arrays.update({f"metric__{name}": values for name, values in metrics.items()})
        return StudyJobResult(
            arrays=arrays,
            summary={
                "job": job.as_dict(),
                "algorithm_seconds": elapsed,
                "samples": self.study.total_samples,
                "final_metrics": {name: float(values[-1]) for name, values in metrics.items()},
                "fixed_leader_policy": self.application.expert.max_policy.tolist(),
                "equation_mode": ("follower_only" if job.method == "follower_ols" else "zero_sum"),
                "final_rank": int(ranks[-1]),
                "warnings": warning_counts,
                "warmup": {"enabled": False, "samples": 0, "not_applicable": True},
            },
        )

    def _active_result(self, result: StudyJobResult, job: ZeroSumComparisonJob) -> StudyJobResult:
        arrays = dict(result.arrays)
        checkpoints = arrays["evaluation_sample_counts"]
        if checkpoints.size and checkpoints[0] == 0:
            keep = checkpoints > 0
            arrays["evaluation_sample_counts"] = checkpoints[keep]
            for name, values in tuple(arrays.items()):
                if name.startswith("metric__"):
                    arrays[name] = values[keep]
        return replace(result, arrays=arrays, summary={**result.summary, "job": job.as_dict()})

    def _validate(self, job: ZeroSumComparisonJob) -> None:
        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this comparison")
