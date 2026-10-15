"""Study definitions independent of Hydra and artifact storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from mairl.algorithm import ActiveIRL, ActiveIRLConfig, ActiveIRLResult
from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig
from mairl.exploration.warmup import PreparedWarmup, prepare_warmup
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.batching import checkpoint_positions, evaluation_checkpoints, run_seed_batch
from mairl_exp.execution.jobs import ExperimentJob, StudyJob, StudyJobResult, StudyMethod
from mairl_exp.studies.baselines import RandomBaselineConfig, RandomExplorationBaseline


@dataclass(frozen=True, slots=True)
class BaselineComparisonStudy:
    """Equal-budget comparison of active and random exploration strategies."""

    name: str = "baseline-comparison"
    seed_start: int = 100
    seed_count: int = 12
    total_samples: int = 180
    warmup_budget: int = 120
    evaluation_interval: int = 10
    lambda_information_multiplier: float = 1.0
    lambda_warm: float = 0.01
    confidence_delta: float = 0.05
    methods: tuple[StudyMethod, ...] = ("active", "fixed_random", "random_each_round")
    warmup_modes: tuple[bool, ...] = (True, False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "warmup_modes", tuple(self.warmup_modes))
        if not self.name:
            raise ConfigurationError("study name must be nonempty")
        if self.seed_count < 1:
            raise ConfigurationError("seed_count must be positive")
        if self.total_samples < 0 or self.warmup_budget < 0:
            raise ConfigurationError("sample budgets must be nonnegative")
        if self.warmup_budget > self.total_samples:
            raise ConfigurationError("warmup_budget cannot exceed total_samples")
        if self.evaluation_interval < 1:
            raise ConfigurationError("evaluation_interval must be positive")
        if not self.lambda_information_multiplier > 0:
            raise ConfigurationError("lambda_information_multiplier must be strictly positive")
        if not self.lambda_warm > 0:
            raise ConfigurationError("lambda_warm must be strictly positive")
        if not 0 < self.confidence_delta < 1:
            raise ConfigurationError("confidence_delta must lie in (0, 1)")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ConfigurationError("methods must be nonempty and unique")
        invalid = set(self.methods) - {"active", "fixed_random", "random_each_round"}
        if invalid:
            raise ConfigurationError(f"unknown study methods: {sorted(invalid)}")
        if not self.warmup_modes or len(set(self.warmup_modes)) != len(self.warmup_modes):
            raise ConfigurationError("warmup_modes must be nonempty and unique")

    def jobs(self) -> tuple[StudyJob, ...]:
        """Return the stable, deterministic execution plan."""

        seeds = range(self.seed_start, self.seed_start + self.seed_count)
        return tuple(
            StudyJob(method, warmup_enabled, seed)
            for warmup_enabled in self.warmup_modes
            for method in self.methods
            for seed in seeds
        )

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> BaselineComparisonExecutor:
        return BaselineComparisonExecutor(self, application, numerics)


class BaselineComparisonExecutor:
    """Prepared execution state reused across jobs in one study."""

    def __init__(self, study: BaselineComparisonStudy, application: ApplicationRuntime, numerics: ExperimentNumerics,
            *, warmup_cache: dict[WarmupConfig, PreparedWarmup] | None = None) -> None:  # fmt: skip
        self.study = study
        self.application = application
        self.numerics = numerics
        self._warmups = warmup_cache if warmup_cache is not None else {}

    def run(self, job: StudyJob, *, on_progress: Callable[[int, int], None] | None = None) -> StudyJobResult:
        """Execute one complete job from its explicit configuration."""

        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this study")
        warmup, prepared = self._prepared_warmup(job.warmup_enabled)
        oracle = self.application.oracle_factory(job.seed)

        def report_progress(samples: int) -> None:
            if on_progress is not None and (
                samples == self.study.total_samples or samples % self.study.evaluation_interval == 0
            ):
                on_progress(samples, self.study.total_samples)

        start = perf_counter()
        if job.method == "active":
            config = ActiveIRLConfig(
                horizon=self.study.total_samples - warmup.effective_budget,
                lambda_information=(self.study.lambda_information_multiplier * self.application.problem.eta**2),
                lambda_warm=self.study.lambda_warm,
                confidence_delta=self.study.confidence_delta,
                mle=self.numerics.mle,
                warmup=warmup,
                policy=self.numerics.active_policy,
                seed=job.seed,
            )
            result = ActiveIRL(self.application.problem, config).run(
                oracle, prepared_warmup=prepared, on_round=lambda item: report_progress(item.total_samples)
            )
            sample_counts = np.asarray(
                [warmup.effective_budget, *(item.total_samples for item in result.diagnostics)], dtype=np.int64
            )
            diagnostic_arrays = self._active_diagnostics(result)
            dataset = result.dataset
        else:
            config = RandomBaselineConfig(
                total_samples=self.study.total_samples,
                method=job.method,
                mle=self.numerics.mle,
                warmup=warmup,
                seed=job.seed,
            )
            result = RandomExplorationBaseline(self.application.problem, config).run(
                oracle, prepared_warmup=prepared, on_sample=report_progress
            )
            sample_counts = result.sample_counts
            diagnostic_arrays = {
                "diagnostic__information_gain": np.empty(0, dtype=np.float64),
                "diagnostic__mle_residual": np.asarray([result.final_mle.projected_gradient_residual]),
                "diagnostic__policy_residual": np.empty(0, dtype=np.float64),
            }
            dataset = result.dataset
        algorithm_seconds = perf_counter() - start
        warmup_design = result.warmup_design
        warmup_eigenvalues = np.linalg.eigvalsh(warmup_design.coverage_matrix)

        checkpoints = evaluation_checkpoints(
            warmup.effective_budget, self.study.total_samples, self.study.evaluation_interval
        )
        estimates = result.omega_history[checkpoint_positions(sample_counts, checkpoints)]
        metrics = self.application.evaluate(estimates, job.seed)
        arrays: dict[str, np.ndarray] = {
            "trace_sample_counts": sample_counts,
            "omega_history": result.omega_history,
            "evaluation_sample_counts": checkpoints,
            "policies": dataset.policies,
            "follower_actions": dataset.follower_actions,
            "warmup__design_weights": warmup_design.weights,
            "warmup__support_actions": warmup_design.support_actions,
            "warmup__allocation_counts": warmup_design.allocation_counts,
            "warmup__coverage_eigenvalues": warmup_eigenvalues,
            **diagnostic_arrays,
        }
        arrays.update({f"metric__{name}": values for name, values in metrics.items()})
        return StudyJobResult(
            arrays=arrays,
            summary={
                "job": job.as_dict(),
                "algorithm_seconds": algorithm_seconds,
                "samples": dataset.size,
                "final_metrics": {name: float(values[-1]) for name, values in metrics.items()},
                "warmup": {
                    "enabled": job.warmup_enabled,
                    "samples": warmup.effective_budget,
                    "support_size": warmup_design.support_size,
                    "design_weights": warmup_design.weights.tolist(),
                    "allocation_counts": warmup_design.allocation_counts.tolist(),
                    "coverage_eigenvalues": warmup_eigenvalues.tolist(),
                },
            },
        )

    def batch_key(self, job: StudyJob) -> tuple[StudyMethod, bool]:
        """Return the compatibility key for vectorized seeded execution."""

        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this study")
        return job.method, job.warmup_enabled

    def run_many(self, jobs: tuple[StudyJob, ...], *, on_progress: Callable[[StudyJob, int, int], None] | None = None,
    ) -> tuple[StudyJobResult, ...]:  # fmt: skip
        """Execute compatible seeded jobs with a leading vectorized dimension."""

        if not jobs:
            return ()
        key = self.batch_key(jobs[0])
        if any(self.batch_key(job) != key for job in jobs):
            raise ConfigurationError("batched jobs must share method and warm-up mode")
        if self.application.problem.is_degenerate:
            return tuple(self.run(job) for job in jobs)
        method, warmup_enabled = key
        _, prepared = self._prepared_warmup(warmup_enabled)

        def report(seed_index: int, samples: int, total: int) -> None:
            if on_progress is not None:
                on_progress(jobs[seed_index], samples, total)

        return run_seed_batch(
            method=method,
            warmup_enabled=warmup_enabled,
            seeds=tuple(job.seed for job in jobs),
            total_samples=self.study.total_samples,
            evaluation_interval=self.study.evaluation_interval,
            lambda_information_multiplier=self.study.lambda_information_multiplier,
            application=self.application,
            numerics=self.numerics,
            prepared_warmup=prepared,
            on_progress=report,
        )

    def _prepared_warmup(self, enabled: bool) -> tuple[WarmupConfig, PreparedWarmup]:
        warmup = self.numerics.warmup.create(budget=self.study.warmup_budget, enabled=enabled)
        prepared = self._warmups.get(warmup)
        if prepared is None:
            prepared = prepare_warmup(self.application.problem, warmup)
            self._warmups[warmup] = prepared
        return warmup, prepared

    @staticmethod
    def _active_diagnostics(result: ActiveIRLResult) -> dict[str, np.ndarray]:
        return {
            "diagnostic__information_gain": np.asarray([item.information_gain for item in result.diagnostics]),
            "diagnostic__mle_residual": np.asarray(
                [item.mle_projected_gradient_residual for item in result.diagnostics]
            ),
            "diagnostic__policy_residual": np.asarray(
                [item.policy_projected_gradient_residual for item in result.diagnostics]
            ),
            "diagnostic__mle_iterations": np.asarray(
                [item.mle_iterations for item in result.diagnostics], dtype=np.int64
            ),
        }


class DelegatingSeedExecutor:
    """Run a meta-study's jobs through cached baseline-comparison delegates.

    Sweeps and tuning grids execute their jobs as ordinary
    baseline-comparison runs at one grid point. Subclasses supply only what varies:
    how a job maps to a delegate, how it maps to a :class:`StudyJob`, and what the
    stored summary gains.
    """

    batch_mismatch_message = "batched jobs must share one delegate"

    def __init__(self, study: object, application: ApplicationRuntime, numerics: ExperimentNumerics) -> None:
        self.study = study
        self.application = application
        self.numerics = numerics
        self._warmups: dict[WarmupConfig, PreparedWarmup] = {}
        self._delegates: dict[object, BaselineComparisonExecutor] = {}

    def batch_key(self, job: ExperimentJob) -> object:
        """Return the grouping key; jobs sharing one may be vectorized together."""

        raise NotImplementedError

    def _delegate_key(self, job: ExperimentJob) -> object:
        """Return the delegate cache key, by default the batch key."""

        return self.batch_key(job)

    def _baseline_study(self, key: object) -> BaselineComparisonStudy:
        raise NotImplementedError

    def _delegate_job(self, job: ExperimentJob) -> StudyJob:
        raise NotImplementedError

    def _adapt(self, result: StudyJobResult, job: ExperimentJob) -> StudyJobResult:
        """Restamp a delegated result with this study's own job identity."""

        return replace(result, summary={**result.summary, "job": job.as_dict()})

    def _delegate(self, key: object) -> BaselineComparisonExecutor:
        delegate = self._delegates.get(key)
        if delegate is None:
            delegate = BaselineComparisonExecutor(
                self._baseline_study(key), self.application, self.numerics, warmup_cache=self._warmups
            )
            self._delegates[key] = delegate
        return delegate

    def run(self, job: ExperimentJob, *, on_progress: Callable[[int, int], None] | None = None) -> StudyJobResult:
        delegate = self._delegate(self._delegate_key(job))
        result = delegate.run(self._delegate_job(job), on_progress=on_progress)
        return self._adapt(result, job)

    def run_many(
        self, jobs: tuple[ExperimentJob, ...], *, on_progress: Callable[[ExperimentJob, int, int], None] | None = None
    ) -> tuple[StudyJobResult, ...]:
        if not jobs:
            return ()
        key = self.batch_key(jobs[0])
        if any(self.batch_key(job) != key for job in jobs):
            raise ConfigurationError(self.batch_mismatch_message)
        jobs_by_seed = {job.seed: job for job in jobs}

        def report(delegated: StudyJob, samples: int, total: int) -> None:
            if on_progress is not None:
                on_progress(jobs_by_seed[delegated.seed], samples, total)

        results = self._delegate(self._delegate_key(jobs[0])).run_many(
            tuple(self._delegate_job(job) for job in jobs), on_progress=report
        )
        return tuple(self._adapt(result, job) for result, job in zip(results, jobs, strict=True))
