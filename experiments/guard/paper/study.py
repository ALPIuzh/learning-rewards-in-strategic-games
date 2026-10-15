"""Nine-protocol comparison study for the Lobeke flagship experiment.

Compares Active IRL, a fixed-expert MLE, Liao's paired-action zero-sum OLS, the
follower-only ("unaware of zero-sum structure") OLS baseline, and the two
Stackelberg random baselines (fixed leader policy drawn once, or redrawn every
round), all measured on one follower-sample-count axis.

The three Stackelberg-exploration methods (active/fixed_random/random_each_round)
delegate to :class:`mairl_exp.studies.comparison.BaselineComparisonExecutor` unchanged, each
with and without warm-up: their warm-up is a genuine D-optimal *data-collection*
phase that uses a different leader policy for its early samples, and those
samples are used in the estimate.

The fixed-expert MLE and two OLS methods have no warm-up variant.  MLE samples
under the source Stackelberg-optimal leader, the natural expert for the deployed
leader objective.  The OLS methods instead sample a joint-QRE expert because
their equilibrium equations require it.  Every method keeps its own fixed
leader policy throughout, so withholding an early slice of i.i.d. samples would
not test any real warm-up mechanism.  Each fixed-policy method therefore gets
exactly one curve, fit on ``follower_actions[:checkpoint]`` from zero.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Literal

import numpy as np

from experiments.guard.paper.application import LobekeZeroSumApplication  # noqa: F401  (documented dependency)
from experiments.guard.paper.rewards import is_unit_interval_reward_metadata
from experiments.zero_sum_comparison.application import ZeroSumApplicationRuntime
from mairl import (
    EstimateOutsideBallWarning,
    FixedLeaderOLSConfig,
    FrequencySmoothingWarning,
    fit_fixed_leader_ols,
    fit_liao_zero_sum_paired_actions,
)
from mairl.core.math import follower_probabilities
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.exceptions import ConfigurationError
from mairl.inference.mle import solve_mle_batch
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.jobs import StudyJob, StudyJobResult
from mairl_exp.studies.comparison import BaselineComparisonExecutor, BaselineComparisonStudy

ComparisonMethod = Literal[
    "active", "fixed_expert_mle", "zero_sum_ols", "follower_ols", "fixed_random", "random_each_round"
]
_STACKELBERG_METHODS: tuple[ComparisonMethod, ...] = ("active", "fixed_random", "random_each_round")
_FIXED_EXPERT_MLE_METHODS: tuple[ComparisonMethod, ...] = ("fixed_expert_mle",)
_OLS_METHODS: tuple[ComparisonMethod, ...] = ("zero_sum_ols", "follower_ols")


@dataclass(frozen=True, slots=True)
class LobekeComparisonJob:
    """One method/seed/warm-up execution in the shared comparison."""

    method: ComparisonMethod
    warmup_enabled: bool
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        warmup = "warmup" if self.warmup_enabled else "no-warmup"
        return f"{warmup}/{self.method}/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class LobekeComparisonStudy:
    """Equal-budget nine-protocol comparison on the Lobeke zero-sum game."""

    ols: FixedLeaderOLSConfig
    name: str = "guard-lobeke-paper"
    seed_start: int = 100
    seed_count: int = 30
    total_samples: int = 1600
    warmup_budget: int = 100
    evaluation_interval: int = 40
    warmup_lambda_information_multiplier: float = 1.0
    no_warmup_lambda_information_multiplier: float = 1.0
    lambda_warm: float = 0.01
    fixed_expert_mle_ridge: float = 1.0e-4
    confidence_delta: float = 0.05
    methods: tuple[ComparisonMethod, ...] = (
        "active",
        "fixed_expert_mle",
        "zero_sum_ols",
        "follower_ols",
        "fixed_random",
        "random_each_round",
    )
    warmup_modes: tuple[bool, ...] = (True, False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "warmup_modes", tuple(self.warmup_modes))
        if not self.name:
            raise ConfigurationError("study name must be nonempty")
        if self.seed_count < 2:
            raise ConfigurationError("comparison requires at least two evaluation seeds")
        if self.total_samples < 0 or self.warmup_budget < 0:
            raise ConfigurationError("sample budgets must be nonnegative")
        if self.warmup_budget > self.total_samples:
            raise ConfigurationError("warmup_budget cannot exceed total_samples")
        if self.evaluation_interval < 1:
            raise ConfigurationError("evaluation_interval must be positive")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ConfigurationError("methods must be nonempty and unique")
        invalid = set(self.methods) - set(_STACKELBERG_METHODS) - set(_OLS_METHODS)
        invalid -= set(_FIXED_EXPERT_MLE_METHODS)
        if invalid:
            raise ConfigurationError(f"unknown comparison methods: {sorted(invalid)}")
        if not self.warmup_modes or len(set(self.warmup_modes)) != len(self.warmup_modes):
            raise ConfigurationError("warmup_modes must be nonempty and unique")
        for name, value in (
            ("warmup_lambda_information_multiplier", self.warmup_lambda_information_multiplier),
            ("no_warmup_lambda_information_multiplier", self.no_warmup_lambda_information_multiplier),
        ):
            if not value > 0:
                raise ConfigurationError(f"{name} must be strictly positive")
        if self.fixed_expert_mle_ridge < 0:
            raise ConfigurationError("fixed_expert_mle_ridge must be nonnegative")

    def jobs(self) -> tuple[LobekeComparisonJob, ...]:
        """Exploration methods cross with warm-up modes; fixed-policy methods do not.

        Fixed-expert MLE and OLS each retain their method-specific expert policy, so
        their jobs carry ``warmup_enabled=False`` regardless of ``warmup_modes``.
        """

        seeds = range(self.seed_start, self.seed_start + self.seed_count)
        stackelberg_methods = tuple(m for m in self.methods if m in _STACKELBERG_METHODS)
        fixed_expert_methods = tuple(m for m in self.methods if m in _FIXED_EXPERT_MLE_METHODS + _OLS_METHODS)
        return tuple(
            LobekeComparisonJob(method, warmup_enabled, seed)
            for warmup_enabled in self.warmup_modes
            for method in stackelberg_methods
            for seed in seeds
        ) + tuple(LobekeComparisonJob(method, False, seed) for method in fixed_expert_methods for seed in seeds)

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> LobekeComparisonExecutor:
        if not isinstance(application, ZeroSumApplicationRuntime):
            raise ConfigurationError("Lobeke comparison requires a ZeroSumApplicationRuntime")
        return LobekeComparisonExecutor(self, application, numerics)

    def _stackelberg_study(self, warmup_enabled: bool) -> BaselineComparisonStudy:
        stackelberg_methods = tuple(m for m in self.methods if m in _STACKELBERG_METHODS)
        multiplier = (
            self.warmup_lambda_information_multiplier
            if warmup_enabled
            else self.no_warmup_lambda_information_multiplier
        )
        return BaselineComparisonStudy(
            name=f"{self.name}-stackelberg",
            seed_start=self.seed_start,
            seed_count=self.seed_count,
            total_samples=self.total_samples,
            warmup_budget=self.warmup_budget,
            evaluation_interval=self.evaluation_interval,
            lambda_information_multiplier=multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=stackelberg_methods,
            warmup_modes=(warmup_enabled,),
        )


class LobekeComparisonExecutor:
    """Delegates Stackelberg methods to the shared executor; fits OLS jobs directly."""

    def __init__(self, study: LobekeComparisonStudy, application: ZeroSumApplicationRuntime,
            numerics: ExperimentNumerics) -> None:  # fmt: skip
        self.study = study
        self.application = application
        self.numerics = numerics
        warmup_cache = {}
        has_stackelberg_methods = any(method in _STACKELBERG_METHODS for method in study.methods)
        self._stackelberg = {
            enabled: BaselineComparisonExecutor(
                study._stackelberg_study(enabled), application, numerics, warmup_cache=warmup_cache
            )
            for enabled in study.warmup_modes
            if has_stackelberg_methods
        }

    def _reward_metric_provenance(self) -> dict[str, object]:
        normalization = self.application.metadata.get("leader_reward_normalization")
        if normalization is None:
            return {}
        if not isinstance(normalization, dict) or not is_unit_interval_reward_metadata(normalization):
            raise ConfigurationError("application is missing leader-reward metric provenance")
        return {"leader_reward_normalization": normalization}

    def batch_key(self, job: LobekeComparisonJob) -> tuple[ComparisonMethod, bool]:
        self._validate(job)
        return job.method, job.warmup_enabled

    def run(self, job: LobekeComparisonJob, *, on_progress: Callable[[int, int], None] | None = None) -> StudyJobResult:
        self._validate(job)
        if job.method in _STACKELBERG_METHODS:
            result = self._stackelberg[job.warmup_enabled].run(
                StudyJob(job.method, job.warmup_enabled, job.seed), on_progress=on_progress
            )
            return replace(result, summary={**result.summary, "job": job.as_dict(), **self._reward_metric_provenance()})
        if job.method in _FIXED_EXPERT_MLE_METHODS:
            return self._fixed_expert_mle_results(
                (job,),
                on_progress=(None if on_progress is None else lambda _job, current, total: on_progress(current, total)),
            )[0]
        return self._ols_result(job, on_progress=on_progress)

    def run_many(self, jobs: tuple[LobekeComparisonJob, ...],
            *, on_progress: Callable[[LobekeComparisonJob, int, int], None] | None = None,
    ) -> tuple[StudyJobResult, ...]:  # fmt: skip
        if not jobs:
            return ()
        method, warmup_enabled = self.batch_key(jobs[0])
        if any(self.batch_key(job) != (method, warmup_enabled) for job in jobs):
            raise ConfigurationError("comparison batches must share method and warm-up mode")
        if method in _FIXED_EXPERT_MLE_METHODS:
            return self._fixed_expert_mle_results(jobs, on_progress=on_progress)
        if method in _OLS_METHODS:
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

        def report(delegated_job: StudyJob, samples: int, total: int) -> None:
            if on_progress is not None:
                on_progress(jobs_by_seed[delegated_job.seed], samples, total)

        delegated = tuple(StudyJob(job.method, job.warmup_enabled, job.seed) for job in jobs)
        results = self._stackelberg[warmup_enabled].run_many(delegated, on_progress=report)
        return tuple(
            replace(result, summary={**result.summary, "job": job.as_dict(), **self._reward_metric_provenance()})
            for result, job in zip(results, jobs, strict=True)
        )

    def _fixed_expert_checkpoints(self) -> np.ndarray:
        values = np.arange(
            self.study.evaluation_interval, self.study.total_samples + 1, self.study.evaluation_interval, dtype=np.int64
        )
        if values.size == 0 or values[-1] != self.study.total_samples:
            values = np.append(values, self.study.total_samples)
        return values

    def _fixed_expert_mle_results(self, jobs: tuple[LobekeComparisonJob, ...],
            *, on_progress: Callable[[LobekeComparisonJob, int, int], None] | None,
    ) -> tuple[StudyJobResult, ...]:  # fmt: skip
        """Fit growing MLEs under the fixed source-task expert, batched over seeds."""

        seed_count = len(jobs)
        total_samples = self.study.total_samples
        problem = self.application.problem
        expert_policy = self.application.fixed_mle_leader_policy
        follower_probability = follower_probabilities(problem, expert_policy, self.application.true_omega)
        policies = np.broadcast_to(expert_policy, (seed_count, total_samples, problem.n_leader_actions))
        follower_actions = np.stack(
            [
                experiment_rng(job.seed, RandomPurpose.EXPERT_FOLLOWER).choice(
                    problem.n_follower_actions, size=total_samples, p=follower_probability
                )
                for job in jobs
            ]
        )
        checkpoints = self._fixed_expert_checkpoints()
        estimates = np.empty((seed_count, checkpoints.size, problem.feature_dimension), dtype=np.float64)
        residuals = np.empty((seed_count, checkpoints.size), dtype=np.float64)
        iterations = np.empty((seed_count, checkpoints.size), dtype=np.int64)
        previous = np.zeros((seed_count, problem.feature_dimension), dtype=np.float64)
        mle_config = replace(self.numerics.mle, ridge=self.study.fixed_expert_mle_ridge)

        started = perf_counter()
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            fits = solve_mle_batch(
                problem, policies[:, :checkpoint], follower_actions[:, :checkpoint], mle_config, initial=previous
            )
            previous = np.stack([fit.omega for fit in fits])
            estimates[:, checkpoint_index] = previous
            residuals[:, checkpoint_index] = [fit.projected_gradient_residual for fit in fits]
            iterations[:, checkpoint_index] = [fit.iterations for fit in fits]
            if on_progress is not None:
                for job in jobs:
                    on_progress(job, int(checkpoint), total_samples)
        elapsed = perf_counter() - started

        fixed_policies = np.broadcast_to(expert_policy, (total_samples, problem.n_leader_actions)).copy()
        results = []
        for seed_index, job in enumerate(jobs):
            metrics = self.application.evaluate(estimates[seed_index], job.seed)
            arrays: dict[str, np.ndarray] = {
                "trace_sample_counts": checkpoints,
                "omega_history": estimates[seed_index],
                "evaluation_sample_counts": checkpoints,
                "policies": fixed_policies,
                "follower_actions": follower_actions[seed_index],
                "diagnostic__mle_residual": residuals[seed_index],
                "diagnostic__mle_iterations": iterations[seed_index],
            }
            arrays.update({f"metric__{name}": values for name, values in metrics.items()})
            results.append(
                StudyJobResult(
                    arrays=arrays,
                    summary={
                        "job": job.as_dict(),
                        "algorithm_seconds": elapsed / seed_count,
                        "vectorized_batch_seconds": elapsed,
                        "samples": total_samples,
                        "final_metrics": {name: float(values[-1]) for name, values in metrics.items()},
                        "fixed_leader_policy": expert_policy.tolist(),
                        "expert_definition": self.application.fixed_mle_expert_definition,
                        "estimator_mode": "fixed_expert_mle",
                        "ridge": self.study.fixed_expert_mle_ridge,
                        **self._reward_metric_provenance(),
                        "warmup": {
                            "enabled": False,
                            "samples": 0,
                            "not_applicable": ("every sample is collected under the fixed source-task expert"),
                        },
                    },
                )
            )
        return tuple(results)

    def _ols_result(self, job: LobekeComparisonJob,
            *, on_progress: Callable[[int, int], None] | None) -> StudyJobResult:  # fmt: skip
        follower_actions = experiment_rng(job.seed, RandomPurpose.EXPERT_FOLLOWER).choice(
            self.application.problem.n_follower_actions,
            size=self.study.total_samples,
            p=self.application.expert.min_policy,
        )
        leader_actions = experiment_rng(job.seed, RandomPurpose.EXPERT_LEADER).choice(
            self.application.game.n_max_actions, size=self.study.total_samples, p=self.application.expert.max_policy
        )
        checkpoints = self._fixed_expert_checkpoints()
        feature_dimension = self.application.problem.feature_dimension

        estimates = np.empty((checkpoints.size, feature_dimension), dtype=np.float64)
        ranks = np.empty(checkpoints.size, dtype=np.int64)
        residuals = np.empty(checkpoints.size, dtype=np.float64)
        constrained = np.empty(checkpoints.size, dtype=np.bool_)
        converged = np.empty(checkpoints.size, dtype=np.bool_)
        warning_counts = {"frequency_smoothing": 0, "estimate_outside_ball": 0}

        started = perf_counter()
        for index, checkpoint in enumerate(checkpoints):
            usable = follower_actions[:checkpoint]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if job.method == "follower_ols":
                    result = fit_fixed_leader_ols(
                        self.application.problem, self.application.fixed_response, usable, self.study.ols
                    )
                else:
                    result = fit_liao_zero_sum_paired_actions(
                        self.application.game,
                        self.application.expert,
                        leader_actions[:checkpoint],
                        usable,
                        self.study.ols,
                    )
                if result.follower_sample_count != usable.size:
                    raise RuntimeError("OLS result reported an incorrect follower sample count")
                omega = result.omega
                rank = result.rank
                residual = result.residual_norm
                is_constrained = result.constrained
            for item in caught:
                if issubclass(item.category, FrequencySmoothingWarning):
                    warning_counts["frequency_smoothing"] += 1
                elif issubclass(item.category, EstimateOutsideBallWarning):
                    warning_counts["estimate_outside_ball"] += 1
                else:
                    warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
            estimates[index] = omega
            ranks[index] = rank
            residuals[index] = residual
            constrained[index] = is_constrained
            converged[index] = True
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
            "diagnostic__converged": converged,
        }
        if job.method == "zero_sum_ols":
            arrays["leader_actions"] = leader_actions
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
                "sampling_protocol": (
                    "follower_actions_only"
                    if job.method == "follower_ols"
                    else "paired_empirical_leader_and_follower_actions"
                ),
                "observed_action_counts": {
                    "follower": self.study.total_samples,
                    "leader": (self.study.total_samples if job.method == "zero_sum_ols" else 0),
                },
                "final_rank": int(ranks[-1]),
                "warnings": warning_counts,
                "converged_checkpoints": int(np.sum(converged)),
                "total_checkpoints": int(converged.size),
                **self._reward_metric_provenance(),
                "warmup": {
                    "enabled": False,
                    "samples": 0,
                    "not_applicable": (
                        "OLS samples every action from one fixed policy throughout; "
                        "there is no policy to build a warm-up phase out of"
                    ),
                },
            },
        )

    def _validate(self, job: LobekeComparisonJob) -> None:
        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this comparison")
