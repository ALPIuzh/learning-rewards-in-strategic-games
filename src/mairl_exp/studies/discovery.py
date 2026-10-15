"""Protected, resumable joint tuning over warm-up budget and online lambda.

This module is application agnostic. Applications only provide the usual
``ApplicationRuntime``; selection, pairing, guardrails, and confirmation remain
internal experiment infrastructure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.artifacts import atomic_write_json
from mairl_exp.execution.jobs import StudyJob, StudyJobResult
from mairl_exp.studies.comparison import BaselineComparisonStudy, DelegatingSeedExecutor
from mairl_exp.studies.selection import (
    actual_lambdas,
    completed_summaries,
    not_worse,
    paired_statistics,
    sample_statistics,
)


@dataclass(frozen=True, slots=True)
class JointDiscoveryJob:
    """One warm-up/lambda candidate evaluated on one paired discovery seed."""

    warmup_budget: int
    lambda_information_multiplier: float
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"warmup={self.warmup_budget}/lambda={self.lambda_information_multiplier:g}eta2/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class JointWarmupLambdaDiscoveryStudy:
    """Produce a guarded shortlist without claiming a final tuned configuration."""

    name: str
    seed_start: int
    seed_count: int
    final_evaluation_seed_start: int
    final_evaluation_seed_count: int
    total_samples: int
    warmup_budgets: tuple[int, ...]
    lambda_information_multipliers: tuple[float, ...]
    lambda_warm: float
    confidence_delta: float
    reference_warmup_budget: int
    reference_multiplier: float
    candidate_pairs: tuple[tuple[int, float], ...] | None = None
    evaluation_interval: int | None = None
    tuning_metric: str = "stackelberg_suboptimality"
    guardrail_metric: str = "fisher_transfer_error"
    guardrail_standard_errors: float = 2.0
    shortlist_standard_errors: float = 1.0
    minimum_seed_count: int = 16

    def __post_init__(self) -> None:
        budgets = tuple(int(value) for value in self.warmup_budgets)
        multipliers = tuple(float(value) for value in self.lambda_information_multipliers)
        object.__setattr__(self, "warmup_budgets", budgets)
        object.__setattr__(self, "lambda_information_multipliers", multipliers)
        if not self.name or not self.tuning_metric or not self.guardrail_metric:
            raise ConfigurationError("study name and metric names must be nonempty")
        if self.seed_count < self.minimum_seed_count:
            raise ConfigurationError("joint discovery has fewer seeds than its configured minimum")
        discovery_seeds = set(range(self.seed_start, self.seed_start + self.seed_count))
        final_seeds = set(
            range(self.final_evaluation_seed_start, self.final_evaluation_seed_start + self.final_evaluation_seed_count)
        )
        if self.final_evaluation_seed_count < 1:
            raise ConfigurationError("final_evaluation_seed_count must be positive")
        if discovery_seeds & final_seeds:
            raise ConfigurationError("discovery seeds overlap the protected final evaluation")
        if (
            not budgets
            or len(set(budgets)) != len(budgets)
            or tuple(sorted(budgets)) != budgets
            or any(value < 0 or value > self.total_samples for value in budgets)
        ):
            raise ConfigurationError("warmup_budgets must be unique, increasing, and within the total budget")
        if (
            not multipliers
            or len(set(multipliers)) != len(multipliers)
            or tuple(sorted(multipliers)) != multipliers
            or any(value <= 0 for value in multipliers)
        ):
            raise ConfigurationError("lambda multipliers must be unique, increasing, and strictly positive")
        if self.reference_warmup_budget not in budgets:
            raise ConfigurationError("reference warm-up budget must be in the discovery grid")
        if self.reference_multiplier not in multipliers:
            raise ConfigurationError("reference multiplier must be in the discovery grid")
        candidates = (
            tuple((budget, multiplier) for budget in budgets for multiplier in multipliers)
            if self.candidate_pairs is None
            else tuple((int(candidate[0]), float(candidate[1])) for candidate in self.candidate_pairs)
        )
        if not candidates or len(set(candidates)) != len(candidates):
            raise ConfigurationError("candidate_pairs must be nonempty and unique")
        outside_grid = any(budget not in budgets or multiplier not in multipliers for budget, multiplier in candidates)
        if outside_grid:
            raise ConfigurationError("every candidate pair must use declared grid values")
        reference = self.reference_warmup_budget, self.reference_multiplier
        if reference not in candidates:
            raise ConfigurationError("reference configuration must be an evaluated candidate")
        object.__setattr__(self, "candidate_pairs", candidates)
        if self.total_samples < 1 or not self.lambda_warm > 0:
            raise ConfigurationError("total_samples and lambda_warm must be positive")
        if not 0 < self.confidence_delta < 1:
            raise ConfigurationError("confidence_delta must lie in (0, 1)")
        if min(self.guardrail_standard_errors, self.shortlist_standard_errors) <= 0:
            raise ConfigurationError("standard-error multipliers must be positive")
        interval = self.total_samples if self.evaluation_interval is None else self.evaluation_interval
        if interval < 1:
            raise ConfigurationError("evaluation_interval must be positive")
        object.__setattr__(self, "evaluation_interval", interval)
        self._baseline_study(*candidates[0])

    def jobs(self) -> tuple[JointDiscoveryJob, ...]:
        return tuple(
            JointDiscoveryJob(budget, multiplier, seed)
            for budget, multiplier in self.candidate_pairs
            for seed in range(self.seed_start, self.seed_start + self.seed_count)
        )

    def executor(self, application: ApplicationRuntime,
            numerics: ExperimentNumerics) -> JointWarmupLambdaDiscoveryExecutor:  # fmt: skip
        return JointWarmupLambdaDiscoveryExecutor(self, application, numerics)

    def _baseline_study(self, warmup_budget: int, multiplier: float) -> BaselineComparisonStudy:
        return BaselineComparisonStudy(
            name=f"{self.name}-warmup-{warmup_budget}-lambda-{multiplier:g}",
            seed_start=self.seed_start,
            seed_count=self.seed_count,
            total_samples=self.total_samples,
            warmup_budget=warmup_budget,
            evaluation_interval=int(self.evaluation_interval),
            lambda_information_multiplier=multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=("active",),
            warmup_modes=(warmup_budget > 0,),
        )

    def finalize(self, run_dir: Path) -> None:
        summaries = completed_summaries(run_dir)
        expected = len(self.jobs())
        if len(summaries) != expected:
            raise ConfigurationError(f"joint discovery completed {len(summaries)} of {expected} jobs")
        observations = self._observations(summaries)
        statistics = self._statistics(observations)
        reference = self._key(self.reference_warmup_budget, self.reference_multiplier)
        reference_fisher = observations[reference][self.guardrail_metric]
        fisher_differences = {
            key: paired_statistics(metrics[self.guardrail_metric], reference_fisher)
            for key, metrics in observations.items()
        }
        eligible = tuple(
            key for key in observations if not_worse(fisher_differences[key], self.guardrail_standard_errors)
        )
        raw_best = min(eligible, key=lambda key: float(statistics[key][self.tuning_metric]["mean"]))
        raw_best_primary = observations[raw_best][self.tuning_metric]
        primary_differences = {
            key: paired_statistics(observations[key][self.tuning_metric], raw_best_primary) for key in eligible
        }
        shortlist = tuple(
            key for key in eligible if not_worse(primary_differences[key], self.shortlist_standard_errors)
        )
        shortlist_payload = [self._candidate_summary(key, statistics) for key in shortlist]
        boundary_shortlist = any(item["on_grid_boundary"] for item in shortlist_payload)
        atomic_write_json(
            run_dir / "joint_discovery.json",
            {
                "status": "discovery_complete",
                "final_selection_made": False,
                "warmup_budgets": list(self.warmup_budgets),
                "lambda_information_multipliers": list(self.lambda_information_multipliers),
                "candidate_pairs": [list(candidate) for candidate in self.candidate_pairs],
                "lambda_definition": "lambda = lambda_information_multiplier * eta^2",
                "actual_lambda_by_multiplier": actual_lambdas(summaries),
                "lambda_warm_tuned": False,
                "lambda_warm": self.lambda_warm,
                "reference_candidate": self._candidate(self.reference_warmup_budget, self.reference_multiplier),
                "tuning_metric": self.tuning_metric,
                "guardrail_metric": self.guardrail_metric,
                "candidates_paired_by_seed": True,
                "final_evaluation_seeds_protected": True,
                "raw_eligible_winner": self._candidate_from_key(raw_best),
                "shortlist": shortlist_payload,
                "shortlist_touches_grid_boundary": boundary_shortlist,
                "requires_human_review_before_refinement": True,
                "statistics": statistics,
                "fisher_paired_differences_vs_reference": fisher_differences,
                "primary_paired_differences_vs_eligible_winner": primary_differences,
            },
        )

    def _observations(self, summaries: tuple[dict[str, object], ...]) -> dict[str, dict[str, dict[int, float]]]:
        grouped = {
            self._key(budget, multiplier): {
                self.tuning_metric: {},
                self.guardrail_metric: {},
                "coefficient_l2_error": {},
            }
            for budget, multiplier in self.candidate_pairs
        }
        for summary in summaries:
            job = summary["job"]
            key = self._key(int(job["warmup_budget"]), float(job["lambda_information_multiplier"]))
            seed = int(job["seed"])
            final_metrics = summary["final_metrics"]
            for metric in grouped[key]:
                grouped[key][metric][seed] = float(final_metrics[metric])
        return grouped

    def _statistics(self, observations: dict[str, dict[str, dict[int, float]]],
    ) -> dict[str, dict[str, dict[str, float | int]]]:  # fmt: skip
        return {
            key: {
                metric: sample_statistics(list(values.values()), self.seed_count) for metric, values in metrics.items()
            }
            for key, metrics in observations.items()
        }

    def _candidate_summary(self, key: str,
            statistics: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, object]:  # fmt: skip
        candidate = self._candidate_from_key(key)
        budget = int(candidate["warmup_budget"])
        multiplier = float(candidate["lambda_information_multiplier"])
        budget_boundary = len(self.warmup_budgets) > 1 and budget in (self.warmup_budgets[0], self.warmup_budgets[-1])
        multiplier_boundary = len(self.lambda_information_multipliers) > 1 and (
            multiplier in (self.lambda_information_multipliers[0], self.lambda_information_multipliers[-1])
        )
        return {**candidate, "on_grid_boundary": budget_boundary or multiplier_boundary, "statistics": statistics[key]}

    @staticmethod
    def _key(warmup_budget: int, multiplier: float) -> str:
        return f"warmup={warmup_budget},lambda={multiplier:g}"

    @staticmethod
    def _candidate(warmup_budget: int, multiplier: float) -> dict[str, float | int]:
        return {"warmup_budget": warmup_budget, "lambda_information_multiplier": multiplier}

    @classmethod
    def _candidate_from_key(cls, key: str) -> dict[str, float | int]:
        warmup, multiplier = key.split(",")
        return cls._candidate(int(warmup.split("=")[1]), float(multiplier.split("=")[1]))


class JointWarmupLambdaDiscoveryExecutor(DelegatingSeedExecutor):
    """Reuse warm-up preparation and vectorize the paired seeds per grid point."""

    batch_mismatch_message = "discovery batches must share warm-up and lambda"

    def batch_key(self, job: JointDiscoveryJob) -> tuple[int, float]:
        self._validate(job)
        return job.warmup_budget, job.lambda_information_multiplier

    def _baseline_study(self, key: tuple[int, float]) -> BaselineComparisonStudy:
        return self.study._baseline_study(*key)

    @staticmethod
    def _delegate_job(job: JointDiscoveryJob) -> StudyJob:
        return StudyJob("active", job.warmup_budget > 0, job.seed)

    def _adapt(self, result: StudyJobResult, job: JointDiscoveryJob) -> StudyJobResult:
        return replace(
            result,
            summary={
                **result.summary,
                "job": job.as_dict(),
                "lambda_information": (job.lambda_information_multiplier * self.application.problem.eta**2),
                "lambda_warm": self.study.lambda_warm,
            },
        )

    def _validate(self, job: JointDiscoveryJob) -> None:
        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this discovery study")
