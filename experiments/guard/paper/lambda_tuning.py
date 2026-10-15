"""Protected lambda tuning for the Lobeke Active IRL variants.

The scientific choice made here is deliberately narrow: tune only the online
information regularizer ``lambda = multiplier * eta**2``.  ``lambda_warm`` remains
fixed and is used only by the warm-up phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

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

_MINIMUM_SEEDS = "lambda tuning requires at least two seeds per split"
_SEED_MISMATCH = "paired lambda candidates do not contain identical seeds"

TuningSplit = Literal["selection", "confirmation"]


@dataclass(frozen=True, slots=True)
class LambdaTuningJob:
    """One lambda candidate, warm-up mode, seed, and protected split."""

    split: TuningSplit
    warmup_enabled: bool
    lambda_information_multiplier: float
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        warmup = "warmup" if self.warmup_enabled else "no-warmup"
        return f"{self.split}/{warmup}/lambda={self.lambda_information_multiplier:g}eta2/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class ActiveLambdaTuningStudy:
    """Tune lambda on selection seeds and verify the frozen choice on confirmation seeds.

    Selection minimizes final source-task Stackelberg sub-optimality.  A candidate is
    eligible only when its Fisher-weighted error is statistically no worse than the
    theorem-motivated reference multiplier.  The one-standard-error rule then chooses
    the smallest eligible multiplier, guarding against an extreme value that wins by
    noise.  Confirmation never reselects a candidate.
    """

    name: str
    selection_seed_start: int
    selection_seed_count: int
    confirmation_seed_start: int
    confirmation_seed_count: int
    final_evaluation_seed_start: int
    final_evaluation_seed_count: int
    total_samples: int
    warmup_budget: int
    lambda_information_multipliers: tuple[float, ...]
    lambda_warm: float
    confidence_delta: float
    warmup_modes: tuple[bool, ...] = (True, False)
    evaluation_interval: int | None = None
    tuning_metric: str = "stackelberg_suboptimality"
    guardrail_metric: str = "fisher_transfer_error"
    reference_multiplier: float = 1.0
    guardrail_standard_errors: float = 2.0
    confirmation_standard_errors: float = 2.0
    minimum_seeds_per_split: int = 30
    reject_upper_boundary: bool = True

    def __post_init__(self) -> None:
        multipliers = tuple(float(value) for value in self.lambda_information_multipliers)
        modes = tuple(self.warmup_modes)
        object.__setattr__(self, "lambda_information_multipliers", multipliers)
        object.__setattr__(self, "warmup_modes", modes)
        if not self.name or not self.tuning_metric or not self.guardrail_metric:
            raise ConfigurationError("study name and metric names must be nonempty")
        if min(self.selection_seed_count, self.confirmation_seed_count) < (self.minimum_seeds_per_split):
            raise ConfigurationError("each lambda tuning split has fewer seeds than the configured minimum")
        if self._seeds("selection") & self._seeds("confirmation"):
            raise ConfigurationError("lambda selection and confirmation seeds must be disjoint")
        final_seeds = set(
            range(self.final_evaluation_seed_start, self.final_evaluation_seed_start + self.final_evaluation_seed_count)
        )
        if self.final_evaluation_seed_count < 1:
            raise ConfigurationError("final_evaluation_seed_count must be positive")
        if final_seeds & (self._seeds("selection") | self._seeds("confirmation")):
            raise ConfigurationError("lambda tuning seeds overlap the protected final evaluation")
        if self.total_samples < 1 or not 0 <= self.warmup_budget <= self.total_samples:
            raise ConfigurationError("sample budgets must satisfy 0 <= warmup <= total")
        interval = self.total_samples if self.evaluation_interval is None else self.evaluation_interval
        if interval < 1:
            raise ConfigurationError("evaluation_interval must be positive")
        object.__setattr__(self, "evaluation_interval", interval)
        if (
            not multipliers
            or len(set(multipliers)) != len(multipliers)
            or tuple(sorted(multipliers)) != multipliers
            or any(value <= 0 for value in multipliers)
        ):
            raise ConfigurationError("lambda_information_multipliers must be unique, increasing, and strictly positive")
        if self.reference_multiplier not in multipliers:
            raise ConfigurationError("reference_multiplier must be one of the candidates")
        if not modes or len(set(modes)) != len(modes):
            raise ConfigurationError("warmup_modes must be nonempty and unique")
        if not self.lambda_warm > 0:
            raise ConfigurationError("lambda_warm must be strictly positive")
        if not 0 < self.confidence_delta < 1:
            raise ConfigurationError("confidence_delta must lie in (0, 1)")
        if min(self.guardrail_standard_errors, self.confirmation_standard_errors) <= 0:
            raise ConfigurationError("standard-error multipliers must be positive")
        self._baseline_study("selection", modes[0], multipliers[0])

    def _seeds(self, split: TuningSplit) -> set[int]:
        start, count = (
            (self.selection_seed_start, self.selection_seed_count)
            if split == "selection"
            else (self.confirmation_seed_start, self.confirmation_seed_count)
        )
        return set(range(start, start + count))

    def jobs(self) -> tuple[LambdaTuningJob, ...]:
        return tuple(
            LambdaTuningJob(split, mode, multiplier, seed)
            for split in ("selection", "confirmation")
            for mode in self.warmup_modes
            for multiplier in self.lambda_information_multipliers
            for seed in sorted(self._seeds(split))
        )

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> ActiveLambdaTuningExecutor:
        return ActiveLambdaTuningExecutor(self, application, numerics)

    def _baseline_study(self, split: TuningSplit, warmup_enabled: bool, multiplier: float) -> BaselineComparisonStudy:
        seeds = self._seeds(split)
        return BaselineComparisonStudy(
            name=f"{self.name}-{split}-{warmup_enabled}-{multiplier:g}",
            seed_start=min(seeds),
            seed_count=len(seeds),
            total_samples=self.total_samples,
            warmup_budget=self.warmup_budget,
            evaluation_interval=int(self.evaluation_interval),
            lambda_information_multiplier=multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=("active",),
            warmup_modes=(warmup_enabled,),
        )

    def finalize(self, run_dir: Path) -> None:
        summaries = completed_summaries(run_dir)
        expected = len(self.jobs())
        if len(summaries) != expected:
            raise ConfigurationError(f"lambda tuning completed {len(summaries)} of {expected} jobs; refusing selection")

        observations = self._observations(summaries)
        statistics = self._statistics(observations)
        lambda_by_multiplier = actual_lambdas(summaries)
        decisions = {
            _mode_name(mode): self._select_mode(mode, statistics[_mode_name(mode)], observations[_mode_name(mode)])
            for mode in self.warmup_modes
        }
        passed = all(decision["status"] == "selected" for decision in decisions.values())
        payload = {
            "status": "selected" if passed else "selection_failed",
            "tuned_parameter": "lambda_information_multiplier",
            "lambda_definition": "lambda = lambda_information_multiplier * eta^2",
            "actual_lambda_by_multiplier": lambda_by_multiplier,
            "lambda_warm_tuned": False,
            "lambda_warm": self.lambda_warm,
            "tuning_metric": self.tuning_metric,
            "guardrail_metric": self.guardrail_metric,
            "selection_rule": (
                "paired Fisher-guardrail eligibility, then smallest multiplier whose paired "
                "source-SubOpt difference is within one standard error of the eligible winner"
            ),
            "confirmation_rule": (
                "the frozen candidate must remain within the configured paired-standard-error "
                "margins of the selection winner, reference source SubOpt, and reference Fisher "
                "error"
            ),
            "selection_and_confirmation_seeds_disjoint": True,
            "candidates_paired_by_seed_within_each_split": True,
            "final_evaluation_seeds_protected": True,
            "decisions": decisions,
            "statistics": statistics,
        }
        atomic_write_json(run_dir / "lambda_selection.json", payload)
        if not passed:
            raise ConfigurationError("lambda selection failed confirmation or selected the upper grid boundary")

    def _observations(self, summaries: tuple[dict[str, object], ...],
    ) -> dict[str, dict[str, dict[float, dict[str, dict[int, float]]]]]:  # fmt: skip
        grouped: dict[str, dict[str, dict[float, dict[str, dict[int, float]]]]] = {
            _mode_name(mode): {
                split: {
                    multiplier: {self.tuning_metric: {}, self.guardrail_metric: {}}
                    for multiplier in self.lambda_information_multipliers
                }
                for split in ("selection", "confirmation")
            }
            for mode in self.warmup_modes
        }
        for summary in summaries:
            job = summary["job"]
            mode = _mode_name(bool(job["warmup_enabled"]))
            split = str(job["split"])
            multiplier = float(job["lambda_information_multiplier"])
            seed = int(job["seed"])
            final_metrics = summary["final_metrics"]
            for metric in (self.tuning_metric, self.guardrail_metric):
                grouped[mode][split][multiplier][metric][seed] = float(final_metrics[metric])
        return grouped

    def _statistics(self, grouped: dict[str, dict[str, dict[float, dict[str, dict[int, float]]]]],
    ) -> dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]]:  # fmt: skip

        expected_seeds = {"selection": self.selection_seed_count, "confirmation": self.confirmation_seed_count}
        return {
            mode: {
                split: {
                    f"{multiplier:g}": {
                        metric: sample_statistics(
                            list(values.values()), expected_seeds[split], minimum_seeds_message=_MINIMUM_SEEDS
                        )
                        for metric, values in metrics.items()
                    }
                    for multiplier, metrics in candidates.items()
                }
                for split, candidates in splits.items()
            }
            for mode, splits in grouped.items()
        }

    def _select_mode(self, warmup_enabled: bool, statistics: dict[str, dict[str, dict[str, dict[str, float | int]]]],
            observations: dict[str, dict[float, dict[str, dict[int, float]]]]) -> dict[str, object]:  # fmt: skip
        selection = statistics["selection"]
        selection_observations = observations["selection"]
        reference_fisher = selection_observations[self.reference_multiplier][self.guardrail_metric]
        fisher_differences = {
            multiplier: paired_statistics(
                selection_observations[multiplier][self.guardrail_metric],
                reference_fisher,
                mismatch_message=_SEED_MISMATCH,
                minimum_seeds_message=_MINIMUM_SEEDS,
            )
            for multiplier in self.lambda_information_multipliers
        }
        eligible = tuple(
            multiplier
            for multiplier in self.lambda_information_multipliers
            if not_worse(fisher_differences[multiplier], self.guardrail_standard_errors)
        )
        raw_best = min(eligible, key=lambda value: float(selection[f"{value:g}"][self.tuning_metric]["mean"]))
        raw_best_primary = selection_observations[raw_best][self.tuning_metric]
        primary_differences = {
            multiplier: paired_statistics(
                selection_observations[multiplier][self.tuning_metric],
                raw_best_primary,
                mismatch_message=_SEED_MISMATCH,
                minimum_seeds_message=_MINIMUM_SEEDS,
            )
            for multiplier in eligible
        }
        one_se_eligible = tuple(
            multiplier for multiplier in eligible if not_worse(primary_differences[multiplier], 1.0)
        )
        selected = min(one_se_eligible)

        confirmation_observations = observations["confirmation"]
        primary_confirmation_difference = paired_statistics(
            confirmation_observations[selected][self.tuning_metric],
            confirmation_observations[raw_best][self.tuning_metric],
            mismatch_message=_SEED_MISMATCH,
            minimum_seeds_message=_MINIMUM_SEEDS,
        )
        primary_reference_confirmation_difference = paired_statistics(
            confirmation_observations[selected][self.tuning_metric],
            confirmation_observations[self.reference_multiplier][self.tuning_metric],
            mismatch_message=_SEED_MISMATCH,
            minimum_seeds_message=_MINIMUM_SEEDS,
        )
        fisher_confirmation_difference = paired_statistics(
            confirmation_observations[selected][self.guardrail_metric],
            confirmation_observations[self.reference_multiplier][self.guardrail_metric],
            mismatch_message=_SEED_MISMATCH,
            minimum_seeds_message=_MINIMUM_SEEDS,
        )
        primary_confirmed = not_worse(primary_confirmation_difference, self.confirmation_standard_errors)
        primary_reference_confirmed = not_worse(
            primary_reference_confirmation_difference, self.confirmation_standard_errors
        )
        guardrail_confirmed = not_worse(fisher_confirmation_difference, self.confirmation_standard_errors)
        upper_boundary = selected == self.lambda_information_multipliers[-1]
        passed = (
            primary_confirmed
            and primary_reference_confirmed
            and guardrail_confirmed
            and not (self.reject_upper_boundary and upper_boundary)
        )
        return {
            "status": "selected" if passed else "rejected",
            "warmup_enabled": warmup_enabled,
            "selected_multiplier": selected if passed else None,
            "provisional_multiplier": selected,
            "raw_selection_winner": raw_best,
            "fisher_guardrail_eligible": list(eligible),
            "one_standard_error_eligible": list(one_se_eligible),
            "selection_fisher_paired_differences_vs_reference": {
                f"{multiplier:g}": values for multiplier, values in fisher_differences.items()
            },
            "selection_primary_paired_differences_vs_winner": {
                f"{multiplier:g}": values for multiplier, values in primary_differences.items()
            },
            "confirmation_primary_paired_difference": primary_confirmation_difference,
            "confirmation_primary_paired_difference_vs_reference": (primary_reference_confirmation_difference),
            "confirmation_fisher_paired_difference": fisher_confirmation_difference,
            "primary_confirmation_passed": primary_confirmed,
            "primary_no_harm_vs_reference_confirmed": primary_reference_confirmed,
            "fisher_guardrail_confirmation_passed": guardrail_confirmed,
            "selected_is_upper_grid_boundary": upper_boundary,
        }


class ActiveLambdaTuningExecutor(DelegatingSeedExecutor):
    """Vectorize seeds while caching the deterministic warm-up across candidates."""

    batch_mismatch_message = "lambda tuning batches must share split, mode, and lambda"

    def batch_key(self, job: LambdaTuningJob) -> tuple[TuningSplit, bool, float]:
        self._validate(job)
        return job.split, job.warmup_enabled, job.lambda_information_multiplier

    def _baseline_study(self, key: tuple[TuningSplit, bool, float]) -> BaselineComparisonStudy:
        return self.study._baseline_study(*key)

    @staticmethod
    def _delegate_job(job: LambdaTuningJob) -> StudyJob:
        return StudyJob("active", job.warmup_enabled, job.seed)

    def _adapt(self, result: StudyJobResult, job: LambdaTuningJob) -> StudyJobResult:
        return replace(
            result,
            summary={
                **result.summary,
                "job": job.as_dict(),
                "lambda_information": (job.lambda_information_multiplier * self.application.problem.eta**2),
                "lambda_warm": self.study.lambda_warm,
            },
        )

    def _validate(self, job: LambdaTuningJob) -> None:
        if job not in self.study.jobs():
            raise ConfigurationError(f"job {job.label!r} is not part of this tuning study")


def _mode_name(warmup_enabled: bool) -> str:
    return "warmup" if warmup_enabled else "no_warmup"
