"""Confirm a frozen warm-up/lambda candidate without allowing reselection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mairl.exceptions import ConfigurationError
from mairl_exp.execution.artifacts import atomic_write_json
from mairl_exp.studies.discovery import JointWarmupLambdaDiscoveryStudy
from mairl_exp.studies.selection import completed_summaries, not_worse, paired_statistics


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenJointCandidateConfirmationStudy(JointWarmupLambdaDiscoveryStudy):
    """Compare one frozen selection winner with its predeclared reference."""

    candidate_warmup_budget: int = -1
    candidate_multiplier: float = -1.0
    selection_run_id: str = ""
    selection_seed_start: int = -1
    selection_seed_count: int = 0
    confirmation_standard_errors: float = 2.0

    def __post_init__(self) -> None:
        super(FrozenJointCandidateConfirmationStudy, self).__post_init__()
        candidate = self.candidate_warmup_budget, self.candidate_multiplier
        reference = self.reference_warmup_budget, self.reference_multiplier
        if not self.selection_run_id:
            raise ConfigurationError("selection_run_id must identify the frozen selection")
        if self.selection_seed_start < 0 or self.selection_seed_count < 1:
            raise ConfigurationError("confirmation requires the protected selection seed range")
        selection_seeds = set(range(self.selection_seed_start, self.selection_seed_start + self.selection_seed_count))
        confirmation_seeds = set(range(self.seed_start, self.seed_start + self.seed_count))
        if selection_seeds & confirmation_seeds:
            raise ConfigurationError("selection and confirmation seeds overlap")
        if self.confirmation_standard_errors <= 0:
            raise ConfigurationError("confirmation_standard_errors must be positive")
        if candidate == reference or set(self.candidate_pairs) != {candidate, reference}:
            raise ConfigurationError("confirmation must contain exactly the frozen candidate and reference")

    def finalize(self, run_dir: Path) -> None:
        summaries = completed_summaries(run_dir)
        expected = len(self.jobs())
        if len(summaries) != expected:
            raise ConfigurationError(f"confirmation completed {len(summaries)} of {expected} jobs")
        observations = self._observations(summaries)
        statistics = self._statistics(observations)
        candidate_key = self._key(self.candidate_warmup_budget, self.candidate_multiplier)
        reference_key = self._key(self.reference_warmup_budget, self.reference_multiplier)
        paired_differences = {
            metric: paired_statistics(observations[candidate_key][metric], observations[reference_key][metric])
            for metric in (self.tuning_metric, self.guardrail_metric)
        }
        checks = {
            metric: not_worse(difference, self.confirmation_standard_errors)
            for metric, difference in paired_differences.items()
        }
        passed = all(checks.values())
        atomic_write_json(
            run_dir / "confirmation.json",
            {
                "status": "confirmed" if passed else "rejected",
                "selection_run_id": self.selection_run_id,
                "selection_and_confirmation_seeds_disjoint": True,
                "candidate_frozen_before_confirmation": True,
                "reselection_performed": False,
                "candidate": self._candidate(self.candidate_warmup_budget, self.candidate_multiplier),
                "reference": self._candidate(self.reference_warmup_budget, self.reference_multiplier),
                "tuning_metric": self.tuning_metric,
                "guardrail_metric": self.guardrail_metric,
                "confirmation_standard_errors": self.confirmation_standard_errors,
                "paired_differences_candidate_minus_reference": paired_differences,
                "checks_passed": checks,
                "statistics": statistics,
                "final_evaluation_seeds_protected": True,
            },
        )
        if not passed:
            raise ConfigurationError("frozen hyperparameter candidate failed confirmation")
