"""Fixed-total-budget sweeps over the grouped warm-up allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.studies import (
    BaselineComparisonStudy,
    DelegatingSeedExecutor,
    StudyJob,
    StudyMethod,
)


@dataclass(frozen=True, slots=True)
class WarmupSweepJob:
    """One method, warm-up budget, and seed in a fixed-total-budget sweep."""

    method: StudyMethod
    warmup_budget: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"warmup={self.warmup_budget}/{self.method}/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class WarmupSweepStudy:
    """Compare final performance while reallocating a fixed sample budget."""

    name: str = "warmup-sweep"
    seed_start: int = 100
    seed_count: int = 100
    total_samples: int = 1_000
    warmup_budgets: tuple[int, ...] = (24, 60, 240, 500, 800, 1_000)
    evaluation_interval: int = 25
    lambda_information_multiplier: float = 1.0
    lambda_warm: float = 0.01
    confidence_delta: float = 0.05
    methods: tuple[StudyMethod, ...] = (
        "active",
        "fixed_random",
        "random_each_round",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "warmup_budgets", tuple(self.warmup_budgets))
        object.__setattr__(self, "methods", tuple(self.methods))
        if not self.warmup_budgets:
            raise ConfigurationError("warmup_budgets must be nonempty")
        if len(set(self.warmup_budgets)) != len(self.warmup_budgets):
            raise ConfigurationError("warmup_budgets must be unique")
        if tuple(sorted(self.warmup_budgets)) != self.warmup_budgets:
            raise ConfigurationError("warmup_budgets must be strictly increasing")
        if any(budget < 0 or budget > self.total_samples for budget in self.warmup_budgets):
            raise ConfigurationError("every warm-up budget must lie in [0, total_samples]")
        # Reuse the baseline study's validation for every shared scientific setting.
        self._baseline_study(self.warmup_budgets[0])

    def jobs(self) -> tuple[WarmupSweepJob, ...]:
        seeds = range(self.seed_start, self.seed_start + self.seed_count)
        return tuple(
            WarmupSweepJob(method, budget, seed)
            for budget in self.warmup_budgets
            for method in self.methods
            for seed in seeds
        )

    def executor(
        self,
        application: ApplicationRuntime,
        numerics: ExperimentNumerics,
    ) -> WarmupSweepExecutor:
        return WarmupSweepExecutor(self, application, numerics)

    def _baseline_study(self, budget: int) -> BaselineComparisonStudy:
        return BaselineComparisonStudy(
            name=f"{self.name}-warmup-{budget}",
            seed_start=self.seed_start,
            seed_count=self.seed_count,
            total_samples=self.total_samples,
            warmup_budget=budget,
            evaluation_interval=self.evaluation_interval,
            lambda_information_multiplier=self.lambda_information_multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=self.methods,
            warmup_modes=(budget > 0,),
        )


class WarmupSweepExecutor(DelegatingSeedExecutor):
    """Delegate each budget to the already verified baseline executor."""

    batch_mismatch_message = "batched sweep jobs must share budget and method"

    def batch_key(self, job: WarmupSweepJob) -> tuple[int, StudyMethod]:
        self._validate_job(job)
        return job.warmup_budget, job.method

    def _delegate_key(self, job: WarmupSweepJob) -> int:
        return self.batch_key(job)[0]

    def _baseline_study(self, key: int) -> BaselineComparisonStudy:
        return self.study._baseline_study(key)

    @staticmethod
    def _delegate_job(job: WarmupSweepJob) -> StudyJob:
        return StudyJob(job.method, job.warmup_budget > 0, job.seed)

    def _validate_job(self, job: WarmupSweepJob) -> None:
        seed_stop = self.study.seed_start + self.study.seed_count
        if (
            job.warmup_budget not in self.study.warmup_budgets
            or job.method not in self.study.methods
            or not self.study.seed_start <= job.seed < seed_stop
        ):
            raise ConfigurationError(f"job {job.label!r} is not part of this sweep")
