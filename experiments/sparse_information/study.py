"""Action-count sweep for the sparse-information mechanism study."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field
from typing import Literal

from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.jobs import StudyJob, StudyJobResult, StudyMethod
from mairl_exp.studies.comparison import BaselineComparisonExecutor, BaselineComparisonStudy

from .application import SparseInformationApplication


@dataclass(frozen=True, slots=True)
class SparseInformationJob:
    """One paired method/seed/action-count job."""

    leader_actions: int
    method: StudyMethod
    seed: int
    warmup_enabled: Literal[False] = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"m={self.leader_actions}/no-warmup/{self.method}/seed={self.seed}"


@dataclass(slots=True)
class SparseInformationSweepStudy:
    """Resumable, vectorized action-count sweep with paired seeds."""

    name: str = "sparse-information-action-sweep"
    leader_action_counts: tuple[int, ...] = (2, 4, 8, 16, 32)
    seed_start: int = 72_000
    seed_count: int = 100
    total_samples: int = 200
    evaluation_interval: int = 10
    lambda_information_multiplier: float = 4.0
    lambda_warm: float = 0.01
    confidence_delta: float = 0.05
    methods: tuple[StudyMethod, ...] = ("active", "fixed_random", "random_each_round")

    _valid_jobs: frozenset[SparseInformationJob] = field(init=False, repr=False, compare=False)
    _executors: dict[int, BaselineComparisonExecutor] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.leader_action_counts = tuple(self.leader_action_counts)
        self.methods = tuple(self.methods)
        if self.leader_action_counts != (2, 4, 8, 16, 32):
            raise ConfigurationError("the frozen action sweep must be exactly (2, 4, 8, 16, 32)")
        # Reuse the generic study's comprehensive scientific-parameter validation.
        self._comparison()

    def _comparison(self) -> BaselineComparisonStudy:
        return BaselineComparisonStudy(
            name=self.name,
            seed_start=self.seed_start,
            seed_count=self.seed_count,
            total_samples=self.total_samples,
            warmup_budget=0,
            evaluation_interval=self.evaluation_interval,
            lambda_information_multiplier=self.lambda_information_multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=self.methods,
            warmup_modes=(False,),
        )

    def jobs(self) -> tuple[SparseInformationJob, ...]:
        return tuple(
            SparseInformationJob(m, method, seed)
            for m in self.leader_action_counts
            for method in self.methods
            for seed in range(self.seed_start, self.seed_start + self.seed_count)
        )

    def executor(self, application: ApplicationRuntime, numerics: ExperimentNumerics) -> SparseInformationSweepStudy:
        if application.name != "sparse-information":
            raise ConfigurationError("sparse-information sweep requires its matching application")
        prepared = copy(self)
        prepared._valid_jobs = frozenset(self.jobs())
        comparison = self._comparison()
        prepared._executors = {
            m: BaselineComparisonExecutor(
                comparison,
                SparseInformationApplication(
                    leader_actions=m,
                    eta=float(application.metadata["eta"]),
                    radius=float(application.metadata["radius"]),
                    true_omega=float(application.metadata["true_omega"]),
                    safe_payoff=float(application.metadata["safe_payoff"]),
                ).build(numerics),
                numerics,
            )
            for m in self.leader_action_counts
        }
        return prepared

    def batch_key(self, job: SparseInformationJob) -> tuple[int, StudyMethod, bool]:
        if job not in self._valid_jobs:
            raise ConfigurationError(f"job {job.label!r} is not part of this study")
        return job.leader_actions, job.method, job.warmup_enabled

    def run_many(self, jobs: tuple[SparseInformationJob, ...], *, on_progress=None):
        if not jobs:
            return ()
        key = self.batch_key(jobs[0])
        if any(self.batch_key(job) != key for job in jobs):
            raise ConfigurationError("batched sparse-information jobs must share m and method")
        executor = self._executors[key[0]]
        base_jobs = tuple(StudyJob(job.method, False, job.seed) for job in jobs)
        callbacks = {base.seed: sparse for base, sparse in zip(base_jobs, jobs, strict=True)}
        return executor.run_many(
            base_jobs,
            on_progress=(
                None if on_progress is None else lambda job, done, total: on_progress(callbacks[job.seed], done, total)
            ),
        )

    def run(self, job: SparseInformationJob, *, on_progress=None) -> StudyJobResult:
        return self.run_many(
            (job,), on_progress=(None if on_progress is None else lambda _job, done, total: on_progress(done, total))
        )[0]
