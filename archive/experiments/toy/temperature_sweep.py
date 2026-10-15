"""Fixed-budget sensitivity study for the follower inverse temperature."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.toy.application import ToyApplication
from mairl.exceptions import ConfigurationError
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import ExperimentNumerics
from mairl_exp.studies import (
    BaselineComparisonExecutor,
    BaselineComparisonStudy,
    DelegatingSeedExecutor,
    StudyJob,
    StudyMethod,
)


@dataclass(frozen=True, slots=True)
class ToyTemperatureJob:
    """One inverse temperature, exploration method, and protected seed."""

    eta: float
    method: StudyMethod
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"eta={self.eta:g}/{self.method}/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class ToyTemperatureSweepStudy:
    """Change response sharpness while fixing the game and sampling protocol."""

    name: str = "structured-toy-temperature-final"
    seed_start: int = 83_000
    seed_count: int = 100
    eta_values: tuple[float, ...] = (0.5, 0.75, 1.25, 2.0, 3.0)
    total_samples: int = 1_000
    warmup_budget: int = 120
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
        object.__setattr__(self, "eta_values", tuple(float(value) for value in self.eta_values))
        object.__setattr__(self, "methods", tuple(self.methods))
        if not self.eta_values or any(value <= 0 for value in self.eta_values):
            raise ConfigurationError("eta_values must be nonempty and strictly positive")
        if len(set(self.eta_values)) != len(self.eta_values):
            raise ConfigurationError("eta_values must be unique")
        if tuple(sorted(self.eta_values)) != self.eta_values:
            raise ConfigurationError("eta_values must be strictly increasing")
        if 1.25 not in self.eta_values:
            raise ConfigurationError("eta_values must retain the canonical eta=1.25 instance")
        self._baseline_study()

    def jobs(self) -> tuple[ToyTemperatureJob, ...]:
        seeds = range(self.seed_start, self.seed_start + self.seed_count)
        return tuple(
            ToyTemperatureJob(eta, method, seed)
            for eta in self.eta_values
            for method in self.methods
            for seed in seeds
        )

    def executor(
        self,
        application: ApplicationRuntime,
        numerics: ExperimentNumerics,
    ) -> ToyTemperatureSweepExecutor:
        return ToyTemperatureSweepExecutor(self, application, numerics)

    def _baseline_study(self) -> BaselineComparisonStudy:
        return BaselineComparisonStudy(
            name=self.name,
            seed_start=self.seed_start,
            seed_count=self.seed_count,
            total_samples=self.total_samples,
            warmup_budget=self.warmup_budget,
            evaluation_interval=self.evaluation_interval,
            lambda_information_multiplier=self.lambda_information_multiplier,
            lambda_warm=self.lambda_warm,
            confidence_delta=self.confidence_delta,
            methods=self.methods,
            warmup_modes=(True,),
        )


class ToyTemperatureSweepExecutor(DelegatingSeedExecutor):
    """Reuse the verified vectorized comparison executor at each temperature."""

    batch_mismatch_message = "batched temperature jobs must share eta and method"

    def __init__(
        self,
        study: ToyTemperatureSweepStudy,
        application: ApplicationRuntime,
        numerics: ExperimentNumerics,
    ) -> None:
        if application.name != "toy":
            raise ConfigurationError("temperature sweep requires the structured toy application")
        super().__init__(study, application, numerics)

    def batch_key(self, job: ToyTemperatureJob) -> tuple[float, StudyMethod]:
        self._validate_job(job)
        return job.eta, job.method

    def _delegate_key(self, job: ToyTemperatureJob) -> float:
        return self.batch_key(job)[0]

    def _delegate(self, key: float) -> BaselineComparisonExecutor:
        """Each temperature needs its own application variant, not just its own study."""

        executor = self._delegates.get(key)
        if executor is None:
            metadata = self.application.metadata
            variant = ToyApplication(
                problem_seed=int(metadata["problem_seed"]),
                eta=key,
                radius=float(metadata["radius"]),
                true_omega=tuple(float(value) for value in metadata["true_omega"]),
                leader_reward=tuple(
                    tuple(float(value) for value in row) for row in metadata["leader_reward"]
                ),
            ).build(self.numerics)
            executor = BaselineComparisonExecutor(
                self.study._baseline_study(),
                variant,
                self.numerics,
            )
            self._delegates[key] = executor
        return executor

    @staticmethod
    def _delegate_job(job: ToyTemperatureJob) -> StudyJob:
        return StudyJob(job.method, True, job.seed)

    def _validate_job(self, job: ToyTemperatureJob) -> None:
        seed_stop = self.study.seed_start + self.study.seed_count
        if (
            job.eta not in self.study.eta_values
            or job.method not in self.study.methods
            or not self.study.seed_start <= job.seed < seed_stop
        ):
            raise ConfigurationError(f"job {job.label!r} is not part of this sweep")
