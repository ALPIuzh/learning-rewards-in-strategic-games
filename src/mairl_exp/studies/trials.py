"""Exact serial orchestration for reproducible hyperparameter trials."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from mairl.algorithm import ActiveIRL, ActiveIRLConfig, ActiveIRLResult
from mairl.core.oracle import FollowerOracle
from mairl.core.problem import CompiledProblem
from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig
from mairl.exploration.warmup import PreparedWarmup, prepare_warmup


@dataclass(frozen=True, slots=True)
class TuningTrial:
    """Named experiment configuration, including its explicit random seed."""

    name: str
    config: ActiveIRLConfig

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("tuning trial name must be nonempty")


@dataclass(frozen=True, slots=True)
class TuningRecord:
    """Result, application metrics, and timing for one completed trial."""

    trial: TuningTrial
    result: ActiveIRLResult
    metrics: Mapping[str, float]
    algorithm_seconds: float


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Completed records plus preparation reuse diagnostics."""

    records: tuple[TuningRecord, ...]
    unique_warmup_designs: int

    def as_dicts(self) -> tuple[dict[str, object], ...]:
        """Return JSON-serializable records with configurations and metrics."""

        return tuple(
            {
                "name": record.trial.name,
                "config": asdict(record.trial.config),
                "metrics": dict(record.metrics),
                "algorithm_seconds": record.algorithm_seconds,
            }
            for record in self.records
        )

    def best(self, metric: str, *, direction: Literal["minimize", "maximize"]) -> TuningRecord:
        """Return the best record with the optimization direction stated explicitly."""

        if direction not in ("minimize", "maximize"):
            raise ConfigurationError("direction must be either 'minimize' or 'maximize'")
        missing = [record.trial.name for record in self.records if metric not in record.metrics]
        if missing:
            raise ConfigurationError(f"metric {metric!r} is missing from trials: {', '.join(missing)}")
        selector = min if direction == "minimize" else max
        return selector(self.records, key=lambda record: record.metrics[metric])


OracleFactory = Callable[[TuningTrial], FollowerOracle]
MetricFunction = Callable[[TuningTrial, ActiveIRLResult], Mapping[str, float]]


def _validated_metrics(trial: TuningTrial, metrics: Mapping[str, float]) -> dict[str, float]:
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError(f"trial {trial.name!r} produced an empty or non-string metric name")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ConfigurationError(f"trial {trial.name!r} produced non-finite metric {name!r}: {numeric}")
        validated[name] = numeric
    return validated


def run_tuning(problem: CompiledProblem, trials: Iterable[TuningTrial],
        *, oracle_factory: OracleFactory, evaluate: MetricFunction) -> TuningResult:  # fmt: skip
    """Run independent trials exactly and reuse only deterministic preparation.

    This low-level helper accepts heterogeneous trial configurations and executes them
    serially. The durable experiment executor separately vectorizes compatible seeded
    replicates while preserving their distinct adaptive trajectories. The compiled
    problem and every identical :class:`WarmupConfig` design are reused here. Each
    ``oracle_factory`` call must return a fresh application oracle for the supplied
    trial.

    Numerical or application failures propagate immediately; incomplete trials are
    never converted into tuning scores.
    """

    scheduled = tuple(trials)
    if not scheduled:
        raise ConfigurationError("at least one tuning trial is required")
    names = [trial.name for trial in scheduled]
    if len(set(names)) != len(names):
        raise ConfigurationError("tuning trial names must be unique")

    prepared_by_config: dict[WarmupConfig, PreparedWarmup] = {}
    records: list[TuningRecord] = []
    for trial in scheduled:
        prepared = prepared_by_config.get(trial.config.warmup)
        if prepared is None:
            prepared = prepare_warmup(problem, trial.config.warmup)
            prepared_by_config[trial.config.warmup] = prepared

        runner = ActiveIRL(problem, trial.config)
        oracle = oracle_factory(trial)
        if not isinstance(oracle, FollowerOracle):
            raise ConfigurationError(f"oracle_factory returned an invalid oracle for trial {trial.name!r}")
        start = perf_counter()
        result = runner.run(oracle, prepared_warmup=prepared)
        elapsed = perf_counter() - start
        metrics = _validated_metrics(trial, evaluate(trial, result))
        records.append(TuningRecord(trial=trial, result=result, metrics=metrics, algorithm_seconds=elapsed))

    return TuningResult(records=tuple(records), unique_warmup_designs=len(prepared_by_config))
