"""Job identities and results shared by studies and run storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np

StudyMethod = Literal["active", "fixed_random", "random_each_round"]


class ExperimentJob(Protocol):
    """Serializable unit of work understood by the durable artifact store."""

    def as_dict(self) -> dict[str, object]: ...

    @property
    def label(self) -> str: ...


@dataclass(frozen=True, slots=True)
class StudyJob:
    """One independently restartable method/seed/warm-up execution."""

    method: StudyMethod
    warmup_enabled: bool
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def label(self) -> str:
        warmup = "warmup" if self.warmup_enabled else "no-warmup"
        return f"{warmup}/{self.method}/seed={self.seed}"


@dataclass(frozen=True, slots=True)
class StudyJobResult:
    """Numeric arrays and summary saved for one experiment job."""

    arrays: dict[str, np.ndarray]
    summary: dict[str, object]
