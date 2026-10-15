"""Shared paths and numerical settings for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mairl.exploration.design import WarmupConfig
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "conf"
GUARD_ROOT = REPOSITORY_ROOT / "experiments" / "guard" / "GUARD"
RUNS_ROOT = REPOSITORY_ROOT / "artifacts" / "runs"
PLOTS_ROOT = REPOSITORY_ROOT / "artifacts" / "plots"
RESULTS_DIR = PLOTS_ROOT / "lobeke"
GENERATED_TIKZ_DIR = RESULTS_DIR / "tex" / "generated"
PAPER_OUTPUT_DIR = REPOSITORY_ROOT / "output" / "pdf"
REPRODUCIBILITY_ROOT = REPOSITORY_ROOT / "reproducibility"
DEFAULT_MANIFEST = REPRODUCIBILITY_ROOT / "figures.toml"
GENERATED_ROOT = REPRODUCIBILITY_ROOT / "tex" / "generated"
FIGURE_STYLE = REPRODUCIBILITY_ROOT / "tex" / "figure_style.tex"


@dataclass(frozen=True, slots=True)
class WarmupNumerics:
    """Numerical controls for WARM, excluding scientific budget and enablement."""

    design_tolerance: float = 0.15
    max_iterations: int = 10_000
    line_search_tolerance: float = 1e-12
    line_search_max_iterations: int = 500
    support_tolerance: float = 1e-12
    matrix_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        self.create(budget=0, enabled=False)

    def create(self, *, budget: int, enabled: bool) -> WarmupConfig:
        return WarmupConfig(
            budget=budget,
            design_tolerance=self.design_tolerance,
            enabled=enabled,
            max_iterations=self.max_iterations,
            line_search_tolerance=self.line_search_tolerance,
            line_search_max_iterations=self.line_search_max_iterations,
            support_tolerance=self.support_tolerance,
            matrix_tolerance=self.matrix_tolerance,
        )


@dataclass(frozen=True, slots=True)
class ExperimentNumerics:
    """Calibrated numerical settings, separate from scientific study parameters."""

    mle: MLEConfig
    warmup: WarmupNumerics
    active_policy: PolicyOptimizationConfig
    evaluation_policy: PolicyOptimizationConfig
