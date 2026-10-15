import pytest

from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl_exp.config import ExperimentNumerics, WarmupNumerics
from mairl_exp.studies.comparison import BaselineComparisonStudy


@pytest.fixture
def experiment_numerics() -> ExperimentNumerics:
    return ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupNumerics(design_tolerance=0.2),
        active_policy=PolicyOptimizationConfig(random_starts=1),
        evaluation_policy=PolicyOptimizationConfig(
            method="grid", grid_resolution=10, grid_max_points=1_000, refine_grid_solution=False
        ),
    )


@pytest.fixture
def baseline_study() -> BaselineComparisonStudy:
    return BaselineComparisonStudy(
        name="resume-test",
        seed_start=4,
        seed_count=1,
        total_samples=4,
        warmup_budget=0,
        evaluation_interval=2,
        methods=("fixed_random", "random_each_round"),
        warmup_modes=(False,),
    )
