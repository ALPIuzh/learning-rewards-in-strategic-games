from functools import partial

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl_exp.config import CONFIG_DIRECTORY, ExperimentNumerics, WarmupNumerics
from mairl_exp.execution.runner import run_experiment
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


@pytest.fixture
def run_study(experiment_numerics, repository_root, tmp_path):
    """Run a study with isolated output and the common test numerics."""
    return partial(
        run_experiment,
        numerics=experiment_numerics,
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        command=("pytest",),
    )


@pytest.fixture
def composed_study():
    """Instantiate one or more study configs from the tracked conf/ tree.

    Uses the production CONFIG_DIRECTORY so tests cannot drift to a different
    config root than the CLI resolves.
    """

    def compose_studies(*config_names: str):
        with initialize_config_dir(config_dir=str(CONFIG_DIRECTORY.resolve()), version_base=None):
            studies = tuple(instantiate(compose(config_name=name).experiment) for name in config_names)
        return studies[0] if len(studies) == 1 else studies

    return compose_studies
