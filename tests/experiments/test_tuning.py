from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mairl.algorithm import ActiveIRL, ActiveIRLConfig
from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.exceptions import ConfigurationError
from mairl.exploration.design import WarmupConfig
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig
from mairl_exp.studies.trials import TuningTrial, run_tuning


def test_tuning_reuses_preparation_without_changing_trials(small_problem, small_true_omega) -> None:
    true_omega = small_true_omega
    base = ActiveIRLConfig(
        horizon=3,
        lambda_information=small_problem.eta**2,
        lambda_warm=0.01,
        confidence_delta=0.05,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=30, design_tolerance=0.2),
        policy=PolicyOptimizationConfig(random_starts=1),
        seed=17,
    )
    trials = (
        TuningTrial("ridge-small", base),
        TuningTrial("ridge-large", replace(base, mle=replace(base.mle, ridge=1e-3))),
    )

    tuning = run_tuning(
        small_problem,
        trials,
        oracle_factory=lambda _: SoftmaxFollowerOracle(small_problem, true_omega),
        evaluate=lambda _, result: {"coefficient_error": np.linalg.norm(result.omega - true_omega)},
    )

    assert tuning.unique_warmup_designs == 1
    assert len(tuning.records) == 2
    assert all(record.algorithm_seconds > 0 for record in tuning.records)
    expected_best = min(tuning.records, key=lambda record: record.metrics["coefficient_error"])
    assert tuning.best("coefficient_error", direction="minimize").trial.name == expected_best.trial.name
    summaries = tuning.as_dicts()
    assert summaries[0]["name"] == "ridge-small"
    assert summaries[0]["config"]["mle"]["ridge"] == 1e-4
    assert summaries[0]["metrics"] == tuning.records[0].metrics

    direct = ActiveIRL(small_problem, base).run(SoftmaxFollowerOracle(small_problem, true_omega))
    reused = tuning.records[0].result
    assert np.array_equal(reused.dataset.policies, direct.dataset.policies)
    assert np.array_equal(reused.dataset.follower_actions, direct.dataset.follower_actions)
    assert np.array_equal(reused.omega, direct.omega)


def test_tuning_rejects_ambiguous_or_invalid_results(small_problem, small_oracle) -> None:
    config = ActiveIRLConfig(
        horizon=0,
        lambda_information=small_problem.eta**2,
        lambda_warm=0.01,
        confidence_delta=0.05,
        mle=MLEConfig(ridge=1e-4),
        warmup=WarmupConfig(budget=20, design_tolerance=0.2),
        seed=2,
    )
    duplicate = (TuningTrial("same", config), TuningTrial("same", config))

    def oracle_factory(_):
        return small_oracle

    with pytest.raises(ConfigurationError, match="names must be unique"):
        run_tuning(small_problem, duplicate, oracle_factory=oracle_factory, evaluate=lambda *_: {})

    with pytest.raises(ConfigurationError, match="non-finite metric"):
        run_tuning(
            small_problem,
            (TuningTrial("invalid-metric", config),),
            oracle_factory=oracle_factory,
            evaluate=lambda *_: {"loss": np.nan},
        )
