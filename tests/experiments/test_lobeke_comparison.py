from __future__ import annotations

from dataclasses import replace

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from experiments.guard.paper.application import _self_concordance_distances, _validated_stackelberg_suboptimality
from experiments.guard.paper.study import LobekeComparisonJob, LobekeComparisonStudy
from experiments.zero_sum_comparison.application import ZeroSumToyApplication
from mairl import FixedLeaderOLSConfig
from mairl.core.math import follower_probabilities
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl_exp.config import CONFIG_DIRECTORY


def test_lobeke_target_suboptimality_rejects_a_false_optimum() -> None:
    np.testing.assert_array_equal(
        _validated_stackelberg_suboptimality(1.0, np.asarray([1.0, 1.0 + 5e-10])), np.zeros(2)
    )
    with np.testing.assert_raises_regex(RuntimeError, "exceeded the computed source optimum"):
        _validated_stackelberg_suboptimality(1.0, np.asarray([1.0 + 2e-9]))


def test_self_concordance_distance_maximizes_over_leader_vertices() -> None:
    features = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]])
    errors = np.asarray([[2.0, -1.0], [0.0, 0.0]])
    np.testing.assert_allclose(_self_concordance_distances(features, errors), np.asarray([4.0, 0.0]))


def test_lobeke_final_protocol_uses_frozen_mode_specific_settings(composed_study) -> None:
    study = composed_study("runs/lobeke/comparison")

    assert study.seed_start == 2026097000
    assert study.seed_count == 50
    assert study.warmup_budget == 50
    assert study.warmup_lambda_information_multiplier == 0.03
    assert study.no_warmup_lambda_information_multiplier == 0.1
    assert study.fixed_expert_mle_ridge == 1e-6
    assert "fixed_expert_mle" in study.methods
    assert len(study.jobs()) == 9 * 50

    with_warmup = study._stackelberg_study(True)
    without_warmup = study._stackelberg_study(False)
    assert with_warmup.warmup_modes == (True,)
    assert with_warmup.lambda_information_multiplier == 0.03
    assert without_warmup.warmup_modes == (False,)
    assert without_warmup.lambda_information_multiplier == 0.1


def test_lobeke_mle_ridge_override_is_shared_with_fixed_expert() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIRECTORY.resolve()), version_base=None):
        config = compose(config_name="runs/lobeke/comparison", overrides=["numerics.mle.ridge=2e-6"])
        study = instantiate(config.experiment)
        numerics = instantiate(config.numerics)
    assert study.fixed_expert_mle_ridge == numerics.mle.ridge == 2e-6


def test_lobeke_fixed_expert_mle_is_batched_and_uses_its_declared_policy(experiment_numerics) -> None:
    runtime = ZeroSumToyApplication().build(experiment_numerics)
    fixed_policy = np.zeros(runtime.problem.n_leader_actions)
    fixed_policy[0] = 1.0
    runtime = replace(runtime, fixed_mle_leader_policy=fixed_policy, fixed_mle_expert_definition="test_fixed_policy")
    study = LobekeComparisonStudy(
        ols=FixedLeaderOLSConfig(pseudocount=0.5, smoothing_warning_ratio=0.1),
        name="fixed-expert-mle-test",
        seed_start=30,
        seed_count=2,
        total_samples=6,
        warmup_budget=0,
        evaluation_interval=2,
        warmup_lambda_information_multiplier=1.0,
        no_warmup_lambda_information_multiplier=1.0,
        methods=("active", "fixed_expert_mle", "zero_sum_ols", "follower_ols"),
        warmup_modes=(False,),
    )
    executor = study.executor(runtime, experiment_numerics)
    mle_jobs = (LobekeComparisonJob("fixed_expert_mle", False, 30), LobekeComparisonJob("fixed_expert_mle", False, 31))
    mle_results = executor.run_many(mle_jobs)

    assert len(mle_results) == 2
    expected_probability = follower_probabilities(runtime.problem, fixed_policy, runtime.true_omega)
    expected_actions = experiment_rng(30, RandomPurpose.EXPERT_FOLLOWER).choice(
        runtime.problem.n_follower_actions, size=6, p=expected_probability
    )
    np.testing.assert_array_equal(mle_results[0].arrays["follower_actions"], expected_actions)
    np.testing.assert_array_equal(
        mle_results[0].arrays["policies"], np.broadcast_to(fixed_policy, (6, runtime.problem.n_leader_actions))
    )
    np.testing.assert_array_equal(mle_results[0].arrays["evaluation_sample_counts"], np.asarray([2, 4, 6]))
    assert mle_results[0].summary["estimator_mode"] == "fixed_expert_mle"
    assert mle_results[0].summary["expert_definition"] == "test_fixed_policy"
    assert mle_results[0].summary["ridge"] == study.fixed_expert_mle_ridge
    assert mle_results[0].summary["samples"] == 6
    assert np.all(np.isfinite(mle_results[0].arrays["omega_history"]))


def test_lobeke_fixed_expert_mle_can_run_alone_with_its_own_ridge(experiment_numerics) -> None:
    runtime = ZeroSumToyApplication().build(experiment_numerics)
    study = LobekeComparisonStudy(
        ols=FixedLeaderOLSConfig(pseudocount=0.5, smoothing_warning_ratio=0.1),
        name="fixed-expert-only-test",
        seed_start=30,
        seed_count=2,
        total_samples=6,
        warmup_budget=0,
        evaluation_interval=6,
        warmup_lambda_information_multiplier=1.0,
        no_warmup_lambda_information_multiplier=1.0,
        fixed_expert_mle_ridge=0.25,
        methods=("fixed_expert_mle",),
        warmup_modes=(False,),
    )

    results = study.executor(runtime, experiment_numerics).run_many(study.jobs())

    assert len(results) == 2
    assert all(result.summary["ridge"] == 0.25 for result in results)
    assert all(result.arrays["evaluation_sample_counts"].tolist() == [6] for result in results)


def test_lobeke_fixed_expert_mle_rejects_negative_ridge() -> None:
    with np.testing.assert_raises_regex(ValueError, "fixed_expert_mle_ridge"):
        LobekeComparisonStudy(
            ols=FixedLeaderOLSConfig(pseudocount=0.5, smoothing_warning_ratio=0.1), fixed_expert_mle_ridge=-1.0
        )


def test_lobeke_comparison_preserves_subtheoretical_lambda_multiplier() -> None:
    study = LobekeComparisonStudy(
        ols=FixedLeaderOLSConfig(pseudocount=0.5, smoothing_warning_ratio=0.1),
        seed_count=2,
        warmup_lambda_information_multiplier=0.03,
        no_warmup_lambda_information_multiplier=0.1,
        methods=("active",),
    )

    assert study._stackelberg_study(True).lambda_information_multiplier == 0.03
    assert study._stackelberg_study(False).lambda_information_multiplier == 0.1
