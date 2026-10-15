from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.minimax_lower_bound.analysis import (
    FinalMetricRecord,
    aggregate_coordinate_risk,
    paired_bootstrap_loglog_slope,
)
from experiments.minimax_lower_bound.application import (
    build_family_runtime,
    build_instance_runtime,
    exact_hard_instance_metrics,
    make_minimax_instance,
)
from experiments.minimax_lower_bound.profile import projected_work_factor
from experiments.minimax_lower_bound.simulation import run_hard_instance_batch, select_hard_information_vertices
from experiments.minimax_lower_bound.solver import solve_one_hot_mle_batch
from experiments.minimax_lower_bound.study import MinimaxLowerBoundStudy, balanced_coordinate_schedule
from mairl.core.math import follower_probabilities
from mairl.exceptions import ConfigurationError
from mairl.exploration.policy import select_information_policy_batch
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.inference.mle import MLEConfig, solve_mle_batch
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy, optimize_leader_policy_batch
from mairl_exp.config import ExperimentNumerics, WarmupNumerics
from mairl_exp.evaluation.metrics import FisherTransferMetricConfig, StackelbergFisherError


def _numerics() -> ExperimentNumerics:
    policy = PolicyOptimizationConfig(
        random_starts=1, max_iterations=100, batch_prefilter_iterations=2, batch_refine_starts=1
    )
    return ExperimentNumerics(
        mle=MLEConfig(ridge=1e-4, max_iterations=100),
        warmup=WarmupNumerics(),
        active_policy=policy,
        evaluation_policy=policy,
    )


def test_minimax_instance_has_exact_features_reward_and_parameters() -> None:
    instance = make_minimax_instance(32, 128, seed=92000, active_coordinate=7)
    expected = np.zeros((32, 2, 32))
    expected[:, 1, :] = np.eye(32)
    expected_reward = np.column_stack((np.zeros(32), np.ones(32)))
    expected_omega = np.zeros(32)
    expected_omega[7] = 0.25

    assert np.array_equal(instance.problem.features, expected)
    assert np.array_equal(instance.leader_reward, expected_reward)
    assert np.array_equal(instance.true_omega, expected_omega)
    assert instance.problem.eta == 1.0
    assert instance.problem.radius == 1.0
    assert instance.epsilon == pytest.approx(0.5 * np.sqrt(32 / 128))
    assert np.linalg.norm(instance.true_omega) <= instance.problem.radius


def test_coordinate_schedule_is_balanced_and_reused_across_k() -> None:
    study = MinimaxLowerBoundStudy(
        dimensions=(32,), samples_per_dimension=(4, 8), seed_start=92000, replicates_per_coordinate=2
    )
    first = study.jobs()
    second = study.jobs()

    assert first == second
    expected = balanced_coordinate_schedule(32, 2, design_seed=study.coordinate_design_seed)
    for ratio in study.samples_per_dimension:
        cell = [job for job in first if job.samples_per_dimension == ratio]
        assert len(cell) == 2
        flattened = [(job.replicate, coordinate) for job in cell for coordinate in job.active_coordinates]
        assert flattened == list(expected)
        assert (
            np.bincount([coordinate for job in cell for coordinate in job.active_coordinates], minlength=32).tolist()
            == [2] * 32
        )
        assert [seed for job in cell for seed in job.seeds] == list(range(92000, 92064))
    assert all(job.as_dict()["active_coordinates"] == job.active_coordinates for job in first)


def test_instance_is_equivariant_to_simultaneous_coordinate_action_permutation() -> None:
    original = make_minimax_instance(32, 128, seed=1, active_coordinate=7)
    permutation = np.random.default_rng(19).permutation(32)
    permuted_coordinate = int(np.flatnonzero(permutation == 7)[0])
    rebuilt = make_minimax_instance(32, 128, seed=1, active_coordinate=permuted_coordinate)

    assert np.array_equal(original.problem.features[permutation][:, :, permutation], rebuilt.problem.features)
    assert np.array_equal(original.true_omega[permutation], rebuilt.true_omega)
    assert np.array_equal(original.leader_reward[permutation], rebuilt.leader_reward)


def test_final_grid_job_and_sample_accounting() -> None:
    study = MinimaxLowerBoundStudy()
    jobs = study.jobs()

    assert len(jobs) == 2 * 5 * 16 == 160
    assert {seed for job in jobs for seed in job.seeds} == set(range(92000, 93024))
    assert {job.dimension for job in jobs} == {33, 64}
    assert {job.samples_per_dimension for job in jobs} == {4, 8, 16, 32, 64}
    assert all(job.total_samples == job.dimension * job.samples_per_dimension for job in jobs)
    assert sum(job.dimension * job.total_samples for job in jobs) == 10_287_040


def test_runtime_metrics_have_finite_checkpoint_shapes() -> None:
    instance = make_minimax_instance(32, 128, seed=92000)
    runtime = build_instance_runtime(instance, _numerics())
    estimates = np.vstack((np.zeros(32), instance.true_omega))
    metrics = runtime.evaluate(estimates, seed=92000)

    assert set(metrics) == {"stackelberg_suboptimality", "fisher_transfer_error", "coefficient_l2_error"}
    assert all(value.shape == (2,) for value in metrics.values())
    assert all(np.all(np.isfinite(value)) for value in metrics.values())
    assert metrics["coefficient_l2_error"][1] == pytest.approx(0.0)
    assert metrics["stackelberg_suboptimality"][1] == pytest.approx(0.0, abs=1e-8)


def test_closed_form_metrics_have_expected_values() -> None:
    instance = make_minimax_instance(32, 128, seed=1, active_coordinate=7)
    estimates = np.vstack((np.zeros(32), instance.true_omega))
    metrics = exact_hard_instance_metrics(instance, estimates)

    expected_gap = 1.0 / (1.0 + np.exp(-instance.epsilon)) - 0.5
    assert metrics["stackelberg_suboptimality"].tolist() == pytest.approx([expected_gap, 0.0])
    assert metrics["fisher_transfer_error"].tolist() == pytest.approx([0.5 * instance.epsilon, 0.0])
    assert metrics["coefficient_l2_error"].tolist() == pytest.approx([instance.epsilon, 0.0])


def test_closed_form_metrics_match_generic_optimizers() -> None:
    instance = make_minimax_instance(32, 128, seed=1, active_coordinate=7)
    rng = np.random.default_rng(91)
    estimates = rng.normal(size=(3, 32))
    estimates /= np.maximum(1.0, np.linalg.norm(estimates, axis=1))[:, None]
    numerics = _numerics()
    leader = LeaderOptimizationConfig(numerics.evaluation_policy)
    deployed = optimize_leader_policy_batch(instance.problem, estimates, instance.leader_reward, leader, (10, 11, 12))
    true_solution = optimize_leader_policy(instance.problem, instance.true_omega, instance.leader_reward, leader)
    generic_suboptimality = np.asarray(
        [
            true_solution.value
            - follower_probabilities(instance.problem, result.policy, instance.true_omega)
            @ (result.policy @ instance.leader_reward)
            for result in deployed
        ]
    )
    generic_fisher = StackelbergFisherError(
        instance.problem, instance.true_omega, instance.leader_reward, FisherTransferMetricConfig(leader=leader)
    ).evaluate_many(estimates, seed=10)
    exact = exact_hard_instance_metrics(instance, estimates)

    assert exact["stackelberg_suboptimality"] == pytest.approx(generic_suboptimality)
    assert exact["fisher_transfer_error"] == pytest.approx(generic_fisher)


def test_one_hot_mle_matches_generic_constrained_solver() -> None:
    instance = make_minimax_instance(32, 128, seed=1, active_coordinate=7)
    rng = np.random.default_rng(151)
    selected = rng.integers(0, 6, size=(3, 40))
    actions = rng.integers(0, 2, size=(3, 40))
    selected[0] = 0
    actions[0] = 1
    policies = np.eye(32)[selected]
    counts = np.stack([np.bincount(row, minlength=32) for row in selected], dtype=np.int64)
    successes = np.stack(
        [np.bincount(row, weights=outcomes, minlength=32) for row, outcomes in zip(selected, actions, strict=True)]
    ).astype(np.int64)
    config = _numerics().mle
    specialized = solve_one_hot_mle_batch(
        counts,
        successes,
        ridge=config.ridge,
        radius=instance.problem.radius,
        projected_gradient_tolerance=config.projected_gradient_tolerance,
        max_iterations=config.max_iterations,
    )
    generic = solve_mle_batch(instance.problem, policies, actions, config)

    assert specialized.omegas == pytest.approx(np.stack([result.omega for result in generic]), abs=2e-7)
    assert np.max(specialized.projected_gradient_residuals) <= config.projected_gradient_tolerance


def test_tiny_job_records_metrics_convergence_diagnostics_and_no_warmup() -> None:
    numerics = _numerics()
    study = MinimaxLowerBoundStudy(dimensions=(32,), samples_per_dimension=(4,), seed_start=92000)
    validation = make_minimax_instance(32, 128, seed=92000)
    result = study.executor(build_instance_runtime(validation, numerics), numerics).run(study.jobs()[0])

    assert result.summary["samples"] == 128
    assert result.summary["warmup"]["enabled"] is False
    assert result.summary["warmup"]["samples"] == 0
    assert result.summary["lambda_information"] == 1.0
    assert result.arrays["evaluation_sample_counts"][0].tolist() == [0, 128]
    assert result.arrays["omega_history"].shape == (32, result.arrays["trace_sample_counts"].shape[1], 32)
    assert result.arrays["trace_sample_counts"][0, [0, -1]].tolist() == [0, 128]
    assert result.arrays["selected_leader_actions"].shape == (32, 128)
    assert np.all(np.sum(result.arrays["final_action_counts"], axis=1) == 128)
    assert np.all(result.arrays["final_action_successes"] <= result.arrays["final_action_counts"])
    assert result.arrays["deployment__coordinate"].shape == (32,)
    assert result.arrays["deployment__active_coordinate_identified"].shape == (32,)
    assert set(result.arrays["instance__active_coordinate"].tolist()) == set(range(32))
    for name in ("metric__stackelberg_suboptimality", "metric__fisher_transfer_error", "metric__coefficient_l2_error"):
        assert result.arrays[name].shape == (32, 2)
        assert np.all(np.isfinite(result.arrays[name]))
    for name in (
        "diagnostic__information_gain",
        "diagnostic__mle_residual",
        "diagnostic__policy_residual",
        "diagnostic__mle_iterations",
    ):
        assert result.arrays[name].shape == (32, 128)
        assert np.all(np.isfinite(result.arrays[name]))


def test_vectorized_distinct_coordinates_are_reproducible_and_valid() -> None:
    numerics = _numerics()
    study = MinimaxLowerBoundStudy(dimensions=(32,), samples_per_dimension=(4,))
    job = study.jobs()[0]
    validation = make_minimax_instance(32, 128, seed=92000)
    executor = study.executor(build_instance_runtime(validation, numerics), numerics)
    first = executor.run(job)
    second = executor.run(job)

    assert set(job.active_coordinates) == set(range(32))
    assert first.summary["job"] == job.as_dict()
    assert first.summary["samples"] == 128
    assert first.summary["instance"]["active_coordinates"] == list(job.active_coordinates)
    for name in (
        "omega_history",
        "selected_leader_actions",
        "follower_actions",
        "metric__stackelberg_suboptimality",
        "metric__fisher_transfer_error",
    ):
        assert np.array_equal(first.arrays[name], second.arrays[name])
        assert np.all(np.isfinite(first.arrays[name]))


def test_vectorized_kernel_is_invariant_to_batch_partition() -> None:
    numerics = _numerics()
    batch = run_hard_instance_batch(
        dimension=32, total_samples=128, seeds=(92000, 92001), active_coordinates=(3, 17), numerics=numerics
    )
    singles = tuple(
        run_hard_instance_batch(
            dimension=32, total_samples=128, seeds=(seed,), active_coordinates=(coordinate,), numerics=numerics
        )[0]
        for seed, coordinate in ((92000, 3), (92001, 17))
    )

    for batched, single in zip(batch, singles, strict=True):
        for name in batched.arrays:
            if np.issubdtype(batched.arrays[name].dtype, np.floating):
                assert batched.arrays[name] == pytest.approx(single.arrays[name], abs=1e-14)
            else:
                assert np.array_equal(batched.arrays[name], single.arrays[name])


def test_profile_work_factor_accounts_for_every_coordinate_sample() -> None:
    study = MinimaxLowerBoundStudy()
    benchmark = max(study.jobs(), key=lambda job: job.dimension * job.total_samples)
    expected = sum(job.dimension * job.total_samples for job in study.jobs()) / (
        benchmark.dimension * benchmark.total_samples
    )
    assert projected_work_factor(study, benchmark.dimension * benchmark.total_samples) == expected


def test_closed_form_information_policy_matches_generic_optimizer() -> None:
    numerics = _numerics()
    instance = make_minimax_instance(32, 128, seed=1, active_coordinate=7)
    random = np.random.default_rng(810)
    omegas = random.normal(size=(3, 32))
    omegas /= np.maximum(1.0, np.linalg.norm(omegas, axis=1))[:, None]
    counts = random.integers(0, 40, size=(3, 32))
    variances = 1.0 / (2.0 + 2.0 * np.cosh(omegas))
    information_diagonal = 1.0 + counts * variances
    closed_coordinates, closed_gains, closed_residuals = select_hard_information_vertices(
        omegas,
        information_diagonal,
        tuple(np.random.default_rng(seed) for seed in (1, 2, 3)),
        tolerance=numerics.active_policy.projected_gradient_tolerance,
    )
    information = np.zeros((3, 32, 32))
    diagonal = np.arange(32)
    information[:, diagonal, diagonal] = information_diagonal
    generic = select_information_policy_batch(
        instance.problem,
        omegas,
        information,
        numerics.active_policy,
        tuple(np.random.default_rng(seed) for seed in (1, 2, 3)),
    )

    assert [result.information_gain for result in generic] == pytest.approx(closed_gains)
    assert [int(np.argmax(result.policy)) for result in generic] == closed_coordinates.tolist()
    assert np.max(closed_residuals) <= numerics.active_policy.projected_gradient_tolerance


def test_family_runtime_rejects_unknown_seed() -> None:
    runtime = build_family_runtime(32, 128, {92000: 7}, _numerics())
    with pytest.raises(ConfigurationError, match="absent from the balanced design"):
        runtime.oracle_factory(92001)


def test_coordinate_risk_aggregation_retains_mean_and_maximum() -> None:
    records = tuple(
        FinalMetricRecord(2, 4, coordinate, replicate, value)
        for coordinate, values in enumerate(((1.0, 3.0), (4.0, 6.0)))
        for replicate, value in enumerate(values)
    )
    cell = aggregate_coordinate_risk(records)[0]

    assert cell["replicates_per_coordinate"] == 2
    assert cell["coordinate_means"] == pytest.approx([2.0, 5.0])
    np.testing.assert_allclose(cell["coordinate_values"], [[1.0, 3.0], [4.0, 6.0]])
    assert cell["coordinate_standard_errors"] == pytest.approx([1.0, 1.0])
    assert cell["uniform_coordinate_mean"] == pytest.approx(3.5)
    assert cell["uniform_coordinate_standard_error"] == pytest.approx(np.std([1.0, 3.0, 4.0, 6.0], ddof=1) / 2)
    assert cell["maximum_coordinate_sample_mean"] == pytest.approx(5.0)
    assert cell["maximizing_coordinates"] == [1]


def test_paired_bootstrap_recovers_exact_loglog_slope() -> None:
    x = np.asarray([16.0, 32.0, 64.0])
    unit_scales = np.linspace(0.8, 1.2, 20)
    values = x[:, None] ** -0.5 * unit_scales[None, :]

    result = paired_bootstrap_loglog_slope(x, values, bootstrap_samples=200, seed=19)

    assert result["estimate"] == pytest.approx(-0.5)
    assert result["confidence_interval"] == pytest.approx([-0.5, -0.5])


@pytest.mark.parametrize(
    "study", [MinimaxLowerBoundStudy(dimensions=(32,)), MinimaxLowerBoundStudy(samples_per_dimension=(4,))]
)
def test_valid_reduced_grids_are_accepted(study: MinimaxLowerBoundStudy) -> None:
    assert study.jobs()


@pytest.mark.parametrize(
    "changes",
    [
        {"dimensions": (31,)},
        {"dimensions": (32, 32)},
        {"samples_per_dimension": (3,)},
        {"samples_per_dimension": (4, 4)},
        {"seed_start": -1},
        {"replicates_per_coordinate": 0},
        {"coordinate_design_seed": -1},
        {"confidence_delta": 1.0},
    ],
)
def test_invalid_study_configuration_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        replace(MinimaxLowerBoundStudy(), **changes)


@pytest.mark.parametrize(
    ("dimension", "samples", "seed", "coordinate"),
    [(31, 128, 0, 0), (32, 127, 0, 0), (32, 128, -1, None), (32, 128, 0, 32)],
)
def test_invalid_instance_configuration_is_rejected(dimension: int,
        samples: int, seed: int, coordinate: int | None) -> None:  # fmt: skip
    with pytest.raises(ConfigurationError):
        make_minimax_instance(dimension, samples, seed, active_coordinate=coordinate)
