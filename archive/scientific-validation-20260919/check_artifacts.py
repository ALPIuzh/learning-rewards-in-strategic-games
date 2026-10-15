"""Check saved confirmation data; report numerical and information diagnostics."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from validate import PROTOCOLS, RECORD, ROOT, SEEDS, load_arrays, runtime, save_json

from mairl.core.math import cumulative_fisher_information_batch, fisher_information
from mairl.inference.data import InteractionDataset
from mairl.inference.mle import likelihood_and_gradient, solve_mle
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy

game, numerics = runtime()
report = {"information_scope": "Fisher information excludes the leader observations used by zero-sum OLS."}
source_reward = np.array(game.metadata["defender_reward"])
strict_policy = LeaderOptimizationConfig(replace(numerics.evaluation_policy, random_starts=64, tolerance=1e-12))
source_oracle = optimize_leader_policy(game.problem, game.true_omega, source_reward, strict_policy, seed=0)
source_before = leader_value_and_gradient(game.problem, game.true_omega, source_reward, game.fixed_mle_leader_policy)[0]
report["source_expert_check"] = dict(
    value_improvement=source_oracle.value - source_before,
    policy_total_variation=float(np.abs(source_oracle.policy - game.fixed_mle_leader_policy).sum() / 2),
    numerical_eta=game.problem.eta,
    raw_eta=game.metadata["eta"],
    parameter_radius=game.problem.radius,
)
geometry = {}
for name, policy in (
    ("source_stackelberg", game.fixed_mle_leader_policy),
    ("joint_qre", game.expert.max_policy),
    ("uniform", np.full(game.problem.n_leader_actions, 1 / game.problem.n_leader_actions)),
):
    eigenvalues = np.linalg.eigvalsh(fisher_information(game.problem, policy, game.true_omega))
    geometry[name] = dict(
        fisher_eigenvalues=eigenvalues.tolist(),
        condition_number=(float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-12 else None),
        positive_eigenvalues_above_1e_minus_12=int(np.sum(eigenvalues > 1e-12)),
    )
report["fixed_policy_geometry"] = geometry
for protocol in PROTOCOLS:
    arrays = load_arrays(protocol)
    assert np.array_equal(arrays["seeds"], SEEDS)
    assert arrays["omega_history"].shape[0] == 50
    assert np.all(arrays["trace_sample_counts"][:, -1] == 1600)
    assert np.all(np.diff(arrays["trace_sample_counts"], axis=1) > 0)
    assert arrays["policies"].shape == (50, 1600, 23)
    assert arrays["follower_actions"].shape == (50, 1600)
    assert np.min(arrays["follower_actions"]) >= 0 and np.max(arrays["follower_actions"]) < 23
    for key, values in arrays.items():
        if np.issubdtype(values.dtype, np.number):
            assert np.isfinite(values).all(), (protocol.slug, key)
    policies = arrays["policies"]
    assert np.min(policies) >= -1e-10
    assert np.max(np.abs(policies.sum(axis=-1) - 1)) < 1e-9
    mle_residual = arrays.get("diagnostic__mle_residual", np.array([0.0]))
    assert np.max(mle_residual) <= 1.01e-6
    policy_residual = arrays.get("diagnostic__policy_residual", np.array([]))
    if protocol.method == "active":
        assert policy_residual.shape == (50, 1600 - protocol.warmup_budget)
    assert np.max(policy_residual, initial=0.0) <= 1.01e-5
    if protocol.method not in ("zero_sum_ols", "follower_ols"):
        assert np.max(np.linalg.norm(arrays["omega_history"], axis=-1)) <= game.problem.radius + 1e-8
    if protocol.method in ("active", "fixed_random", "random_each_round") and not protocol.warmup_enabled:
        assert np.all(arrays["omega_history"][:, 0] == 0)

    true_omegas = np.broadcast_to(game.true_omega, (50, game.problem.feature_dimension))
    info = cumulative_fisher_information_batch(game.problem, policies, true_omegas)
    eigenvalues = np.linalg.eigvalsh(info)
    entry = dict(
        max_mle_residual=(float(np.max(mle_residual)) if "diagnostic__mle_residual" in arrays else None),
        fisher_eigenvalues_at_truth=eigenvalues.tolist(),
        final_coefficient_errors=np.linalg.norm(arrays["omega_history"][:, -1] - game.true_omega, axis=1).tolist(),
    )
    if protocol.method not in ("zero_sum_ols", "follower_ols"):
        checks = []
        strict = replace(numerics.mle, objective_tolerance=1e-14, projected_gradient_tolerance=1e-9)
        for i in range(5):
            dataset = InteractionDataset(policies[i], arrays["follower_actions"][i])
            omega = arrays["omega_history"][i, -1]
            fitted = solve_mle(game.problem, dataset, strict, initial=omega)
            before = likelihood_and_gradient(game.problem, dataset, omega, ridge=0.0)[0]
            checks.append(
                dict(
                    seed=SEEDS[i],
                    max_coefficient_change=float(np.max(np.abs(fitted.omega - omega))),
                    likelihood_improvement=float(fitted.log_likelihood - before),
                    strict_residual=float(fitted.projected_gradient_residual),
                )
            )
        entry["strict_mle_checks"] = checks
    report[protocol.slug] = entry
    print(f"validated {protocol.slug}", flush=True)

warm = [load_arrays(p) for p in PROTOCOLS if p.warmup_enabled]
for arrays in warm[1:]:
    assert np.array_equal(arrays["policies"][:, :200], warm[0]["policies"][:, :200])
    assert np.array_equal(arrays["follower_actions"][:, :200], warm[0]["follower_actions"][:, :200])
    assert np.array_equal(arrays["omega_history"][:, 0], warm[0]["omega_history"][:, 0])
ols = [load_arrays(p) for p in PROTOCOLS if p.method in ("zero_sum_ols", "follower_ols")]
assert np.array_equal(ols[0]["follower_actions"], ols[1]["follower_actions"])
report["paired_warmup_data_and_initial_estimates_identical"] = True
report["ols_follower_observations_identical"] = True

original = json.loads((RECORD / "existing-data-sha256.json").read_text())
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in original.items())
report["original_files_verified_unchanged"] = len(original)
save_json(RECORD / "quality-checks.json", report)
files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.rglob("*")) if path.is_file()}
save_json(RECORD / "new-artifact-sha256.json", files)
