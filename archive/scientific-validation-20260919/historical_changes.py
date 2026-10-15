"""Locate historical curve repairs and stress-check the materially changed deployments."""

import json
from dataclasses import replace

import historical_curves as h
import numpy as np

from mairl.planning.leader import leader_value_and_gradient, optimize_leader_policy

h.initialize()
old = np.load(h.SOURCE)
new = np.load(h.RECORD / "historical-strict-curves.npz")
checks = []
policies = []
for name in dict.fromkeys(h.DATA["protocols"]):
    key = f"{name}__taskwise_transfer_suboptimality"
    for task, i, j in np.argwhere(np.abs(new[key] - old[key]) > 1e-4):
        seed = int(old[f"{name}__seeds"][i])
        index = np.flatnonzero(
            (h.DATA["protocols"] == name) & (h.DATA["seeds"] == seed) & (h.DATA["checkpoints"] == j)
        ).item()
        omega = h.DATA["flat_omegas"][index]
        reward = h.DATA["tasks_9200"][task]
        opt_seed = 9200 * 1_000_000 + int(task) * 100_000 + int(h.DATA["optimization_offsets"][index])
        solutions = [
            optimize_leader_policy(
                h.PROBLEM,
                omega,
                reward,
                replace(h.CONFIG, policy=replace(h.CONFIG.policy, random_starts=starts, tolerance=tol)),
                seed=opt_seed,
            )
            for starts, tol in ((64, 1e-12), (256, 1e-13))
        ]
        values = [leader_value_and_gradient(h.PROBLEM, h.DATA["true_omega"], reward, s.policy)[0] for s in solutions]
        checks.append(
            dict(
                protocol=name,
                task=int(task),
                seed=seed,
                samples=int(old[f"{name}__checkpoints"][j]),
                original_loss=float(old[key][task, i, j]),
                repaired_loss=float(new[key][task, i, j]),
                fitted_objective_improvement_256=float(solutions[1].value - solutions[0].value),
                deployed_value_change_256=float(values[1] - values[0]),
                original_mean_at_checkpoint=float(old[key][:, :, j].mean()),
                repaired_mean_at_checkpoint=float(new[key][:, :, j].mean()),
            )
        )
        policies.append(np.stack([s.policy for s in solutions]))
(h.RECORD / "historical-material-changes.json").write_text(json.dumps(checks, indent=2) + "\n")
np.savez_compressed(h.RECORD / "historical-material-policies.npz", policies=np.stack(policies))
print(json.dumps(checks, indent=2))
