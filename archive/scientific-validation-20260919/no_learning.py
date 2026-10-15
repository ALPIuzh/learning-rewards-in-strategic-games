"""Measure a zero-reward-estimate deployment rule that uses no follower observations."""

import json
import time

import numpy as np
import strict_evaluation as strict

from mairl.planning.leader import leader_value_and_gradient, optimize_leader_policy

start = time.monotonic()
strict.initialize(strict.RECORD / "compact-data.npz")
values = {}
summary = {}
for suite in (202609200, 202609201, 202609202):
    losses = []
    for task, reward in enumerate(strict.DATA[f"tasks_{suite}"]):
        oracle = strict.oracle(suite, task)
        estimate = optimize_leader_policy(
            strict.PROBLEM, np.zeros(strict.PROBLEM.feature_dimension), reward, strict.CONFIG, seed=suite + task
        )
        loss = (
            oracle.value
            - leader_value_and_gradient(strict.PROBLEM, strict.DATA["true_omega"], reward, estimate.policy)[0]
        )
        assert loss >= -1e-9
        losses.append(max(0.0, loss))
    values[f"subopt_{suite}"] = np.array(losses)
    summary[str(suite)] = dict(
        mean=float(np.mean(losses)), minimum=float(np.min(losses)), maximum=float(np.max(losses)), tasks=len(losses)
    )
    print(f"no-learning suite {suite} complete", flush=True)
np.savez_compressed(strict.RECORD / "no-learning-endpoints.npz", **values)
(strict.RECORD / "no-learning-baseline.json").write_text(
    json.dumps(
        dict(
            seconds=time.monotonic() - start,
            rule="deploy optimally under omega=0; no follower observations",
            suites=summary,
        ),
        indent=2,
    )
    + "\n"
)
