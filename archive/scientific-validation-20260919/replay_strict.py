"""Replay a bounded sample of both strict confirmations using only compact data."""

import json
import time

import numpy as np
import strict_evaluation as strict

started = time.monotonic()
checks = []
for data, endpoint, suite in (
    ("compact-data.npz", "strict-endpoints-202609200.npz", 202609200),
    ("tuned-baseline-compact-data.npz", "tuned-baseline-endpoints.npz", 202609300),
):
    strict.initialize(strict.RECORD / data)
    expected = np.load(strict.RECORD / endpoint)
    assert np.array_equal(expected["seeds"], strict.DATA["seeds"])
    for task in (0, 1):
        oracle = strict.oracle(suite, task)
        for p, name in enumerate(expected["protocols"]):
            for i in (0, 1):
                _, deployed, _ = strict.solve((suite, task, strict.NAMES.index(name), i))
                loss = max(0.0, oracle.value - deployed)
                difference = abs(loss - expected["subopt"][task, p, i])
                assert difference <= 1e-12, (suite, task, name, i, difference)
                checks.append(
                    dict(
                        suite=suite,
                        task=task,
                        protocol=str(name),
                        seed=int(strict.DATA["seeds"][i]),
                        absolute_difference=float(difference),
                    )
                )
(strict.RECORD / "strict-compact-replay.json").write_text(
    json.dumps(dict(seconds=time.monotonic() - started, deployments=len(checks), checks=checks), indent=2) + "\n"
)
print(f"Reproduced {len(checks)} strict deployments from compact data.")
