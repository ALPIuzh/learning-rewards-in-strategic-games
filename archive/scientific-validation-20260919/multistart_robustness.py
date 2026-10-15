"""Stress-check the repaired measurements with 256 starts and tighter tolerance."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import strict_evaluation as strict

started = time.monotonic()
records = []
root = Path("artifacts/scientific-validation-20260919")
for data, directory, suites in (
    ("compact-data.npz", root / "strict-evaluation", (202609200, 202609201, 202609202)),
    ("tuned-baseline-compact-data.npz", root / "tuned-baseline" / "strict-evaluation", (202609300,)),
):
    strict.initialize(strict.RECORD / data)
    strict.CONFIG = replace(strict.CONFIG, policy=replace(strict.CONFIG.policy, random_starts=256, tolerance=1e-13))
    for suite in suites:
        example = np.load(directory / str(suite) / "task-000.npz")
        names = list(example["protocols"])
        selected_names = [
            name
            for name in names
            if name
            in (
                "active_no_warm_lambda0p03",
                "active_warm200_lambda0p003",
                "fixed_expert_matched_ridge",
                "fixed_expert_ridge10",
            )
            or name.startswith("fixed_expert_selected")
        ]
        cases = {(task, names.index(name), i) for task in (0, 1) for name in selected_names for i in (0, 1)}
        if suite == 202609200:
            original = np.load(strict.RECORD / "endpoints-202609200.npz")["subopt"]
            ridge = np.load(strict.RECORD / "ridge-sensitivity-endpoints.npz")["subopt"]
            previous = np.concatenate((original, ridge), axis=1)
            current = np.load(strict.RECORD / "strict-endpoints-202609200.npz")["subopt"]
            differences = np.abs(current - previous)
            indices = np.argsort(differences.ravel(), kind="stable")[-10:]
            cases.update(tuple(int(i) for i in np.unravel_index(index, differences.shape)) for index in indices)
        oracles = {}
        for task, p, i in sorted(cases):
            if task not in oracles:
                oracles[task] = strict.oracle(suite, task)
            old = np.load(directory / str(suite) / f"task-{task:03d}.npz")
            fitted, deployed, _ = strict.solve((suite, task, strict.NAMES.index(names[p]), i))
            loss = oracles[task].value - deployed
            assert loss >= -1e-9
            records.append(
                dict(
                    suite=suite,
                    task=task,
                    protocol=names[p],
                    seed=int(strict.DATA["seeds"][i]),
                    fitted_objective_improvement=float(fitted - old["fitted_values"][p, i]),
                    oracle_improvement=float(oracles[task].value - old["oracle_value"]),
                    deployed_value_change=float(deployed - old["deployed_values"][p, i]),
                    suboptimality_change=float(max(0.0, loss) - old["subopt"][p, i]),
                )
            )
        print(f"256-start checks complete for {suite}", flush=True)
(strict.RECORD / "multistart-robustness.json").write_text(
    json.dumps(
        dict(
            seconds=time.monotonic() - started,
            selection=(
                "first two tasks/seeds for retained protocols and expert controls; "
                "also ten largest primary fast-to-strict changes"
            ),
            checks=records,
        ),
        indent=2,
    )
    + "\n"
)
