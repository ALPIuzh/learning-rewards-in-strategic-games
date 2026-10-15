"""Pack the model, final estimates, and tasks; replay without GUARD or raw trajectories."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig

RECORD = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pack", action="store_true")
parser.add_argument("--all", action="store_true", help="replay all 200 tasks; default checks two tasks per suite")
args = parser.parse_args()
started = time.monotonic()
if args.pack:
    from validate import PROTOCOLS, ROOT, SEEDS, SOURCE, SUITES, load_arrays, runtime

    game, numerics = runtime()
    ridge = np.load(ROOT / "ridge-sensitivity" / "estimates.npz")
    arrays = dict(
        omegas=np.stack([load_arrays(p)["omega_history"][:, -1] for p in PROTOCOLS]),
        protocols=np.array([p.slug for p in PROTOCOLS]),
        seeds=np.array(SEEDS),
        source_config_sha256=np.array(hashlib.sha256(SOURCE.read_bytes()).hexdigest()),
        features=game.problem.features,
        eta=np.array(game.problem.eta),
        radius=np.array(game.problem.radius),
        true_omega=game.true_omega,
        evaluation_policy=np.array(json.dumps(asdict(numerics.evaluation_policy), sort_keys=True)),
        ridge_omegas=ridge["omegas"],
        ridge_values=ridge["ridges"],
    )
    arrays.update(
        {f"tasks_{seed}": np.load(ROOT / "evaluation" / str(seed) / "tasks.npz")["rewards"] for seed, _ in SUITES}
    )
    path = RECORD / "compact-data.npz"
    if path.exists():
        with np.load(path) as previous:
            assert set(previous.files) == set(arrays)
            assert all(np.array_equal(previous[key], value) for key, value in arrays.items())
    else:
        np.savez_compressed(path, **arrays)
    print("packed model, 550 final estimates, 200 ridge refits, and 200 exact task matrices")
else:
    # No application build or raw-data loader is imported in this branch.
    with np.load(RECORD / "compact-data.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    seeds = tuple(int(x) for x in arrays["seeds"])
    protocols = list(arrays["protocols"])
    problem = compile_problem(arrays["features"], eta=float(arrays["eta"]), radius=float(arrays["radius"]))
    config = LeaderOptimizationConfig(PolicyOptimizationConfig(**json.loads(str(arrays["evaluation_policy"]))))
    estimates = arrays["omegas"].reshape(-1, problem.feature_dimension)
    traces = [
        SeedTrace(seed, np.array([1600]), {}, estimates[i : i + 1]) for i, seed in enumerate(seeds * len(protocols))
    ]
    report = []
    for name in sorted(key for key in arrays if key.startswith("tasks_")):
        task_seed = int(name.removeprefix("tasks_"))
        expected = np.load(RECORD / f"endpoints-{task_seed}.npz")
        assert np.array_equal(expected["seeds"], seeds)
        assert np.array_equal(expected["protocols"], protocols)
        count = len(arrays[name])
        for index, reward in enumerate(arrays[name][: count if args.all else 2]):
            values, _ = task_transfer_metrics(
                problem, arrays["true_omega"], reward, config, traces, task_seed=task_seed, task_index=index
            )
            values = values.reshape(len(protocols), len(seeds))
            delta = float(np.max(np.abs(values - expected["subopt"][index])))
            assert delta <= 1e-9, (task_seed, index, delta)
            report.append(
                dict(
                    suite=task_seed,
                    task=index,
                    max_absolute_difference=delta,
                    bit_identical=bool(np.array_equal(values, expected["subopt"][index])),
                )
            )
            if (index + 1) % 10 == 0:
                print(f"replayed {task_seed}: {index + 1}/{count} tasks", flush=True)
    output = RECORD / ("compact-replay-all.json" if args.all else "compact-replay-smoke.json")
    output.write_text(
        json.dumps(
            dict(
                seconds=time.monotonic() - started,
                checked_tasks=len(report),
                deployments=len(report) * len(protocols) * len(seeds),
                checks=report,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"verified {len(report)} tasks from compact data in {time.monotonic() - started:.1f}s")
