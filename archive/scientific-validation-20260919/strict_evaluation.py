"""Full scalar multistart evaluation, parallel over independent seeded deployments."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz

RECORD = Path(__file__).parent
DEFAULT_OUTPUT = Path("artifacts/scientific-validation-20260919/strict-evaluation")


def initialize(path):
    global DATA, PROBLEM, CONFIG, OMEGAS, NAMES
    with np.load(path) as stored:
        DATA = {key: stored[key] for key in stored.files}
    PROBLEM = compile_problem(DATA["features"], eta=float(DATA["eta"]), radius=float(DATA["radius"]))
    config = PolicyOptimizationConfig(**json.loads(str(DATA["evaluation_policy"])))
    CONFIG = LeaderOptimizationConfig(replace(config, random_starts=64, tolerance=1e-12))
    OMEGAS = DATA["omegas"]
    NAMES = list(DATA["protocols"])
    if "ridge_omegas" in DATA:
        OMEGAS = np.concatenate((OMEGAS, DATA["ridge_omegas"]))
        NAMES += [f"fixed_expert_ridge{ridge:g}" for ridge in DATA["ridge_values"]]


def solve(request):
    task_seed, task, protocol, seed_index = request
    reward = DATA[f"tasks_{task_seed}"][task]
    seed = int(DATA["seeds"][seed_index])
    fitted = optimize_leader_policy(
        PROBLEM, OMEGAS[protocol, seed_index], reward, CONFIG, seed=task_seed * 1_000_000 + task * 100_000 + seed * 100
    )
    deployed = leader_value_and_gradient(PROBLEM, DATA["true_omega"], reward, fitted.policy)[0]
    return fitted.value, deployed, fitted.policy


def oracle(task_seed, task):
    reward = DATA[f"tasks_{task_seed}"][task]
    return optimize_leader_policy(PROBLEM, DATA["true_omega"], reward, CONFIG, seed=task_seed + task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "run"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", type=Path, default=RECORD / "compact-data.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    initialize(args.data)
    start = time.monotonic()
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    suites = sorted(int(key.removeprefix("tasks_")) for key in DATA if key.startswith("tasks_"))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.data,)) as pool:
        if args.mode == "pilot":
            reference = json.loads((RECORD / "confirmation-numerics.json").read_text())
            references = {(r["task"], r["protocol"], r["seed"]): r["strict_subopt"] for r in reference}
            checks = []
            for task in range(10):
                optimum = oracle(suites[0], task)
                requests = [(suites[0], task, p, i) for p in range(11) for i in range(5)]
                for request, (_, deployed, _) in zip(requests, pool.map(solve, requests, chunksize=5), strict=True):
                    _, _, p, i = request
                    actual = optimum.value - deployed
                    expected = references[(task, NAMES[p], int(DATA["seeds"][i]))]
                    checks.append(abs(actual - expected))
                print(f"parallel pilot: {task + 1}/10 tasks", flush=True)
            assert max(checks) <= 1e-12
            atomic_write_json(
                RECORD / "strict-parallel-pilot.json",
                {
                    "seconds": time.monotonic() - start,
                    "workers": args.workers,
                    "deployments": len(checks),
                    "maximum_difference_from_serial_full_multistart": max(checks),
                    "data_sha256": data_hash,
                },
            )
            return
        for task_seed in suites:
            if task_seed == suites[0]:
                selected = list(range(len(NAMES)))
            else:
                selected = [
                    NAMES.index(name)
                    for name in (
                        "active_no_warm_lambda0p03",
                        "active_warm200_lambda0p003",
                        "fixed_expert_matched_ridge",
                    )
                ]
            names = np.array([NAMES[i] for i in selected])
            for task in range(len(DATA[f"tasks_{task_seed}"])):
                path = args.output_dir / str(task_seed) / f"task-{task:03d}.npz"
                if path.exists():
                    with np.load(path) as saved:
                        assert str(saved["data_sha256"]) == data_hash
                        assert np.array_equal(saved["protocols"], names)
                        assert np.array_equal(saved["seeds"], DATA["seeds"])
                    continue
                began = time.monotonic()
                optimum = oracle(task_seed, task)
                requests = [(task_seed, task, p, i) for p in selected for i in range(len(DATA["seeds"]))]
                results = list(pool.map(solve, requests, chunksize=5))
                fitted_values = np.array([result[0] for result in results]).reshape(len(selected), -1)
                deployed_values = np.array([result[1] for result in results]).reshape(len(selected), -1)
                losses = optimum.value - deployed_values
                if losses.min() < -1e-9:
                    raise ValueError(f"a deployed policy exceeds the full-multistart oracle: {losses.min()}")
                atomic_write_npz(
                    path,
                    dict(
                        subopt=np.maximum(0.0, losses),
                        deployed_values=deployed_values,
                        fitted_values=fitted_values,
                        policies=np.array([r[2] for r in results]).reshape(len(selected), len(DATA["seeds"]), -1),
                        oracle_policy=optimum.policy,
                        oracle_value=np.array(optimum.value),
                        protocols=names,
                        seeds=DATA["seeds"],
                        data_sha256=np.array(data_hash),
                    ),
                )
                print(
                    f"strict suite {task_seed}, task {task + 1}/{len(DATA[f'tasks_{task_seed}'])}: "
                    f"{time.monotonic() - began:.1f}s",
                    flush=True,
                )
    atomic_write_json(
        args.output_dir / "execution.json",
        {
            "seconds": time.monotonic() - start,
            "workers": args.workers,
            "data_sha256": data_hash,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "random_starts": 64,
            "tolerance": 1e-12,
        },
    )


if __name__ == "__main__":
    main()
