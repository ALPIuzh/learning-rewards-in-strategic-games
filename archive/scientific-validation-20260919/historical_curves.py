"""Preserve and numerically audit every retained Lobeke transfer-curve checkpoint."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from mairl.core.problem import compile_problem
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz

RECORD = Path(__file__).parent
ROOT = Path("artifacts/scientific-validation-20260919/historical-curves")
PACK = RECORD / "historical-curve-inputs.npz"
SOURCE = Path("reproducibility/data/lobeke/final_evaluation_unit_reward_v2/data.npz")


def initialize():
    global DATA, PROBLEM, CONFIG
    DATA = dict(np.load(PACK))
    PROBLEM = compile_problem(DATA["features"], eta=float(DATA["eta"]), radius=float(DATA["radius"]))
    CONFIG = LeaderOptimizationConfig(PolicyOptimizationConfig(**json.loads(str(DATA["evaluation_policy"]))))


def solve(request):
    task, index, replay = request
    seed = 9200 * 1_000_000 + task * 100_000 + int(DATA["optimization_offsets"][index])
    reward = DATA["tasks_9200"][task]
    omega = DATA["flat_omegas"][index]
    full = optimize_leader_policy(
        PROBLEM,
        omega,
        reward,
        replace(CONFIG, policy=replace(CONFIG.policy, random_starts=64, tolerance=1e-12)),
        seed=seed,
    )
    deployed = leader_value_and_gradient(PROBLEM, DATA["true_omega"], reward, full.policy)[0]
    fast_value = np.nan
    if replay:
        fast = optimize_leader_policy_batch(PROBLEM, omega[None], reward, CONFIG, (seed,))[0]
        fast_value = leader_value_and_gradient(PROBLEM, DATA["true_omega"], reward, fast.policy)[0]
    return deployed, fast_value


def pack():
    from experiments.guard.paper.figures.corrected_traces import load_figure_run_traces
    from experiments.guard.paper.figures.data import METHOD_SLUGS
    from experiments.guard.paper.figures.export_final import _subsampled_traces

    runs = (
        Path("artifacts/runs/guard-lobeke-paper-mle-corrected-v2/349676e5922332fd"),
        Path("artifacts/runs/guard-lobeke-paper-ols-corrected-v2/cf7b0f494e850f2d"),
    )
    traces = load_figure_run_traces(runs)
    previous = dict(np.load(RECORD / "historical-compact-data.npz"))
    old = np.load(SOURCE)
    arrays = {
        k: previous[k]
        for k in ("features", "eta", "radius", "true_omega", "evaluation_policy", "tasks_9200", "source_sha256")
    }
    omegas, offsets, names, seeds, checkpoints = [], [], [], [], []
    for key, name in METHOD_SLUGS.items():
        for trace in _subsampled_traces(traces[key]):
            i = list(old[f"{name}__seeds"]).index(trace.seed)
            assert np.array_equal(trace.omega_history[-1], old[f"{name}__final_omega"][i])
            assert np.array_equal(trace.checkpoints, old[f"{name}__checkpoints"])
            for j, omega in enumerate(trace.omega_history):
                omegas.append(omega)
                offsets.append(trace.seed * 100 + j)
                names.append(name)
                seeds.append(trace.seed)
                checkpoints.append(j)
    arrays.update(
        flat_omegas=np.array(omegas),
        optimization_offsets=np.array(offsets),
        protocols=np.array(names),
        seeds=np.array(seeds),
        checkpoints=np.array(checkpoints),
    )
    if PACK.exists():
        with np.load(PACK) as existing:
            assert all(np.array_equal(existing[k], v) for k, v in arrays.items())
    else:
        atomic_write_npz(PACK, arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "run"))
    args = parser.parse_args()
    start = time.monotonic()
    if not PACK.exists():
        pack()
    initialize()
    old = np.load(SOURCE)
    final_mask = np.array(
        [
            checkpoint == len(old[f"{name}__checkpoints"]) - 1
            for name, checkpoint in zip(DATA["protocols"], DATA["checkpoints"], strict=True)
        ]
    )
    selected = np.flatnonzero(DATA["seeds"] < 30002) if args.mode == "pilot" else np.flatnonzero(~final_mask)
    final = dict(np.load(RECORD / "historical-strict-endpoints.npz"))
    tasks = range(1) if args.mode == "pilot" else range(10)
    workers = 4
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for task in tasks:
            output = ROOT / f"task-{task:03d}.npz"
            if args.mode == "run" and output.exists():
                continue
            reward = DATA["tasks_9200"][task]
            strict_config = replace(CONFIG, policy=replace(CONFIG.policy, random_starts=64, tolerance=1e-12))
            oracle = optimize_leader_policy(PROBLEM, DATA["true_omega"], reward, strict_config, seed=9200 + task)
            values = np.array(
                list(pool.map(solve, [(task, int(i), args.mode == "pilot") for i in selected], chunksize=5))
            )
            losses = oracle.value - values[:, 0]
            assert losses.min() >= -1e-9
            if args.mode == "pilot":
                fast_oracle = optimize_leader_policy_batch(
                    PROBLEM, DATA["true_omega"][None], reward, CONFIG, (9200 + task,)
                )[0]
                expected = np.array(
                    [
                        old[f"{DATA['protocols'][i]}__taskwise_transfer_suboptimality"][
                            task,
                            list(old[f"{DATA['protocols'][i]}__seeds"]).index(DATA["seeds"][i]),
                            DATA["checkpoints"][i],
                        ]
                        for i in selected
                    ]
                )
                delta = float(np.max(np.abs(fast_oracle.value - values[:, 1] - expected)))
                assert delta < 1e-9
                atomic_write_json(
                    RECORD / "historical-curves-pilot.json",
                    dict(
                        seconds=time.monotonic() - start,
                        deployments=len(selected),
                        workers=workers,
                        maximum_replay_difference=delta,
                        maximum_strict_change=float(np.max(np.abs(losses - expected))),
                    ),
                )
            else:
                complete = np.empty(len(DATA["seeds"]))
                complete[selected] = losses
                for i in np.flatnonzero(final_mask):
                    complete[i] = final["subopt"][
                        task,
                        list(final["protocols"]).index(DATA["protocols"][i]),
                        list(final["seeds"]).index(DATA["seeds"][i]),
                    ]
                atomic_write_npz(output, dict(subopt=np.maximum(0, complete), oracle_value=np.array(oracle.value)))
            print(f"historical curves task {task + 1}/10", flush=True)
    if args.mode == "run":
        values = np.stack([np.load(ROOT / f"task-{i:03d}.npz")["subopt"] for i in range(10)])
        arrays = {}
        report = {}
        for name in dict.fromkeys(DATA["protocols"]):
            indices = np.flatnonzero(DATA["protocols"] == name)
            previous = old[f"{name}__taskwise_transfer_suboptimality"]
            current = values[:, indices].reshape(previous.shape)
            difference = current - previous
            arrays[f"{name}__taskwise_transfer_suboptimality"] = current
            report[name] = dict(
                maximum_absolute_deployment_change=float(np.max(np.abs(difference))),
                maximum_mean_curve_change=float(np.max(np.abs(difference.mean(axis=(0, 1))))),
                count_over_1e_minus_4=int(np.sum(np.abs(difference) > 1e-4)),
            )
        atomic_write_npz(RECORD / "historical-strict-curves.npz", arrays)
        atomic_write_json(
            RECORD / "historical-curves-numerics.json",
            dict(seconds=time.monotonic() - start, deployments=values.size, methods=report),
        )


if __name__ == "__main__":
    main()
