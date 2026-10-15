"""Audit final retained Lobeke endpoints without changing their published data."""

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import strict_evaluation as strict
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.data import load_runtime
from experiments.guard.paper.figures.task_sets import TRANSFER_EVALUATION_TASKS, normalized_structured_tasks
from mairl.planning.leader import leader_value_and_gradient, optimize_leader_policy_batch
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz

RECORD = Path(__file__).parent
OUTPUT = Path("artifacts/scientific-validation-20260919/historical-numerics")
SOURCE = Path("reproducibility/data/lobeke/final_evaluation_unit_reward_v2/data.npz")
RUN = Path("artifacts/runs/guard-lobeke-paper-mle-corrected-v2/349676e5922332fd")


def solve(request):
    task, protocol, index = request
    reward = strict.DATA["tasks_9200"][task]
    seed = 9200 * 1_000_000 + task * 100_000 + int(strict.DATA["seeds"][index]) * 100
    seed += int(strict.DATA["checkpoint_offsets"][protocol])
    config = replace(strict.CONFIG, policy=replace(strict.CONFIG.policy, random_starts=16, tolerance=1e-9))
    fast = optimize_leader_policy_batch(
        strict.PROBLEM, strict.OMEGAS[protocol, index : index + 1], reward, config, (seed,)
    )[0]
    full = strict.optimize_leader_policy(
        strict.PROBLEM, strict.OMEGAS[protocol, index], reward, strict.CONFIG, seed=seed
    )
    return (
        leader_value_and_gradient(strict.PROBLEM, strict.DATA["true_omega"], reward, fast.policy)[0],
        leader_value_and_gradient(strict.PROBLEM, strict.DATA["true_omega"], reward, full.policy)[0],
        full.value - fast.value,
    )


def main():
    start = time.monotonic()
    with np.load(SOURCE) as saved:
        names = [k.removesuffix("__final_omega") for k in saved.files if k.endswith("__final_omega")]
        old = np.stack([saved[f"{n}__taskwise_transfer_suboptimality"][:, :, -1] for n in names], axis=1)
        game = load_runtime(RUN)
        numerics = instantiate(OmegaConf.load(RUN / "resolved_config.yaml").numerics)
        arrays = dict(
            features=game.problem.features,
            eta=np.array(game.problem.eta),
            radius=np.array(game.problem.radius),
            true_omega=game.true_omega,
            evaluation_policy=np.array(json.dumps(asdict(numerics.evaluation_policy), sort_keys=True)),
            omegas=np.stack([saved[f"{n}__final_omega"] for n in names]),
            protocols=np.array(names),
            seeds=saved[f"{names[0]}__seeds"],
            checkpoint_offsets=np.array([len(saved[f"{n}__checkpoints"]) - 1 for n in names]),
            tasks_9200=normalized_structured_tasks(game, TRANSFER_EVALUATION_TASKS),
            original_subopt=old,
            source_sha256=np.array(hashlib.sha256(SOURCE.read_bytes()).hexdigest()),
        )
        assert all(np.array_equal(saved[f"{n}__seeds"], arrays["seeds"]) for n in names)
    packed = RECORD / "historical-compact-data.npz"
    if packed.exists():
        with np.load(packed) as previous:
            assert all(np.array_equal(previous[k], v) for k, v in arrays.items())
    else:
        atomic_write_npz(packed, arrays)
    strict.initialize(packed)
    with ProcessPoolExecutor(max_workers=2, initializer=strict.initialize, initargs=(packed,)) as pool:
        for task in range(10):
            output = OUTPUT / f"task-{task:03d}.npz"
            if output.exists():
                continue
            full_oracle = strict.oracle(9200, task)
            config = replace(strict.CONFIG, policy=numerics.evaluation_policy)
            fast_oracle = optimize_leader_policy_batch(
                game.problem, game.true_omega[None], arrays["tasks_9200"][task], config, (9200 + task,)
            )[0]
            requests = [(task, p, i) for p in range(len(names)) for i in range(50)]
            values = np.array(list(pool.map(solve, requests, chunksize=5))).reshape(len(names), 50, 3)
            fast_loss = fast_oracle.value - values[:, :, 0]
            full_loss = full_oracle.value - values[:, :, 1]
            assert np.max(np.abs(fast_loss - old[task])) < 1e-9
            assert full_loss.min() >= -1e-9
            atomic_write_npz(
                output,
                dict(
                    subopt=np.maximum(0, full_loss),
                    fast_replay=fast_loss,
                    fitted_improvement=values[:, :, 2],
                    oracle_improvement=np.array(full_oracle.value - fast_oracle.value),
                ),
            )
            print(f"historical task {task + 1}/10", flush=True)
    full = np.stack([np.load(OUTPUT / f"task-{i:03d}.npz")["subopt"] for i in range(10)])
    atomic_write_npz(
        RECORD / "historical-strict-endpoints.npz", dict(subopt=full, protocols=np.array(names), seeds=arrays["seeds"])
    )
    atomic_write_json(
        RECORD / "historical-numerics.json",
        dict(
            seconds=time.monotonic() - start,
            workers=2,
            source_sha256=str(arrays["source_sha256"]),
            methods={
                n: dict(
                    original_mean=float(old[:, i].mean()),
                    strict_mean=float(full[:, i].mean()),
                    max_absolute_change=float(np.max(np.abs(full[:, i] - old[:, i]))),
                    changed_over_1e_minus_6=int(np.sum(np.abs(full[:, i] - old[:, i]) > 1e-6)),
                )
                for i, n in enumerate(names)
            },
        ),
    )


if __name__ == "__main__":
    main()
