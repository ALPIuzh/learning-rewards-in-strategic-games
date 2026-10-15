"""Reproduce the frozen validation protocol without changing existing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.task_sets import StructuredTaskSet, normalized_structured_tasks
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from experiments.guard.paper.run_sensitivity import PilotDesign, Protocol, _protocols, _run_protocols
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)

ROOT = Path("artifacts/scientific-validation-20260919")
RECORD = Path("archive/scientific-validation-20260919")
SOURCE = Path("reproducibility/data/lobeke/ridge_zero_tuning/source_config.yaml")
SEEDS = tuple(range(2026092000, 2026092050))
PROTOCOLS = (
    *_protocols((0, 200), (0.003, 0.03)),
    *(
        Protocol(
            f"{method}_{'warm200' if warm else 'no_warm'}", method, warm, 200 if warm else 0, 0.003 if warm else 0.03
        )
        for method in ("fixed_random", "random_each_round")
        for warm in (False, True)
    ),
    *(Protocol(method, method, False, 0, 1.0) for method in ("follower_ols", "zero_sum_ols")),
)
SUITES = ((202609200, 100), (202609201, 50), (202609202, 50))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime():
    cfg = OmegaConf.load(SOURCE)
    numerics = instantiate(cfg.numerics)
    numerics = replace(numerics, mle=replace(numerics.mle, ridge=0.0))
    return instantiate(cfg.application).build(numerics), numerics


def train():
    for start in range(0, len(SEEDS), 10):
        seeds = SEEDS[start : start + 10]
        design = PilotDesign(
            seeds, StructuredTaskSet("prospective", 100, SUITES[0][0], 1000), PROTOCOLS, (0, 200), (0.003, 0.03), 0.0
        )
        began = time.monotonic()
        _run_protocols(SOURCE, ROOT / "confirmation" / f"batch-{start:02d}", 1600, design)
        print(f"batch {start} complete in {time.monotonic() - began:.1f}s", flush=True)


def load_arrays(protocol):
    batches = []
    for start in range(0, len(SEEDS), 10):
        with np.load(ROOT / "confirmation" / f"batch-{start:02d}" / f"{protocol.slug}.npz") as data:
            batches.append({key: data[key] for key in data.files if data[key].ndim > 0})
    return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}


def evaluate():
    game, numerics = runtime()
    config = LeaderOptimizationConfig(numerics.evaluation_policy)
    estimates = np.concatenate([load_arrays(p)["omega_history"][:, -1, :] for p in PROTOCOLS])
    traces = [
        SeedTrace(seed, np.array([1600]), {}, estimates[i : i + 1]) for i, seed in enumerate(SEEDS * len(PROTOCOLS))
    ]
    for task_seed, count in SUITES:
        directory = ROOT / "evaluation" / str(task_seed)
        directory.mkdir(parents=True, exist_ok=True)
        rewards = normalized_structured_tasks(game, StructuredTaskSet("prospective", count, task_seed, 1000))
        task_file = directory / "tasks.npz"
        if task_file.exists():
            assert np.array_equal(np.load(task_file)["rewards"], rewards)
        else:
            np.savez_compressed(task_file, rewards=rewards)
        for index, reward in enumerate(rewards):
            path = directory / f"task-{index:03d}.npz"
            if path.exists():
                continue
            began = time.monotonic()
            subopt, fisher = task_transfer_metrics(
                game.problem, game.true_omega, reward, config, traces, task_seed=task_seed, task_index=index
            )
            np.savez_compressed(
                path,
                subopt=subopt.reshape(len(PROTOCOLS), len(SEEDS)),
                fisher=fisher.reshape(len(PROTOCOLS), len(SEEDS)),
            )
            print(f"suite {task_seed} task {index + 1}/{count}: {time.monotonic() - began:.1f}s", flush=True)


def numerical_check(pilot=False):
    game, numerics = runtime()
    config = LeaderOptimizationConfig(numerics.evaluation_policy)
    strict = LeaderOptimizationConfig(replace(numerics.evaluation_policy, random_starts=64, tolerance=1e-12))
    if pilot:
        paths = sorted((ROOT / "pilot").glob("*.npz"))
        estimates = [
            (p.stem, int(data["seeds"][i]), data["omega_history"][i, -1])
            for p in paths
            for data in [dict(np.load(p))]
            for i in range(2)
        ]
        task_seed = 202609190
        rewards = normalized_structured_tasks(game, StructuredTaskSet("pilot", 5, task_seed, 1000))[:3]
    else:
        estimates = [(p.slug, SEEDS[i], load_arrays(p)["omega_history"][i, -1]) for p in PROTOCOLS for i in range(5)]
        task_seed = SUITES[0][0]
        rewards = np.load(ROOT / "evaluation" / str(task_seed) / "tasks.npz")["rewards"][:10]
    records = []
    for index, reward in enumerate(rewards):
        normal_oracle = optimize_leader_policy_batch(
            game.problem, game.true_omega[None, :], reward, config, (task_seed + index,)
        )[0]
        strict_oracle = optimize_leader_policy(game.problem, game.true_omega, reward, strict, seed=task_seed + index)
        for slug, seed, omega in estimates:
            opt_seed = task_seed * 1_000_000 + index * 100_000 + seed * 100
            normal = optimize_leader_policy_batch(game.problem, omega[None, :], reward, config, (opt_seed,))[0]
            refined = optimize_leader_policy(game.problem, omega, reward, strict, seed=opt_seed)
            normal_value = leader_value_and_gradient(game.problem, game.true_omega, reward, normal.policy)[0]
            refined_value = leader_value_and_gradient(game.problem, game.true_omega, reward, refined.policy)[0]
            records.append(
                dict(
                    task=index,
                    protocol=slug,
                    seed=seed,
                    oracle_improvement=strict_oracle.value - normal_oracle.value,
                    fitted_objective_improvement=refined.value - normal.value,
                    deployed_value_change=refined_value - normal_value,
                    normal_subopt=normal_oracle.value - normal_value,
                    strict_subopt=strict_oracle.value - refined_value,
                )
            )
            if not pilot:
                saved = np.load(ROOT / "evaluation" / str(task_seed) / f"task-{index:03d}.npz")["subopt"]
                value = float(saved[[p.slug for p in PROTOCOLS].index(slug), SEEDS.index(seed)])
                records[-1].update(
                    saved_subopt=value,
                    singleton_replay_change=records[-1]["normal_subopt"] - value,
                    strict_change_from_saved=records[-1]["strict_subopt"] - value,
                )
        print(f"numerical check task {index} complete", flush=True)
    save_json(RECORD / ("pilot-numerics.json" if pilot else "confirmation-numerics.json"), records)


def interval(values, confidence=0.95):
    values = np.asarray(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(student_t.ppf((1 + confidence) / 2, len(values) - 1) * se)
    return dict(mean=mean, standard_error=se, lower=mean - half, upper=mean + half)


def analyze():
    slugs = [p.slug for p in PROTOCOLS]
    retained = ("active_no_warm_lambda0p03", "active_warm200_lambda0p003")
    primary_pairs = [(slug, "fixed_expert_matched_ridge") for slug in retained]
    secondary_pairs = [(f"active_warm200_lambda{mult}", f"active_no_warm_lambda{mult}") for mult in ("0p003", "0p03")]
    secondary_pairs += [(slug, p.slug) for slug in retained for p in PROTOCOLS[5:]]
    results = {}
    for task_seed, count in SUITES:
        values = np.stack(
            [np.load(ROOT / "evaluation" / str(task_seed) / f"task-{i:03d}.npz")["subopt"] for i in range(count)]
        )
        assert values.shape == (count, len(PROTOCOLS), len(SEEDS))
        assert np.isfinite(values).all() and np.min(values) >= 0
        means = values.mean(axis=0)
        comparisons = {}
        for left, right in primary_pairs + secondary_pairs:
            differences = means[slugs.index(left)] - means[slugs.index(right)]
            stat = interval(differences)
            se = stat["standard_error"]
            stat["p_two_sided"] = float(2 * student_t.sf(abs(stat["mean"] / se), len(SEEDS) - 1)) if se else 1.0
            stat["simultaneous95_for_two_primary"] = (
                interval(differences, 0.975) if (left, right) in primary_pairs else None
            )
            rng = np.random.default_rng(202609299)
            boot = differences[rng.integers(len(SEEDS), size=(20000, len(SEEDS)))].mean(axis=1)
            stat["bootstrap95"] = np.quantile(boot, [0.025, 0.975]).tolist()
            stat["paired_seed_differences"] = differences.tolist()
            comparisons[f"{left}_minus_{right}"] = stat
        keys = [f"{a}_minus_{b}" for a, b in primary_pairs]
        ordered = sorted(keys, key=lambda key: comparisons[key]["p_two_sided"])
        running = 0.0
        for rank, key in enumerate(ordered):
            running = max(running, min(1.0, (2 - rank) * comparisons[key]["p_two_sided"]))
            comparisons[key]["holm_p_two_primary"] = running
        interaction = (
            means[slugs.index("active_warm200_lambda0p03")]
            - means[slugs.index("active_no_warm_lambda0p03")]
            - means[slugs.index("active_warm200_lambda0p003")]
            + means[slugs.index("active_no_warm_lambda0p003")]
        )
        results[str(task_seed)] = dict(
            task_count=count,
            means={s: interval(means[i]) for i, s in enumerate(slugs)},
            comparisons=comparisons,
            interaction=interval(interaction),
        )
        np.savez_compressed(
            RECORD / f"endpoints-{task_seed}.npz", subopt=values, seeds=np.array(SEEDS), protocols=np.array(slugs)
        )
    results["source_task"] = {
        p.slug: interval(load_arrays(p)["metric__stackelberg_suboptimality"][:, -1]) for p in PROTOCOLS
    }
    save_json(RECORD / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "pilot-numerics", "numerics", "analyze"))
    args = parser.parse_args()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    {
        "train": train,
        "evaluate": evaluate,
        "pilot-numerics": lambda: numerical_check(True),
        "numerics": numerical_check,
        "analyze": analyze,
    }[args.command]()
    save_json(
        RECORD / f"execution-{args.command}.json",
        dict(
            command=args.command,
            seconds=time.monotonic() - started,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            script_sha256=script_sha256,
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((RECORD / "PROTOCOL.md").read_bytes()).hexdigest(),
        ),
    )
