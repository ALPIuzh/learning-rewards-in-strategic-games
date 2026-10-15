"""Exploratory final-estimator refits on unchanged fixed-expert observations."""

import time
from dataclasses import replace

import numpy as np
from validate import PROTOCOLS, RECORD, ROOT, SEEDS, interval, load_arrays, runtime, save_json

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from mairl.inference.mle import solve_mle_batch
from mairl.planning.leader import LeaderOptimizationConfig

started = time.monotonic()
directory = ROOT / "ridge-sensitivity"
directory.mkdir(exist_ok=True)
ridges = (0.01, 0.1, 1.0, 10.0)
game, numerics = runtime()
data = load_arrays(next(p for p in PROTOCOLS if p.method == "fixed_expert_mle"))
path = directory / "estimates.npz"
if path.exists():
    estimates = np.load(path)["omegas"]
else:
    fits = [
        solve_mle_batch(game.problem, data["policies"], data["follower_actions"], replace(numerics.mle, ridge=r))
        for r in ridges
    ]
    estimates = np.array([[fit.omega for fit in group] for group in fits])
    np.savez_compressed(
        path,
        omegas=estimates,
        seeds=np.array(SEEDS),
        ridges=np.array(ridges),
        residuals=np.array([[fit.projected_gradient_residual for fit in group] for group in fits]),
    )
    print("refitted four ridge values from zero on the same 50 trajectories", flush=True)
flat = estimates.reshape(-1, game.problem.feature_dimension)
traces = [SeedTrace(seed, np.array([1600]), {}, flat[i : i + 1]) for i, seed in enumerate(SEEDS * len(ridges))]
rewards = np.load(ROOT / "evaluation" / "202609200" / "tasks.npz")["rewards"]
config = LeaderOptimizationConfig(numerics.evaluation_policy)
for index, reward in enumerate(rewards):
    output = directory / f"task-{index:03d}.npz"
    if not output.exists():
        subopt, _ = task_transfer_metrics(
            game.problem, game.true_omega, reward, config, traces, task_seed=202609200, task_index=index
        )
        np.savez_compressed(output, subopt=subopt.reshape(len(ridges), len(SEEDS)))
    if (index + 1) % 10 == 0:
        print(f"ridge grid: {index + 1}/100 tasks; {time.monotonic() - started:.1f}s", flush=True)
    if time.monotonic() - started > 600:
        raise TimeoutError("Exploratory ridge check exceeded ten minutes; inspect saved checkpoints before continuing")
values = np.stack([np.load(directory / f"task-{index:03d}.npz")["subopt"] for index in range(100)])
means = values.mean(axis=0)
primary = np.load(RECORD / "endpoints-202609200.npz")
report = {
    "status": "exploratory; added after confirmation training began",
    "seconds": time.monotonic() - started,
    "task_seed": 202609200,
    "task_count": 100,
    "seeds": list(SEEDS),
    "ridges": {},
}
for i, ridge in enumerate(ridges):
    contrasts = {}
    for slug in ("active_no_warm_lambda0p03", "active_warm200_lambda0p003"):
        position = list(primary["protocols"]).index(slug)
        difference = primary["subopt"][:, position].mean(axis=0) - means[i]
        contrasts[slug + "_minus_expert"] = interval(difference)
    report["ridges"][str(ridge)] = {"mean_transfer": interval(means[i]), "paired_differences": contrasts}
save_json(RECORD / "ridge-sensitivity.json", report)
np.savez_compressed(RECORD / "ridge-sensitivity-endpoints.npz", subopt=values, seeds=np.array(SEEDS), ridges=ridges)
