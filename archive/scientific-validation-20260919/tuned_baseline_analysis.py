"""Fit the frozen selected baseline and analyze its independent confirmation."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from analyze_strict import comparison, interval

RECORD = Path(__file__).parent
ROOT = Path("artifacts/scientific-validation-20260919/tuned-baseline")
SELECTION = RECORD / "tuned-baseline-selection.json"


def pack():
    # Selection must be recorded before this function reads any fresh outcome.
    choice = json.loads(SELECTION.read_text())
    from tuned_baseline import SEEDS, SELECTED, TASKS
    from validate import runtime

    from experiments.guard.paper.figures.task_sets import normalized_structured_tasks
    from mairl.inference.mle import solve_mle_batch

    game, numerics = runtime()
    estimates = []
    names = []
    diagnostics = []
    for protocol in SELECTED:
        arrays = [dict(np.load(ROOT / f"batch-{start:02d}" / f"{protocol.slug}.npz")) for start in range(0, 50, 10)]
        seeds = np.concatenate([a["seeds"] for a in arrays])
        assert np.array_equal(seeds, SEEDS)
        assert all(a["policies"].shape[1] == 1600 and np.isfinite(a["omega_history"]).all() for a in arrays)
        for batch in arrays:
            assert np.isfinite(batch["policies"]).all()
            assert np.min(batch["policies"]) >= -1e-10
            assert np.max(np.abs(batch["policies"].sum(axis=-1) - 1)) < 1e-9
            assert np.max(np.linalg.norm(batch["omega_history"], axis=-1)) <= game.problem.radius + 1e-8
            assert np.min(batch["follower_actions"]) >= 0
            assert np.max(batch["follower_actions"]) < game.problem.n_follower_actions
        if protocol.slug == "fixed_expert_matched_ridge":
            policies = np.concatenate([a["policies"] for a in arrays])
            actions = np.concatenate([a["follower_actions"] for a in arrays])
            config = replace(numerics.mle, ridge=choice["ridge"])
            fits = solve_mle_batch(game.problem, policies, actions, config)
            omegas = np.stack([f.omega for f in fits])
            strict = solve_mle_batch(
                game.problem, policies, actions, replace(config, projected_gradient_tolerance=1e-9), initial=omegas
            )
            diagnostics = [
                dict(
                    seed=int(seed),
                    residual=float(f.projected_gradient_residual),
                    strict_coefficient_change=float(np.max(np.abs(f.omega - g.omega))),
                )
                for seed, f, g in zip(seeds, fits, strict, strict=True)
            ]
            names.append(f"fixed_expert_selected_ridge{choice['ridge']:g}")
        else:
            omegas = np.concatenate([a["omega_history"][:, -1] for a in arrays])
            names.append(protocol.slug)
        estimates.append(omegas)
    data = dict(
        omegas=np.stack(estimates),
        protocols=np.array(names),
        seeds=np.array(SEEDS),
        features=game.problem.features,
        eta=np.array(game.problem.eta),
        radius=np.array(game.problem.radius),
        true_omega=game.true_omega,
        evaluation_policy=np.array(json.dumps(asdict(numerics.evaluation_policy), sort_keys=True)),
        tasks_202609300=normalized_structured_tasks(game, TASKS),
        selection_sha256=np.array(hashlib.sha256(SELECTION.read_bytes()).hexdigest()),
    )
    path = RECORD / "tuned-baseline-compact-data.npz"
    if path.exists():
        with np.load(path) as previous:
            assert all(np.array_equal(previous[k], v) for k, v in data.items())
    else:
        np.savez_compressed(path, **data)
    (RECORD / "tuned-baseline-fit-checks.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def analyze():
    choice = json.loads(SELECTION.read_text())
    endpoint = RECORD / "tuned-baseline-endpoints.npz"
    if endpoint.exists():
        stored = dict(np.load(endpoint))
        names = list(stored["protocols"])
        seeds = stored["seeds"]
        values = stored["subopt"]
    else:
        rows = [dict(np.load(ROOT / "strict-evaluation" / "202609300" / f"task-{i:03d}.npz")) for i in range(50)]
        names = list(rows[0]["protocols"])
        seeds = rows[0]["seeds"]
        assert all(np.array_equal(r["seeds"], seeds) and np.array_equal(r["protocols"], names) for r in rows)
        values = np.stack([r["subopt"] for r in rows])
    assert values.shape == (50, 3, 50) and len(seeds) == 50
    assert np.isfinite(values).all() and values.min() >= 0
    means = values.mean(axis=0)
    baseline = next(name for name in names if name.startswith("fixed_expert"))
    comparisons = {
        f"{name}_minus_{baseline}": comparison(means[i] - means[names.index(baseline)], True)
        for i, name in enumerate(names)
        if name != baseline
    }
    running = 0.0
    for rank, key in enumerate(sorted(comparisons, key=lambda k: comparisons[k]["p_two_sided"])):
        running = max(running, min(1.0, (2 - rank) * comparisons[key]["p_two_sided"]))
        comparisons[key]["holm_p_two_primary"] = running
    results = dict(
        selected_ridge=choice["ridge"],
        learning_seeds=seeds.tolist(),
        task_seed=202609300,
        task_count=50,
        means={name: interval(means[i]) for i, name in enumerate(names)},
        comparisons=comparisons,
    )
    (RECORD / "tuned-baseline-results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(RECORD / "tuned-baseline-endpoints.npz", subopt=values, protocols=np.array(names), seeds=seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "analyze"))
    args = parser.parse_args()
    {"pack": pack, "analyze": analyze}[args.command]()
