"""Read-only checks of the retained paper evidence and sparse-game claims."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import t

DEST = Path("archive/scientific-validation-20260919/existing-evidence.json")


def interval(values):
    mean = float(np.mean(values))
    half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, lower=mean - half, upper=mean + half)


result = {}
with np.load("reproducibility/data/lobeke/final_evaluation_unit_reward_v2/data.npz") as data:
    slugs = [k.removesuffix("__seeds") for k in data.files if k.endswith("__seeds")]
    metrics = ("target_suboptimality", "averaged_transfer_suboptimality")
    for metric in metrics:
        vectors = {slug: data[f"{slug}__{metric}"][:, -1] for slug in slugs}
        result[metric] = {
            "means": {slug: interval(values) for slug, values in vectors.items()},
            "paired_differences": {
                f"{active}_minus_{control}": interval(vectors[active] - vectors[control])
                for active in ("active_warm", "active_no_warm")
                for control in slugs
                if control not in ("active_warm", "active_no_warm")
            },
        }

run = Path("artifacts/runs/sparse-information-action-sweep/219b99056804830c")
manifest = json.loads((run / "manifest.json").read_text())
mass = []
for job in manifest["expected_jobs"]:
    if job["spec"]["method"] != "active":
        continue
    with np.load(run / "jobs" / job["job_id"] / "result.npz") as data:
        mass.append(data["policies"][:, 0])
mass = np.concatenate(mass)
eta = 2.0


def fisher(x, omega):
    return eta**2 * x**2 * expit(eta * omega * x) * expit(-eta * omega * x)


optimum = minimize_scalar(lambda x: -fisher(x, 2.0), bounds=(0.0, 1.0), method="bounded")
result["sparse_policy_claim"] = dict(
    active_jobs=len(mass) // 200,
    observed_policies=len(mass),
    min_mass_on_informative_action=float(mass.min()),
    fraction_interior=float(np.mean((mass > 1e-8) & (mass < 1 - 1e-8))),
    fisher_optimal_mass_at_estimate_2=float(optimum.x),
    fisher_at_interior=float(-optimum.fun),
    fisher_at_pure=float(fisher(1.0, 2.0)),
    explanation="Zero information iff mu[0]=0. With eta=2 and estimated omega=2, the information optimum is interior.",
)
DEST.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["sparse_policy_claim"], indent=2))

summary = json.loads(Path("reproducibility/data/lower_bound/summary.json").read_text())
rows = []
for cell in summary["metrics"]["stackelberg_suboptimality"]:
    d = cell["dimension"]
    budget = cell["total_samples"]
    epsilon = 0.5 * np.sqrt(d / budget)
    # Each replicate covers every coordinate; preserve that grouping for uncertainty.
    risk = interval(np.array(cell["coordinate_values"]).mean(axis=0))
    rows.append(
        dict(
            dimension=d,
            total_samples=budget,
            samples_per_dimension=cell["samples_per_dimension"],
            epsilon=epsilon,
            active_coordinate_mean=risk["mean"],
            active_ci95=risk,
            uniform_policy_no_learning=float(expit(epsilon) - expit(epsilon / d)),
            random_pure_policy_no_learning=float((1 - 1 / d) * (expit(epsilon) - 0.5)),
            worst_policy_ceiling=float(expit(epsilon) - 0.5),
        )
    )
(DEST.parent / "hard-family-baselines.json").write_text(json.dumps(rows, indent=2) + "\n")
