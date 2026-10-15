"""Analyze repaired endpoints using the original paired comparisons."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).parent
RAW = Path("artifacts/scientific-validation-20260919/strict-evaluation")
RETAINED = ("active_no_warm_lambda0p03", "active_warm200_lambda0p003")
EXPERT = "fixed_expert_matched_ridge"


def interval(values, confidence=0.95):
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    half = float(student_t.ppf((1 + confidence) / 2, values.size - 1) * se)
    return dict(mean=mean, standard_error=se, lower=mean - half, upper=mean + half)


def comparison(values, primary=False):
    result = interval(values)
    se = result["standard_error"]
    result["p_two_sided"] = (
        float(2 * student_t.sf(abs(result["mean"] / se), values.size - 1))
        if se
        else (1.0 if result["mean"] == 0 else 0.0)
    )
    if primary:
        result["simultaneous95_for_two_primary"] = interval(values, 0.975)
    rng = np.random.default_rng(202609299)
    boot = values[rng.integers(values.size, size=(20000, values.size))].mean(axis=1)
    result.update(
        bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
        paired_seed_differences=values.tolist(),
        fraction_negative=float(np.mean(values < 0)),
        median=float(np.median(values)),
    )
    return result


def main():
    results = {}
    for task_seed, count in ((202609200, 100), (202609201, 50), (202609202, 50)):
        endpoint = ROOT / f"strict-endpoints-{task_seed}.npz"
        if endpoint.exists():
            stored = dict(np.load(endpoint))
            names = list(stored["protocols"])
            seeds = stored["seeds"]
            values = stored["subopt"]
        else:
            rows = [dict(np.load(RAW / str(task_seed) / f"task-{i:03d}.npz")) for i in range(count)]
            names = list(rows[0]["protocols"])
            seeds = rows[0]["seeds"]
            assert all(np.array_equal(row["protocols"], names) and np.array_equal(row["seeds"], seeds) for row in rows)
            values = np.stack([row["subopt"] for row in rows])
        assert len(seeds) == 50 and len(set(seeds)) == 50
        assert values.shape == (count, len(names), len(seeds))
        assert np.isfinite(values).all() and values.min() >= 0
        means = values.mean(axis=0)
        primary_pairs = [(name, EXPERT) for name in RETAINED]
        pairs = primary_pairs.copy()
        if task_seed == 202609200:
            pairs += [(f"active_warm200_lambda{m}", f"active_no_warm_lambda{m}") for m in ("0p003", "0p03")]
            pairs += [
                (name, baseline)
                for name in RETAINED
                for baseline in names
                if not baseline.startswith("active") and baseline != EXPERT
            ]
        comparisons = {
            f"{left}_minus_{right}": comparison(
                means[names.index(left)] - means[names.index(right)],
                task_seed == 202609200 and (left, right) in primary_pairs,
            )
            for left, right in pairs
        }
        if task_seed == 202609200:
            keys = sorted([f"{a}_minus_{b}" for a, b in primary_pairs], key=lambda k: comparisons[k]["p_two_sided"])
            running = 0.0
            for rank, key in enumerate(keys):
                running = max(running, min(1.0, (2 - rank) * comparisons[key]["p_two_sided"]))
                comparisons[key]["holm_p_two_primary"] = running
        old = np.load(ROOT / f"endpoints-{task_seed}.npz")
        ridge = np.load(ROOT / "ridge-sensitivity-endpoints.npz") if task_seed == 202609200 else None
        changes = {}
        for i, name in enumerate(names):
            if name in old["protocols"]:
                previous = old["subopt"][:, list(old["protocols"]).index(name)]
            else:
                ridge_value = float(name.removeprefix("fixed_expert_ridge"))
                previous = ridge["subopt"][:, list(ridge["ridges"]).index(ridge_value)]
            difference = values[:, i] - previous
            changes[name] = dict(
                mean_change=float(difference.mean()),
                maximum_absolute_change=float(np.max(np.abs(difference))),
                count_over_1e_minus_6=int(np.sum(np.abs(difference) > 1e-6)),
                count_over_1e_minus_4=int(np.sum(np.abs(difference) > 1e-4)),
            )
        results[str(task_seed)] = dict(
            task_count=count,
            means={name: interval(means[i]) for i, name in enumerate(names)},
            comparisons=comparisons,
            changes_from_fast_evaluation=changes,
        )
        if task_seed == 202609200:
            interaction = (
                means[names.index("active_warm200_lambda0p03")]
                - means[names.index("active_no_warm_lambda0p03")]
                - means[names.index("active_warm200_lambda0p003")]
                + means[names.index("active_no_warm_lambda0p003")]
            )
            results[str(task_seed)]["interaction"] = interval(interaction)
        np.savez_compressed(
            ROOT / f"strict-endpoints-{task_seed}.npz", subopt=values, protocols=np.array(names), seeds=seeds
        )
    (ROOT / "strict-results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
