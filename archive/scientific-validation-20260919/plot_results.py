"""Render additional audit figures; leave every retained paper figure untouched."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "mairl-scientific-validation-20260919",
    }
)
LABELS = {
    "active_no_warm_lambda0p003": "Active: no warm-up, λ/η² = .003",
    "active_no_warm_lambda0p03": "Active: no warm-up, λ/η² = .03 (retained)",
    "active_warm200_lambda0p003": "Active: warm-up 200, λ/η² = .003 (retained)",
    "active_warm200_lambda0p03": "Active: warm-up 200, λ/η² = .03",
    "fixed_expert_matched_ridge": "Fixed-expert MLE, ridge 0",
    "fixed_random_no_warm": "Fixed random: no warm-up",
    "fixed_random_warm200": "Fixed random: warm-up 200",
    "random_each_round_no_warm": "Online random: no warm-up",
    "random_each_round_warm200": "Online random: warm-up 200",
    "follower_ols": "Follower-only OLS",
    "zero_sum_ols": "Zero-sum OLS (also observes leader)",
}


def save(fig, name):
    for suffix in ("png", "svg", "pdf"):
        metadata = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}}.get(suffix)
        fig.savefig(ROOT / f"{name}.{suffix}", dpi=180, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


baseline = json.loads((ROOT / "hard-family-baselines.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
for dimension, color in ((33, "#0066aa"), (64, "#00847f")):
    rows = [r for r in baseline if r["dimension"] == dimension]
    x = np.array([r["samples_per_dimension"] for r in rows])
    active = np.array([r["active_coordinate_mean"] for r in rows])
    uniform = np.array([r["uniform_policy_no_learning"] for r in rows])
    axes[0].loglog(x, active, "o-", color=color, label=f"Active IRL, d={dimension}")
    axes[0].loglog(x, uniform, "--", color=color, label=f"No observations, uniform policy, d={dimension}")
    axes[1].plot(x, active / uniform, "o-", color=color, label=f"d={dimension}")
    axes[1].fill_between(
        x,
        np.array([r["active_ci95"]["lower"] for r in rows]) / uniform,
        np.array([r["active_ci95"]["upper"] for r in rows]) / uniform,
        color=color,
        alpha=0.12,
    )
axes[0].set(
    ylabel="Mean SubOpt", xlabel="Observations per dimension, K/d", title="The instance separation shrinks with K"
)
axes[0].legend(fontsize=8)
axes[1].axhline(1, color="0.5", linestyle="--")
axes[1].set(
    xscale="log",
    ylabel="Active mean / no-observation loss",
    xlabel="K/d",
    title="Descriptive ratio to an analytical baseline",
)
axes[1].legend()
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
fig.supxlabel("Shading: pointwise 95% intervals over 16 coordinate-balanced replicates.", fontsize=9)
save(fig, "hard-family-with-no-learning-baseline")

if (ROOT / "strict-results.json").exists():
    results = json.loads((ROOT / "strict-results.json").read_text())
    primary = results["202609200"]
    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    for i, (slug, _label) in enumerate(LABELS.items()):
        stat = primary["means"][slug]
        color = "#0066aa" if slug.startswith("active") else "#697581"
        ax.errorbar(
            stat["mean"] * 1e3,
            i,
            xerr=np.array([[stat["mean"] - stat["lower"]], [stat["upper"] - stat["mean"]]]) * 1e3,
            fmt="o",
            color=color,
            capsize=3,
        )
    ax.set(
        yticks=range(len(LABELS)),
        yticklabels=LABELS.values(),
        xlabel="Mean transfer SubOpt x 1,000",
        title="Fresh confirmation: 50 seeds, 100 tasks, full multistart evaluation",
    )
    ax.invert_yaxis()
    ax.text(0, -0.16, "Pointwise 95% seed-level intervals; conditional on the task suite.", transform=ax.transAxes)
    save(fig, "fresh-confirmation-methods")

    pairs = [
        ("active_no_warm_lambda0p03_minus_fixed_expert_matched_ridge", "Retained no-warm - fixed expert", True),
        ("active_warm200_lambda0p003_minus_fixed_expert_matched_ridge", "Retained warm - fixed expert", True),
        ("active_warm200_lambda0p003_minus_active_no_warm_lambda0p003", "Warm - no-warm, multiplier .003", False),
        ("active_warm200_lambda0p03_minus_active_no_warm_lambda0p03", "Warm - no-warm, multiplier .03", False),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
    for i, (key, _label, is_primary) in enumerate(pairs):
        stat = primary["comparisons"][key]
        if is_primary:
            stat = stat["simultaneous95_for_two_primary"]
        ax.errorbar(
            stat["mean"] * 1e3,
            i,
            xerr=np.array([[stat["mean"] - stat["lower"]], [stat["upper"] - stat["mean"]]]) * 1e3,
            fmt="o",
            color="#0066aa" if is_primary else "#bd7300",
            capsize=4,
        )
    ax.axvline(0, color="0.6", linewidth=1)
    ax.set(
        yticks=range(4),
        yticklabels=[p[1] for p in pairs],
        xlabel="Paired mean difference x 1,000",
        title="Primary transfer tests and matched warm-up contrasts",
    )
    ax.invert_yaxis()
    ax.text(
        0,
        -0.26,
        "Blue: simultaneous 95% intervals for the two primary contrasts.\n"
        "Orange: unadjusted 95% intervals for secondary warm-up contrasts.",
        transform=ax.transAxes,
        fontsize=9,
    )
    save(fig, "fresh-confirmation-contrasts")

if (ROOT / "tuned-baseline-results.json").exists():
    result = json.loads((ROOT / "tuned-baseline-results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), layout="constrained")
    names = list(result["means"])
    labels = ["Active, no warm-up", "Active, warm-up 200", f"Fixed expert, ridge {result['selected_ridge']:g}"]
    for i, name in enumerate(names):
        stat = result["means"][name]
        axes[0].errorbar(
            stat["mean"] * 1e3,
            i,
            xerr=np.array([[stat["mean"] - stat["lower"]], [stat["upper"] - stat["mean"]]]) * 1e3,
            fmt="o",
            capsize=4,
            color="#0066aa" if i < 2 else "#697581",
        )
    axes[0].set(
        yticks=range(3),
        yticklabels=labels,
        xlabel="Mean transfer SubOpt x 1,000",
        title="Pointwise 95% seed-level intervals",
    )
    for i, stat in enumerate(result["comparisons"].values()):
        ci = stat["simultaneous95_for_two_primary"]
        axes[1].errorbar(
            ci["mean"] * 1e3,
            i,
            xerr=np.array([[ci["mean"] - ci["lower"]], [ci["upper"] - ci["mean"]]]) * 1e3,
            fmt="o",
            capsize=4,
            color="#0066aa",
        )
    axes[1].axvline(0, color="0.6", linewidth=1)
    axes[1].set(
        yticks=range(2),
        yticklabels=["No-warm Active - expert", "Warm Active - expert"],
        xlabel="Paired difference x 1,000",
        title="Simultaneous 95% intervals",
    )
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle("Independent selected-baseline check: 50 new seeds, 50 new tasks", fontsize=12)
    fig.supxlabel("Neither comparison passes the adjusted threshold (Holm p = .185 and .055).", fontsize=9)
    save(fig, "independent-tuned-baseline")

if (ROOT / "historical-strict-curves.npz").exists():
    from scipy.stats import t as student_t

    original = np.load("reproducibility/data/lobeke/final_evaluation_unit_reward_v2/data.npz")
    corrected = np.load(ROOT / "historical-strict-curves.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
    for ax, name, label in zip(
        axes,
        ("fixed_expert_mle", "online_random_no_warm"),
        ("Fixed-expert MLE", "Online random, no warm-up"),
        strict=True,
    ):
        key = f"{name}__taskwise_transfer_suboptimality"
        values = corrected[key].mean(axis=0)
        mean = values.mean(axis=0)
        half = student_t.ppf(0.975, 49) * values.std(axis=0, ddof=1) / np.sqrt(50)
        x = original[f"{name}__checkpoints"]
        ax.plot(x, original[key].mean(axis=(0, 1)), "--", color="#bd7300", label="Retained fast evaluation")
        ax.plot(x, mean, color="#0066aa", label="Full multistart repair")
        ax.fill_between(x, mean - half, mean + half, color="#0066aa", alpha=0.12)
        ax.set(title=label, xlabel="Follower observations", ylabel="Mean transfer SubOpt")
    axes[0].legend(fontsize=8)
    fig.supxlabel("Same 50 seeds and ten tasks. Shading: pointwise 95% seed-level intervals.", fontsize=9)
    save(fig, "historical-transfer-curves")
