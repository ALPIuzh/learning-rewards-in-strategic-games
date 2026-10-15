"""Audit the protected Lobeke arrays without mutating source artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

METHODS = (
    "active_warm",
    "active_no_warm",
    "fixed_random_warm",
    "fixed_random_no_warm",
    "online_random_warm",
    "online_random_no_warm",
    "fixed_expert_mle",
    "zero_sum_ols",
    "follower_ols",
)
METRICS = (
    "target_suboptimality",
    "target_fisher_error",
    "coefficient_l2_error",
    "averaged_transfer_suboptimality",
    "averaged_transfer_fisher_error",
)


def mean_se_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(student_t.ppf(0.975, df=values.size - 1))
    return mean, se, mean - critical * se, mean + critical * se


def print_final_summary(data: np.lib.npyio.NpzFile) -> None:
    print("FINAL PROTECTED EVALUATION")
    for metric in METRICS:
        print(f"\nmetric={metric}")
        for method in METHODS:
            values = data[f"{method}__{metric}"][:, -1]
            mean, se, lower, upper = mean_se_ci(values)
            print(
                f"  {method:24s} mean={mean:.8g} se={se:.3g} "
                f"ci95=[{lower:.8g}, {upper:.8g}] median={np.median(values):.8g}"
            )


def print_paired_difference(
    data: np.lib.npyio.NpzFile,
    left: str,
    right: str,
    metric: str,
) -> None:
    left_values = data[f"{left}__{metric}"][:, -1]
    right_values = data[f"{right}__{metric}"][:, -1]
    difference = left_values - right_values
    mean, se, lower, upper = mean_se_ci(difference)
    print(
        f"{left} - {right}: mean={mean:.8g} se={se:.3g} "
        f"ci95=[{lower:.8g}, {upper:.8g}] median={np.median(difference):.8g} "
        f"left_better={int(np.count_nonzero(difference < 0))}/{difference.size}"
    )


def print_final_pairing(data: np.lib.npyio.NpzFile) -> None:
    print("\nPAIRED FINAL DIFFERENCES")
    pairs = (
        ("active_warm", "active_no_warm"),
        ("fixed_random_warm", "fixed_random_no_warm"),
        ("online_random_warm", "online_random_no_warm"),
        ("zero_sum_ols", "active_warm"),
        ("zero_sum_ols", "active_no_warm"),
        ("zero_sum_ols", "fixed_expert_mle"),
        ("zero_sum_ols", "follower_ols"),
    )
    for metric in METRICS:
        print(f"\nmetric={metric}")
        for left, right in pairs:
            print_paired_difference(data, left, right, metric)


def print_checkpoints_and_omega(data: np.lib.npyio.NpzFile) -> None:
    print("\nCHECKPOINTS AND FINAL OMEGA")
    for method in METHODS:
        checkpoints = data[f"{method}__checkpoints"]
        omega = data[f"{method}__final_omega"]
        print(
            f"{method:24s} checkpoints={checkpoints.tolist()} "
            f"omega_mean={np.mean(omega, axis=0).round(7).tolist()} "
            f"omega_sd={np.std(omega, axis=0, ddof=1).round(7).tolist()}"
        )


def print_warmup_grid(grid: np.lib.npyio.NpzFile) -> None:
    budgets = grid["warmup_budgets"]
    multipliers = grid["lambda_information_multipliers"]
    scores = grid["task_scores"].mean(axis=-1)  # budget, lambda, seed
    print("\nTRANSFER GRID: PAIRED BUDGET EFFECTS AT FIXED LAMBDA")
    for lambda_index, multiplier in enumerate(multipliers):
        baseline = scores[budgets == 0, lambda_index, :][0]
        comparisons = []
        for budget_index, budget in enumerate(budgets):
            difference = scores[budget_index, lambda_index] - baseline
            mean, se, lower, upper = mean_se_ci(difference)
            comparisons.append(f"K0={int(budget)} diff={mean:+.3g} ci=[{lower:+.3g},{upper:+.3g}]")
        print(f"lambda/eta^2={multiplier:g}: " + "; ".join(comparisons))

    def cell(budget: int, multiplier: float) -> np.ndarray:
        budget_index = int(np.flatnonzero(budgets == budget)[0])
        multiplier_index = int(np.flatnonzero(multipliers == multiplier)[0])
        return scores[budget_index, multiplier_index]

    print("\nTRANSFER GRID: PREDECLARED COMPARISONS")
    comparisons = (
        ((200, 12.0), (0, 3.0), "final warm setting - final no-warm setting"),
        ((200, 12.0), (0, 12.0), "warm-up effect at lambda=12"),
        ((200, 3.0), (0, 3.0), "warm-up effect at lambda=3"),
        ((100, 1.0), (0, 1.0), "theorem reference warm-up effect at lambda=1"),
    )
    for left, right, label in comparisons:
        difference = cell(*left) - cell(*right)
        mean, se, lower, upper = mean_se_ci(difference)
        print(
            f"{label}: mean={mean:+.8g} se={se:.3g} "
            f"ci95=[{lower:+.8g}, {upper:+.8g}] "
            f"left_better={int(np.count_nonzero(difference < 0))}/{difference.size}"
        )

    mean_surface = scores.mean(axis=-1)
    best = np.unravel_index(int(np.argmin(mean_surface)), mean_surface.shape)
    print(
        "grid minimum: "
        f"K0={int(budgets[best[0]])}, lambda/eta^2={multipliers[best[1]]:g}, "
        f"mean={mean_surface[best]:.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        type=Path,
        default=Path("reproducibility/data/lobeke/final_evaluation/data.npz"),
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("reproducibility/data/lobeke/tuning_grid.npz"),
    )
    args = parser.parse_args()
    with np.load(args.final, allow_pickle=False) as final:
        print_final_summary(final)
        print_final_pairing(final)
        print_checkpoints_and_omega(final)
    with np.load(args.grid, allow_pickle=False) as grid:
        print_warmup_grid(grid)


if __name__ == "__main__":
    main()
