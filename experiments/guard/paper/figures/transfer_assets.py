"""Generate title-free policy assets and task-specific transfer summaries.

The output is deliberately layout-agnostic.  LaTeX decides which policies are
large, which baselines are subordinate, and where method names and numerical
results appear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from experiments.guard.geographic.map_lobeke import _grid_array, build_game, home_base_position
from experiments.guard.paper.application import _validated_stackelberg_suboptimality
from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_figure_run_traces
from experiments.guard.paper.figures.data import MethodKey, SeedTrace, load_runtime, mean_interval
from experiments.guard.paper.figures.style import (
    ACTIVE,
    EXPERT_MLE,
    FIXED_RANDOM,
    FOLLOWER_OLS,
    GRID,
    INK,
    LIAO,
    RANDOM_ROUND,
    paper_style,
)
from experiments.guard.paper.figures.task_sets import (
    ILLUSTRATION_BEHAVIOR_QUANTILES,
    TRANSFER_EVALUATION_TASKS,
    BehavioralTaskSelection,
    StructuredTaskSet,
    select_behaviorally_stratified_tasks,
)
from mairl.core.math import follower_probabilities
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)
from mairl_exp.config import RESULTS_DIR as PAPER_RESULTS_DIR
from mairl_exp.evaluation.mpl import LinearSegmentedColormap, plt

RESULTS_DIR = PAPER_RESULTS_DIR / "figure2_assets_behavioral_v2"

METHODS: dict[tuple[str, bool], tuple[str, str, str]] = {
    ("active", True): ("active", "Active IRL", ACTIVE),
    ("active", False): ("active_nowarm", "Active IRL without warm-up", ACTIVE),
    ("fixed_expert_mle", False): ("expert", "Fixed-expert MLE", EXPERT_MLE),
    ("zero_sum_ols", False): ("liao", "Zero-sum OLS", LIAO),
    ("follower_ols", False): ("follower", "Follower-only OLS", FOLLOWER_OLS),
    ("fixed_random", True): ("fixed", "Fixed random", FIXED_RANDOM),
    ("fixed_random", False): ("fixed_nowarm", "Fixed random without warm-up", FIXED_RANDOM),
    ("random_each_round", True): ("random", "Random each round", RANDOM_ROUND),
    ("random_each_round", False): ("random_nowarm", "Random each round without warm-up", RANDOM_ROUND),
}
RewardEvaluation = tuple[np.ndarray, dict[tuple[str, bool], np.ndarray], dict[tuple[str, bool], dict[str, float]]]
MethodSummaries = dict[str, dict[str, object]]


def _coverage_grid(gsg, target_nodes, coverage: np.ndarray) -> np.ndarray:
    return _grid_array(gsg, dict(zip(target_nodes, coverage, strict=True)))


def _save_policy_asset(path: Path, coverage_grid: np.ndarray, color: str) -> None:
    cmap = LinearSegmentedColormap.from_list("policy", ["#FAFBFC", "#DCE8EF", color])
    with paper_style():
        figure, axis = plt.subplots(figsize=(1.0, 1.0))
        axis.imshow(
            coverage_grid, origin="upper", extent=(0, 1, 0, 1), cmap=cmap, vmin=0.0, vmax=0.85, interpolation="nearest"
        )
        axis.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
        for boundary in np.linspace(0, 1, 6):
            axis.axhline(boundary, color="white", linewidth=0.5)
            axis.axvline(boundary, color="white", linewidth=0.5)
        base_x, base_y = home_base_position()
        axis.scatter([base_x], [base_y], marker="^", s=12, color=INK, edgecolor="white", linewidth=0.35, zorder=3)
        for spine in axis.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.55)
        figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(figure)


def _save_source_asset(path: Path, coverage_grid: np.ndarray, target_nodes: list[object],
        attack_probability: np.ndarray, gsg) -> None:  # fmt: skip
    """Show the fixed expert patrol and the poacher actions OLS observes."""

    cmap = LinearSegmentedColormap.from_list("source_policy", ["#FAFBFC", "#DCE8EF", ACTIVE])
    with paper_style():
        figure, axis = plt.subplots(figsize=(1.0, 1.0))
        axis.imshow(
            coverage_grid,
            origin="upper",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.25, float(np.max(coverage_grid))),
            interpolation="nearest",
        )
        positions = np.asarray(
            [
                (
                    (gsg.graph.nodes[node]["position"][1] + 0.5) / gsg.num_columns,
                    1.0 - (gsg.graph.nodes[node]["position"][0] + 0.5) / gsg.num_rows,
                )
                for node in target_nodes
            ]
        )
        relative = attack_probability / max(float(np.max(attack_probability)), 1e-12)
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=3.0 + 18.0 * relative**1.4,
            color=FOLLOWER_OLS,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        axis.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
        for boundary in np.linspace(0, 1, 6):
            axis.axhline(boundary, color="white", linewidth=0.5)
            axis.axvline(boundary, color="white", linewidth=0.5)
        base_x, base_y = home_base_position()
        axis.scatter([base_x], [base_y], marker="^", s=12, color=INK, edgecolor="white", linewidth=0.35, zorder=4)
        for spine in axis.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.55)
        figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(figure)


def _format_tex_estimate(mean: float, half_width: float) -> str:
    if mean == 0.0:
        return r"0"
    exponent = int(np.floor(np.log10(abs(mean))))
    scale = 10.0**exponent
    estimate = rf"{mean / scale:.2f}\mathbin{{\pm}}{half_width / scale:.2f}"
    return rf"({estimate})\!\times\!10^{{{exponent}}}"


def _write_tex_values(summary: dict[str, object], path: Path) -> None:
    lines = ["% Generated by experiments.guard.paper.figures.transfer_assets; do not edit."]
    task_words = ("One", "Two", "Three", "Four", "Five")
    methods = summary["methods"]
    assert isinstance(methods, dict)
    for slug, method_summary in methods.items():
        assert isinstance(method_summary, dict)
        task_results = method_summary["tasks"]
        assert isinstance(task_results, list)
        if len(task_results) > len(task_words):
            raise ValueError(f"TeX export supports at most {len(task_words)} downstream tasks")
        macro_slug = str(slug).capitalize()
        for task_index, result in enumerate(task_results, start=1):
            assert isinstance(result, dict)
            value = _format_tex_estimate(float(result["mean"]), float(result["ci_half_width"]))
            task_word = task_words[task_index - 1]
            lines.append(rf"\newcommand{{\Subopt{macro_slug}{task_word}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


def _suboptimality_summary(regrets: np.ndarray) -> dict[str, float]:
    regrets = np.asarray(regrets, dtype=np.float64)
    if regrets.ndim != 1 or regrets.size < 2:
        raise ValueError("suboptimality summaries require at least two scalar regrets")
    if not np.all(np.isfinite(regrets)):
        raise ValueError("suboptimality regrets must all be finite")
    if np.any(regrets < 0.0):
        raise ValueError("validated suboptimality regrets must all be nonnegative")
    mean, lower, upper = mean_interval(regrets[:, None], confidence_level=0.95)
    half_width = max(float(mean[0] - lower[0]), float(upper[0] - mean[0]))
    return {
        "mean": float(mean[0]),
        "ci_lower": float(lower[0]),
        "ci_upper": float(upper[0]),
        "ci_half_width": half_width,
    }


def _evaluate_reward(runtime, reward: np.ndarray, *, coverage_indicator: np.ndarray, gsg, target_nodes: list[object],
        leader_config: LeaderOptimizationConfig, method_estimates: dict[tuple[str, bool], np.ndarray],
        method_seeds: dict[tuple[str, bool], tuple[int, ...]], oracle_seed: int) -> RewardEvaluation:  # fmt: skip
    """Return the oracle grid, mean deployed grids, and task-specific suboptimalities."""

    oracle = optimize_leader_policy(runtime.problem, runtime.true_omega, reward, leader_config, seed=oracle_seed)
    oracle_grid = _coverage_grid(gsg, target_nodes, coverage_indicator.T @ oracle.policy)
    method_grids: dict[tuple[str, bool], np.ndarray] = {}
    method_results: dict[tuple[str, bool], dict[str, float]] = {}
    for key in METHODS:
        solutions = optimize_leader_policy_batch(
            runtime.problem, method_estimates[key], reward, leader_config, method_seeds[key]
        )
        policies = np.asarray([solution.policy for solution in solutions])
        mean_coverage = coverage_indicator.T @ policies.mean(axis=0)
        method_grids[key] = _coverage_grid(gsg, target_nodes, mean_coverage)

        # NaN initialization makes an accidentally skipped assignment fail closed
        # in _suboptimality_summary instead of becoming a plausible-looking badge.
        regrets = np.full(len(solutions), np.nan, dtype=np.float64)
        for seed_index, solution in enumerate(solutions):
            deployed_value, _ = leader_value_and_gradient(runtime.problem, runtime.true_omega, reward, solution.policy)
            regrets[seed_index] = _validated_stackelberg_suboptimality(oracle.value, np.asarray([deployed_value]))[0]
        method_results[key] = _suboptimality_summary(regrets)
    return oracle_grid, method_grids, method_results


def _final_method_inputs(traces: dict[MethodKey, list[SeedTrace]],
) -> tuple[dict[MethodKey, np.ndarray], dict[MethodKey, tuple[int, ...]], MethodSummaries, int]:  # fmt: skip
    """Collect final estimates, paired seeds, and empty summary records."""

    estimates: dict[MethodKey, np.ndarray] = {}
    seeds: dict[MethodKey, tuple[int, ...]] = {}
    summaries: MethodSummaries = {}
    shared_sample_count: int | None = None
    for key in METHODS:
        seed_traces = traces[key]
        final_counts = {int(trace.checkpoints[-1]) for trace in seed_traces}
        if len(final_counts) != 1:
            raise ValueError(f"inconsistent final sample counts for {key}: {final_counts}")
        sample_count = final_counts.pop()
        if shared_sample_count not in (None, sample_count):
            raise ValueError("methods do not share a common final sample count")
        shared_sample_count = sample_count
        estimates[key] = np.asarray([trace.omega_history[-1] for trace in seed_traces])
        seeds[key] = tuple(trace.seed for trace in seed_traces)
        slug, label, _ = METHODS[key]
        summaries[slug] = {"label": label, "tasks": []}
    assert shared_sample_count is not None
    return estimates, seeds, summaries, shared_sample_count


def _selection_provenance(task_set: StructuredTaskSet, selection: BehavioralTaskSelection) -> dict[str, object]:
    return {
        "candidate_suite": {
            "name": task_set.name,
            "count": task_set.count,
            "seed": task_set.seed,
            "candidate_pool": task_set.candidate_pool,
            "scale": task_set.scale,
        },
        "criterion": (
            "nearest tasks to the 0%, 50%, and 100% quantiles of total-variation "
            "distance between source and downstream true optimal leader policies"
        ),
        "behavior_quantiles": list(ILLUSTRATION_BEHAVIOR_QUANTILES),
        "selected_suite_indices_zero_based": list(selection.suite_indices),
        "selected_suite_indices_one_based": [index + 1 for index in selection.suite_indices],
        "source_policy_total_variation": list(selection.source_policy_total_variation),
        "selected_pairwise_policy_total_variation": (selection.pairwise_policy_total_variation.tolist()),
        "suite_source_policy_total_variation": list(selection.suite_source_policy_total_variation),
        "uses_algorithm_performance": False,
    }


def generate(run_dirs: Path | Sequence[Path], *, output_dir: Path, correction_dir: Path,
        task_set: StructuredTaskSet = TRANSFER_EVALUATION_TASKS) -> Path:  # fmt: skip
    paths = (run_dirs,) if isinstance(run_dirs, Path) else tuple(Path(path) for path in run_dirs)
    if not paths:
        raise ValueError("at least one Lobeke source run is required")
    traces = load_figure_run_traces(paths, correction_dir)
    run_manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    correction_provenance = json.loads((correction_dir / "provenance.json").read_text())
    correction_applied = len(paths) == 1 and (
        correction_provenance.get("source", {}).get("run_id") == run_manifests[0].get("run_id")
    )
    runtime = load_runtime(paths[0])
    gsg = build_game()
    target_nodes = gsg.schedule_form_dict["targets"]
    coverage_indicator = np.asarray(runtime.metadata["coverage_indicator"], dtype=np.float64)
    leader_config = LeaderOptimizationConfig(
        PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=2)
    )
    selection = select_behaviorally_stratified_tasks(runtime, task_set, leader_config)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing transfer assets: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_policy = np.asarray(runtime.fixed_mle_leader_policy, dtype=np.float64)
    source_follower = follower_probabilities(runtime.problem, source_policy, runtime.true_omega)
    source_grid = _coverage_grid(gsg, target_nodes, coverage_indicator.T @ source_policy)
    _save_source_asset(output_dir / "source.pdf", source_grid, target_nodes, source_follower, gsg)
    policy_grids: dict[str, np.ndarray] = {"source": source_grid}

    method_estimates, method_seeds, method_summaries, sample_count = _final_method_inputs(traces)

    summary: dict[str, object] = {
        "schema_version": 4,
        "source": {
            "run_ids": [manifest["run_id"] for manifest in run_manifests],
            "manifests": [
                {
                    "run_id": manifest["run_id"],
                    "sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                }
                for path, manifest in zip(paths, run_manifests, strict=True)
            ],
            "paired_ols_correction_applied": correction_applied,
            "paired_ols_correction_data_sha256": (correction_provenance["data_sha256"] if correction_applied else None),
            "fixed_mle_expert_definition": runtime.fixed_mle_expert_definition,
        },
        "sample_count": sample_count,
        "leader_reward_normalization": "unit interval per task; R_max = 1",
        "task_selection": _selection_provenance(task_set, selection),
        "methods": method_summaries,
    }

    reference_reward = runtime.metadata.get("defender_reward")
    if reference_reward is None:
        raise ValueError("reference-task defender_reward missing from application metadata")
    reference_oracle, reference_grids, reference_results = _evaluate_reward(
        runtime,
        np.asarray(reference_reward, dtype=np.float64),
        coverage_indicator=coverage_indicator,
        gsg=gsg,
        target_nodes=target_nodes,
        leader_config=leader_config,
        method_estimates=method_estimates,
        method_seeds=method_seeds,
        oracle_seed=task_set.seed,
    )
    policy_grids["reference_oracle"] = reference_oracle
    _save_policy_asset(output_dir / "reference-oracle.pdf", reference_oracle, INK)
    for key, (slug, _label, color) in METHODS.items():
        policy_grids[f"reference_{slug}"] = reference_grids[key]
        _save_policy_asset(output_dir / f"reference-{slug}.pdf", reference_grids[key], color)
        method_summary = method_summaries[slug]
        assert isinstance(method_summary, dict)
        method_summary["reference"] = reference_results[key]

    for task_index, (suite_index, reward) in enumerate(
        zip(selection.suite_indices, selection.tasks, strict=True), start=1
    ):
        oracle_grid, task_grids, task_results_by_method = _evaluate_reward(
            runtime,
            reward,
            coverage_indicator=coverage_indicator,
            gsg=gsg,
            target_nodes=target_nodes,
            leader_config=leader_config,
            method_estimates=method_estimates,
            method_seeds=method_seeds,
            oracle_seed=task_set.seed + suite_index,
        )
        policy_grids[f"task{task_index}_oracle"] = oracle_grid
        _save_policy_asset(output_dir / f"task{task_index}-oracle.pdf", oracle_grid, INK)

        for key, (slug, _label, color) in METHODS.items():
            mean_grid = task_grids[key]
            policy_grids[f"task{task_index}_{slug}"] = mean_grid
            _save_policy_asset(output_dir / f"task{task_index}-{slug}.pdf", mean_grid, color)

            method_summary = method_summaries[slug]
            assert isinstance(method_summary, dict)
            task_results = method_summary["tasks"]
            assert isinstance(task_results, list)
            task_results.append(
                {
                    "task": task_index,
                    "suite_index_zero_based": suite_index,
                    "source_policy_total_variation": (selection.source_policy_total_variation[task_index - 1]),
                    **task_results_by_method[key],
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output_dir / "policy_grids.npz", **policy_grids)
    _write_tex_values(summary, output_dir / "values.tex")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, action="append")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    args = parser.parse_args()
    summary_path = generate(args.run_dir, output_dir=args.output_dir, correction_dir=args.correction_dir)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
