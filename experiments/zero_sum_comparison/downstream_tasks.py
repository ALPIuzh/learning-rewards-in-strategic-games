"""Evaluate downstream-task transfer of the estimators compared by this experiment.

Reuses the reward estimates already stored in a completed
``zero-sum-estimator-comparison`` run: for each method and seed, takes the final
omega_hat (at the full 200-sample budget) and asks how well it supports planning
against several leader-reward objectives that are independent of the zero-sum game
the estimate was fit on. Every task shares the same game geometry (features, eta,
radius); only the leader's payoff matrix differs, which is the object the
Stackelberg reward-transfer lemma quantifies over.
"""

from __future__ import annotations

import argparse
import io
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from experiments.zero_sum_comparison.export_plots import METHOD_LABELS
from mairl.core.math import fisher_information
from mairl.core.problem import CompiledProblem
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy
from mairl_exp.config import REPOSITORY_ROOT
from mairl_exp.evaluation.mpl import plt
from mairl_exp.evaluation.statistics import mean_confidence_interval
from mairl_exp.execution.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

_METHOD_MARKERS = {"active": "o", "follower_ols": "s", "zero_sum_ols": "^"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots/zero_sum_comparison"))
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--task-seed", type=int, default=20260731)
    parser.add_argument("--task-scale", type=float, default=1.0)
    parser.add_argument("--task-candidate-pool", type=int, default=200)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--band-alpha", type=float, default=0.08)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _draw_diverse_tasks(*, shape: tuple[int, int], num_tasks: int, seed: int,
        scale: float, candidate_pool: int) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Draw ``num_tasks`` leader-reward matrices that are pairwise as distinct as
    possible, by greedily selecting the candidate least similar to those already
    picked. Returns the tasks and their pairwise cosine similarity matrix."""

    if num_tasks < 1:
        raise ConfigurationError("num_tasks must be positive")
    if candidate_pool < num_tasks:
        raise ConfigurationError("task_candidate_pool must be at least num_tasks")
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(-scale, scale, size=(candidate_pool, *shape))
    flattened = candidates.reshape(candidate_pool, -1)
    unit = flattened / np.linalg.norm(flattened, axis=1, keepdims=True)

    selected = [0]
    while len(selected) < num_tasks:
        similarity_to_selected = np.abs(unit @ unit[selected].T).max(axis=1)
        similarity_to_selected[selected] = np.inf
        selected.append(int(np.argmin(similarity_to_selected)))

    tasks = candidates[selected]
    task_unit = unit[selected]
    pairwise_cosine = task_unit @ task_unit.T
    return tasks, pairwise_cosine


JobTrace = tuple[int, np.ndarray, np.ndarray, np.ndarray]


def _load_run_traces(run_dir: Path) -> tuple[dict[str, list[JobTrace]], list[dict[str, str]]]:
    """Load, per method, one ``(seed, checkpoints, omega_at_checkpoints,
    original_task_l1_error)`` entry per job: the reward-estimate trace at each
    evaluation checkpoint, plus the already-computed in-distribution
    ``regularized_qre_leader_l1_error`` trace, reused as-is from the stored result.
    """

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigurationError(f"comparison manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ConfigurationError("comparison run is not complete")

    traces: dict[str, list[JobTrace]] = {}
    inputs = []
    for expected in manifest["expected_jobs"]:
        result_path = run_dir / "jobs" / expected["job_id"] / "result.npz"
        status_path = result_path.with_name("status.json")
        if not result_path.exists() or not status_path.exists():
            raise ConfigurationError(f"missing comparison job {expected['label']}")
        status = json.loads(status_path.read_text())
        digest = sha256_file(result_path)
        if status.get("status") != "complete" or status.get("result_sha256") != digest:
            raise ConfigurationError(f"invalid comparison job {expected['label']}")
        with np.load(result_path, allow_pickle=False) as arrays:
            trace_sample_counts = arrays["trace_sample_counts"]
            evaluation_sample_counts = arrays["evaluation_sample_counts"].copy()
            checkpoint_positions = np.searchsorted(trace_sample_counts, evaluation_sample_counts)
            if not np.array_equal(trace_sample_counts[checkpoint_positions], evaluation_sample_counts):
                raise ConfigurationError(
                    f"evaluation checkpoints are not a subset of the stored trace for job {expected['label']}"
                )
            omega_at_checkpoints = arrays["omega_history"][checkpoint_positions].copy()
            original_task_l1_error = arrays["metric__regularized_qre_leader_l1_error"].copy()
        method = str(expected["spec"]["method"])
        seed = int(expected["spec"]["seed"])
        traces.setdefault(method, []).append(
            (seed, evaluation_sample_counts, omega_at_checkpoints, original_task_l1_error)
        )
        inputs.append({"job_id": str(expected["job_id"]), "result_sha256": digest})
    if set(traces) != set(METHOD_LABELS):
        raise ConfigurationError("comparison run does not contain all three estimators")
    for method_traces in traces.values():
        method_traces.sort(key=lambda item: item[0])
    return traces, inputs


def _mean_interval(values: np.ndarray, confidence_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and CI along axis 0; ``values`` may be 1-D (one point) or 2-D (a trace)."""

    return mean_confidence_interval(values, confidence_level)


def _evaluate_task(problem: CompiledProblem, true_omega: np.ndarray,
        leader_reward: np.ndarray, leader_config: LeaderOptimizationConfig,
        estimates: dict[str, list[tuple[int, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:  # fmt: skip
    true_solution = optimize_leader_policy(problem, true_omega, leader_reward, leader_config, seed=0)
    per_method: dict[str, dict[str, np.ndarray]] = {}
    for method, method_estimates in estimates.items():
        suboptimalities = np.empty(len(method_estimates), dtype=np.float64)
        fisher_errors = np.empty(len(method_estimates), dtype=np.float64)
        for index, (seed, estimate) in enumerate(method_estimates):
            estimated_solution = optimize_leader_policy(problem, estimate, leader_reward, leader_config, seed=seed)
            deployed_value, _ = leader_value_and_gradient(problem, true_omega, leader_reward, estimated_solution.policy)
            suboptimalities[index] = max(0.0, true_solution.value - deployed_value)

            error = true_omega - estimate
            true_policy_information = fisher_information(problem, true_solution.policy, estimate)
            estimated_policy_information = fisher_information(problem, estimated_solution.policy, estimate)
            fisher_errors[index] = np.sqrt(
                max(
                    0.0,
                    float(error @ true_policy_information @ error),
                    float(error @ estimated_policy_information @ error),
                )
            )
        per_method[method] = {"stackelberg_suboptimality": suboptimalities, "fisher_transfer_error": fisher_errors}
    return per_method


def _render(per_task_results: list[dict[str, dict[str, np.ndarray]]], *, confidence_level: float,
        dpi: int, output_path: Path) -> dict[str, object]:  # fmt: skip
    metrics = (
        ("stackelberg_suboptimality", "Deployed value loss vs. true optimum"),
        ("fisher_transfer_error", "Task Fisher-transfer error"),
    )
    num_tasks = len(per_task_results)
    figure, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 4.8))
    summary: dict[str, object] = {}
    bar_width = 0.8 / len(METHOD_LABELS)
    task_positions = np.arange(num_tasks)

    for axis, (metric, label) in zip(axes, metrics, strict=True):
        summary[metric] = []
        for method_index, (method, method_label) in enumerate(METHOD_LABELS.items()):
            means, lowers, uppers = [], [], []
            for task_result in per_task_results:
                values = task_result[method][metric]
                mean, lower, upper = _mean_interval(values, confidence_level)
                means.append(mean)
                lowers.append(mean - lower)
                uppers.append(upper - mean)
            offset = (method_index - (len(METHOD_LABELS) - 1) / 2.0) * bar_width
            axis.bar(
                task_positions + offset, means, width=bar_width, label=method_label, yerr=(lowers, uppers), capsize=3
            )
            summary[metric].append({"method": method, "task_means": means})
        axis.set(xticks=task_positions, xticklabels=[f"Task {index + 1}" for index in range(num_tasks)], ylabel=label)
        axis.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Downstream-task transfer of the final (200-sample) reward estimate\n"
        f"(mean with {confidence_level:.0%} CI, 100 seeds per method per task)",
        y=1.02,
    )
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output_path, buffer.getvalue())
    return summary


def _paired_summary(per_task_results: list[dict[str, dict[str, np.ndarray]]],
        confidence_level: float) -> list[dict[str, object]]:  # fmt: skip
    pairs = (("active", "follower_ols"), ("active", "zero_sum_ols"), ("zero_sum_ols", "follower_ols"))
    metrics = ("stackelberg_suboptimality", "fisher_transfer_error")
    results = []
    for task_index, task_result in enumerate(per_task_results):
        task_summary: dict[str, object] = {"task_index": task_index}
        for metric in metrics:
            task_summary[metric] = {}
            for first, second in pairs:
                first_values = task_result[first][metric]
                second_values = task_result[second][metric]
                differences = first_values - second_values
                standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
                critical = float(student_t.ppf(0.5 + confidence_level / 2.0, df=differences.size - 1))
                mean_difference = float(np.mean(differences))
                task_summary[metric][f"{first}_minus_{second}"] = {
                    "mean_difference": mean_difference,
                    "lower": mean_difference - critical * standard_error,
                    "upper": mean_difference + critical * standard_error,
                    "first_method_win_rate": float(np.mean(first_values < second_values)),
                }
        results.append(task_summary)
    return results


def _final_estimates(traces: dict[str, list[JobTrace]]) -> dict[str, list[tuple[int, np.ndarray]]]:
    return {
        method: [(seed, omega_at_checkpoints[-1]) for seed, _, omega_at_checkpoints, _ in method_traces]
        for method, method_traces in traces.items()
    }


def _leader_l1_trace(problem: CompiledProblem, true_omega: np.ndarray,
        leader_reward: np.ndarray, leader_config: LeaderOptimizationConfig,
        traces: dict[str, list[JobTrace]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """Per method: shared checkpoints and a (seeds, checkpoints) L1-error array
    between the leader policy planned from each omega_hat and the truly optimal
    leader policy for this task."""

    true_policy = optimize_leader_policy(problem, true_omega, leader_reward, leader_config, seed=0).policy
    per_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, method_traces in traces.items():
        checkpoints = method_traces[0][1]
        errors = np.empty((len(method_traces), checkpoints.shape[0]), dtype=np.float64)
        for row, (seed, job_checkpoints, omega_at_checkpoints, _) in enumerate(method_traces):
            if not np.array_equal(job_checkpoints, checkpoints):
                raise ConfigurationError(f"{method} uses inconsistent evaluation checkpoints")
            for column, estimate in enumerate(omega_at_checkpoints):
                estimated_policy = optimize_leader_policy(
                    problem, estimate, leader_reward, leader_config, seed=seed + column
                ).policy
                errors[row, column] = np.linalg.norm(estimated_policy - true_policy, ord=1)
        per_method[method] = (checkpoints, errors)
    return per_method


def _render_task_lines(panels: list[dict[str, object]], *, warmup_end: int, total_samples: int,
        confidence_level: float, band_alpha: float, dpi: int, output_path: Path) -> dict[str, object]:  # fmt: skip
    """One log-scale leader-policy-L1-error panel per task, laid out side by side;
    the boxed panel is the original in-distribution task."""

    # Some tasks have a leader action that is optimal regardless of the exact omega,
    # so a good estimate can recover it up to floating-point noise (~1e-16). Floor at
    # a scale below which policy recovery is unambiguously exact, rather than at the
    # numerical noise floor, so those tasks don't stretch the log axis by ~12 decades.
    floor = 1e-4
    figure, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 4.9))
    if len(panels) == 1:
        axes = [axes]
    summary: dict[str, object] = {}
    for axis, panel in zip(axes, panels, strict=True):
        panel_summary = {}
        for method, method_label in METHOD_LABELS.items():
            checkpoints, values = panel["per_method"][method]
            mean, lower, upper = _mean_interval(values, confidence_level)
            line = axis.plot(
                checkpoints,
                np.maximum(mean, floor),
                label=method_label,
                linewidth=1.8,
                marker=_METHOD_MARKERS[method],
                markersize=4.0,
            )[0]
            axis.fill_between(
                checkpoints,
                np.maximum(lower, floor),
                np.maximum(upper, floor),
                color=line.get_color(),
                alpha=band_alpha,
            )
            panel_summary[method] = {
                "checkpoints": checkpoints.tolist(),
                "mean": mean.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            }
        summary[panel["key"]] = panel_summary
        axis.set(
            xlim=(0, total_samples),
            xlabel="Follower samples",
            ylabel="Leader-policy L1 error" if axis is axes[0] else "",
            yscale="log",
            title=panel["title"],
        )
        axis.axvspan(0, warmup_end, color="black", alpha=0.035, zorder=0)
        axis.axvline(warmup_end, color="0.35", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.25)
        if panel["boxed"]:
            for spine in axis.spines.values():
                spine.set_edgecolor("0.15")
                spine.set_linewidth(2.2)
    axes[0].legend(fontsize=7.5, loc="lower left")
    figure.suptitle(
        "Leader-policy transfer across tasks (mean with "
        f"{confidence_level:.0%} CI); boxed panel is the original in-sample task",
        y=1.03,
    )
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    atomic_write_bytes(output_path, buffer.getvalue())
    return summary


def main() -> None:
    args = _parser().parse_args()
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must lie in (0, 1)")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")

    comparison_run = _resolve(args.comparison_run)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces, inputs = _load_run_traces(comparison_run)
    estimates = _final_estimates(traces)

    cfg = OmegaConf.load(comparison_run / "resolved_config.yaml")
    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    runtime = application.build(numerics)
    # The run's own `evaluation_policy` numerics (a deterministic grid solve) is tuned
    # for well-converged final estimates. Early/noisy checkpoints on arbitrary
    # downstream tasks need the more robust multistart default used elsewhere for
    # candidate policies during estimation itself.
    leader_config = LeaderOptimizationConfig()

    tasks, pairwise_cosine = _draw_diverse_tasks(
        shape=(runtime.problem.n_leader_actions, runtime.problem.n_follower_actions),
        num_tasks=args.num_tasks,
        seed=args.task_seed,
        scale=args.task_scale,
        candidate_pool=args.task_candidate_pool,
    )

    per_task_results = [
        _evaluate_task(runtime.problem, runtime.true_omega, task, leader_config, estimates) for task in tasks
    ]

    plot_path = output_dir / "downstream_tasks.png"
    plot_summary = _render(
        per_task_results, confidence_level=args.confidence_level, dpi=args.dpi, output_path=plot_path
    )
    paired_summary = _paired_summary(per_task_results, args.confidence_level)

    warmup_end = int(traces["active"][0][1][0])
    total_samples = int(max(method_traces[0][1][-1] for method_traces in traces.values()))
    original_panel = {
        "key": "original",
        "title": "Original zero-sum task",
        "boxed": True,
        "per_method": {
            method: (method_traces[0][1], np.stack([trace[3] for trace in method_traces]))
            for method, method_traces in traces.items()
        },
    }
    downstream_panels = [
        {
            "key": f"task_{index + 1}",
            "title": f"Downstream task {index + 1}",
            "boxed": False,
            "per_method": _leader_l1_trace(runtime.problem, runtime.true_omega, task, leader_config, traces),
        }
        for index, task in enumerate(tasks)
    ]
    lines_plot_path = output_dir / "downstream_tasks_leader_l1_error.png"
    lines_summary = _render_task_lines(
        [original_panel, *downstream_panels],
        warmup_end=warmup_end,
        total_samples=total_samples,
        confidence_level=args.confidence_level,
        band_alpha=args.band_alpha,
        dpi=args.dpi,
        output_path=lines_plot_path,
    )

    inputs.sort(key=lambda item: item["job_id"])
    max_pairwise_cosine = float(
        max((abs(pairwise_cosine[i, j]) for i, j in combinations(range(len(tasks)), 2)), default=0.0)
    )
    atomic_write_json(
        output_dir / "downstream_tasks_manifest.json",
        {
            "schema_version": 1,
            "comparison_run": str(comparison_run.relative_to(REPOSITORY_ROOT)),
            "comparison_manifest_sha256": sha256_file(comparison_run / "manifest.json"),
            "comparison_inputs_sha256": sha256_bytes(canonical_json(inputs).encode()),
            "task_seed": args.task_seed,
            "task_scale": args.task_scale,
            "num_tasks": args.num_tasks,
            "tasks": [task.tolist() for task in tasks],
            "task_pairwise_cosine_similarity": pairwise_cosine.tolist(),
            "max_pairwise_cosine_similarity": max_pairwise_cosine,
            "confidence_level": args.confidence_level,
            "plot_summary": plot_summary,
            "paired_summary": paired_summary,
            "plot": {"path": str(plot_path.relative_to(REPOSITORY_ROOT)), "sha256": sha256_file(plot_path)},
            "leader_l1_error_plot": {
                "path": str(lines_plot_path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(lines_plot_path),
                "summary": lines_summary,
            },
        },
    )
    print(f"plot={plot_path}")
    print(f"plot={lines_plot_path}")
    print(f"manifest={output_dir / 'downstream_tasks_manifest.json'}")


if __name__ == "__main__":
    main()
