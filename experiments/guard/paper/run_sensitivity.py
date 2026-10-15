"""Run resumable warm-up and regularization sensitivity experiments.

The caller chooses seeds and downstream tasks, including whether they are fresh.
Each completed candidate cell is saved and verified on resumption.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from experiments.guard.paper.figures.data import SeedTrace
from experiments.guard.paper.figures.task_sets import StructuredTaskSet, normalized_structured_tasks
from experiments.guard.paper.figures.transfer_diagnostics import task_transfer_metrics
from experiments.guard.paper.study import LobekeComparisonJob, LobekeComparisonStudy
from mairl.planning.leader import LeaderOptimizationConfig
from mairl_exp.execution.artifacts import atomic_write_json, atomic_write_npz, sha256_file


@dataclass(frozen=True, slots=True)
class Protocol:
    slug: str
    method: str
    warmup_enabled: bool
    warmup_budget: int
    lambda_information_multiplier: float


def _multiplier_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _protocols(warmup_budgets: tuple[int, ...], lambda_multipliers: tuple[float, ...]) -> tuple[Protocol, ...]:
    active = tuple(
        Protocol(
            f"active_{'warm' + str(warmup_budget) if warmup_budget else 'no_warm'}"
            f"_lambda{_multiplier_slug(multiplier)}",
            "active",
            warmup_budget > 0,
            warmup_budget,
            multiplier,
        )
        for warmup_budget in warmup_budgets
        for multiplier in lambda_multipliers
    )
    return (*active, Protocol("fixed_expert_matched_ridge", "fixed_expert_mle", False, 0, 1.0))


def _protocols_from_pairs(pairs: tuple[tuple[int, float], ...]) -> tuple[Protocol, ...]:
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("candidate pairs must be nonempty and unique")
    active = tuple(
        Protocol(
            f"active_{'warm' + str(budget) if budget else 'no_warm'}_lambda{_multiplier_slug(multiplier)}",
            "active",
            budget > 0,
            budget,
            multiplier,
        )
        for budget, multiplier in pairs
    )
    return (*active, Protocol("fixed_expert_matched_ridge", "fixed_expert_mle", False, 0, 1.0))


@dataclass(frozen=True, slots=True)
class PilotDesign:
    seeds: tuple[int, ...]
    tasks: StructuredTaskSet
    protocols: tuple[Protocol, ...]
    warmup_budgets: tuple[int, ...]
    lambda_multipliers: tuple[float, ...]
    mle_ridge: float

    def __post_init__(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("pilot seeds must contain at least two unique values")
        if not self.warmup_budgets or any(value < 0 for value in self.warmup_budgets):
            raise ValueError("warm-up budgets must be nonnegative")
        if not self.lambda_multipliers or any(
            not np.isfinite(value) or value <= 0 for value in self.lambda_multipliers
        ):
            raise ValueError("lambda multipliers must be finite and strictly positive")
        if not np.isfinite(self.mle_ridge) or self.mle_ridge < 0:
            raise ValueError("MLE ridge must be finite and nonnegative")
        active_definitions = {
            (protocol.warmup_budget, protocol.lambda_information_multiplier)
            for protocol in self.protocols
            if protocol.method == "active"
        }
        if not active_definitions:
            raise ValueError("pilot design must contain at least one active protocol")
        if active_definitions - set(
            (budget, multiplier) for budget in self.warmup_budgets for multiplier in self.lambda_multipliers
        ):
            raise ValueError("protocol definitions are inconsistent with the pilot grid")


def _study(base: LobekeComparisonStudy, protocol: Protocol, total_samples: int,
        seeds: tuple[int, ...], mle_ridge: float) -> LobekeComparisonStudy:  # fmt: skip
    return replace(
        base,
        name=f"active-expert-warmup-audit-{protocol.slug}",
        seed_start=seeds[0],
        seed_count=len(seeds),
        total_samples=total_samples,
        warmup_budget=protocol.warmup_budget,
        warmup_lambda_information_multiplier=protocol.lambda_information_multiplier,
        no_warmup_lambda_information_multiplier=protocol.lambda_information_multiplier,
        fixed_expert_mle_ridge=mle_ridge,
        methods=(protocol.method,),
        warmup_modes=(protocol.warmup_enabled,),
    )


def _save_protocol(path: Path, protocol: Protocol, seeds: tuple[int, ...],
        mle_ridge: float, results: tuple[object, ...]) -> None:  # fmt: skip
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int64),
        "protocol": np.asarray(protocol.slug),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "mle_ridge": np.asarray(mle_ridge, dtype=np.float64),
    }
    names = sorted(results[0].arrays)
    if any(sorted(result.arrays) != names for result in results):
        raise ValueError("protocol results do not share an array schema")
    for name in names:
        arrays[name] = np.stack([result.arrays[name] for result in results])
    atomic_write_npz(path, arrays)


def _load_protocol(path: Path, protocol: Protocol, seeds: tuple[int, ...],
        mle_ridge: float, total_samples: int) -> dict[str, np.ndarray]:  # fmt: skip
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    schema_version = int(arrays.pop("schema_version").item())
    if schema_version not in (1, 2):
        raise ValueError(f"unsupported protocol artifact schema: {path}")
    if str(arrays.pop("protocol").item()) != protocol.slug:
        raise ValueError(f"protocol artifact identity mismatch: {path}")
    stored_ridge = 1.0e-4 if schema_version == 1 else float(arrays.pop("mle_ridge").item())
    if stored_ridge != mle_ridge:
        raise ValueError(f"protocol artifact MLE-ridge mismatch: {path}")
    if not np.array_equal(arrays["seeds"], seeds):
        raise ValueError(f"protocol artifact seed mismatch: {path}")
    if arrays["omega_history"].shape[0] != len(seeds):
        raise ValueError(f"protocol artifact result count mismatch: {path}")
    sample_counts = np.asarray(arrays["trace_sample_counts"], dtype=np.int64)
    if sample_counts.shape[0] != len(seeds) or not np.all(sample_counts[:, -1] == total_samples):
        raise ValueError(f"protocol artifact sample-budget mismatch: {path}")
    return arrays


def _run_protocols(source_config: Path, output_dir: Path, total_samples: int,
        design: PilotDesign) -> tuple[object, object, LobekeComparisonStudy]:  # fmt: skip
    cfg = OmegaConf.load(source_config)
    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    numerics = replace(numerics, mle=replace(numerics.mle, ridge=design.mle_ridge))
    runtime = application.build(numerics)
    base_study = instantiate(cfg.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)

    for protocol in design.protocols:
        path = output_dir / f"{protocol.slug}.npz"
        if path.exists():
            _load_protocol(path, protocol, design.seeds, design.mle_ridge, total_samples)
            print(f"validated existing {protocol.slug}", flush=True)
            continue
        study = _study(base_study, protocol, total_samples, design.seeds, design.mle_ridge)
        executor = study.executor(runtime, numerics)
        jobs = tuple(
            LobekeComparisonJob(method=protocol.method, warmup_enabled=protocol.warmup_enabled, seed=seed)
            for seed in design.seeds
        )
        started = perf_counter()
        results = executor.run_many(jobs)
        _save_protocol(path, protocol, design.seeds, design.mle_ridge, results)
        print(f"completed {protocol.slug} in {perf_counter() - started:.1f}s", flush=True)
    return runtime, numerics, base_study


def _paired_interval(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    standard_error = float(np.std(vector, ddof=1) / np.sqrt(vector.size))
    half_width = float(student_t.ppf(0.975, df=vector.size - 1) * standard_error)
    mean = float(np.mean(vector))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def _summarize(output_dir: Path, runtime: object, numerics: object, source_config: Path,
        total_samples: int, design: PilotDesign) -> Path:  # fmt: skip
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        expected_protocols = {protocol.slug for protocol in design.protocols}
        expected_task_policy = {
            "name": design.tasks.name,
            "seed": design.tasks.seed,
            "count": design.tasks.count,
            "candidate_pool": design.tasks.candidate_pool,
            "normalized_reward_range": [0.0, 1.0],
        }
        stored_protocols = summary.get("protocols", {})
        valid = (
            summary.get("schema_version") == 1
            and summary.get("sample_count") == total_samples
            and float(summary.get("mle_ridge", np.nan)) == design.mle_ridge
            and summary.get("seeds") == list(design.seeds)
            and summary.get("task_policy") == expected_task_policy
            and summary.get("source_config_sha256", summary.get("source_run_config_sha256"))
            == sha256_file(source_config)
            and set(stored_protocols) == expected_protocols
            and all(
                stored_protocols[protocol.slug].get("artifact_sha256")
                == sha256_file(output_dir / f"{protocol.slug}.npz")
                for protocol in design.protocols
            )
        )
        if not valid:
            raise ValueError(f"existing summary does not match the requested design: {summary_path}")
        print(f"validated existing summary {summary_path}", flush=True)
        return summary_path
    tasks = normalized_structured_tasks(runtime, design.tasks)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    protocol_summaries: dict[str, object] = {}
    per_seed_transfer: dict[str, np.ndarray] = {}

    for protocol in design.protocols:
        path = output_dir / f"{protocol.slug}.npz"
        arrays = _load_protocol(path, protocol, design.seeds, design.mle_ridge, total_samples)
        checkpoints = arrays["trace_sample_counts"][0]
        if not np.all(arrays["trace_sample_counts"] == checkpoints):
            raise ValueError(f"checkpoint mismatch in {path}")
        traces = [
            SeedTrace(
                seed=int(seed),
                checkpoints=np.asarray([checkpoints[-1]], dtype=np.int64),
                metrics={},
                omega_history=arrays["omega_history"][index, -1:, :],
            )
            for index, seed in enumerate(design.seeds)
        ]
        task_suboptimality = []
        task_fisher = []
        for task_index, reward in enumerate(tasks):
            suboptimality, fisher = task_transfer_metrics(
                runtime.problem,
                runtime.true_omega,
                reward,
                leader_config,
                traces,
                task_seed=design.tasks.seed,
                task_index=task_index,
            )
            task_suboptimality.append(suboptimality[:, 0])
            task_fisher.append(fisher[:, 0])
        task_suboptimality_array = np.stack(task_suboptimality)
        transfer_per_seed = np.mean(task_suboptimality_array, axis=0)
        per_seed_transfer[protocol.slug] = transfer_per_seed
        target = arrays["metric__stackelberg_suboptimality"][:, -1]
        protocol_summaries[protocol.slug] = {
            "definition": {
                "method": protocol.method,
                "warmup_enabled": protocol.warmup_enabled,
                "warmup_budget": protocol.warmup_budget,
                "lambda_information_multiplier": protocol.lambda_information_multiplier,
                "mle_ridge": design.mle_ridge,
            },
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
            "target_suboptimality": _paired_interval(target),
            "averaged_transfer_suboptimality": _paired_interval(transfer_per_seed),
            "averaged_transfer_fisher_error": _paired_interval(np.mean(np.stack(task_fisher), axis=0)),
            "taskwise_transfer_suboptimality_mean": np.mean(task_suboptimality_array, axis=1).tolist(),
        }

    comparisons = {}
    active_slugs = {
        (protocol.warmup_budget, protocol.lambda_information_multiplier): protocol.slug
        for protocol in design.protocols
        if protocol.method == "active"
    }
    comparison_pairs = [
        (slug, active_slugs[(0, multiplier)])
        for (budget, multiplier), slug in active_slugs.items()
        if budget > 0 and (0, multiplier) in active_slugs
    ] + [(slug, "fixed_expert_matched_ridge") for slug in active_slugs.values()]
    for left, right in comparison_pairs:
        comparisons[f"{left}_minus_{right}"] = _paired_interval(per_seed_transfer[left] - per_seed_transfer[right])

    atomic_write_json(
        summary_path,
        {
            "schema_version": 1,
            "purpose": "post-freeze estimator-matched sensitivity; not model selection",
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "sample_count": total_samples,
            "mle_ridge": design.mle_ridge,
            "seeds": list(design.seeds),
            "task_policy": {
                "name": design.tasks.name,
                "seed": design.tasks.seed,
                "count": design.tasks.count,
                "candidate_pool": design.tasks.candidate_pool,
                "normalized_reward_range": [0.0, 1.0],
            },
            "protocols": protocol_summaries,
            "paired_transfer_comparisons": comparisons,
        },
    )
    return summary_path


def run(source_config: Path, output_dir: Path, total_samples: int, design: PilotDesign) -> Path:
    if total_samples < 200:
        raise ValueError("pilot requires at least 200 samples")
    if max(design.warmup_budgets) > total_samples:
        raise ValueError("warm-up budget exceeds the pilot sample count")
    runtime, numerics, _ = _run_protocols(source_config, output_dir, total_samples, design)
    return _summarize(output_dir, runtime, numerics, source_config, total_samples, design)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-config", type=Path, help="self-contained resolved YAML configuration")
    source.add_argument(
        "--source-run", type=Path, help="deprecated compatibility form; reads resolved_config.yaml from this run"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-samples", type=int, default=1_600)
    parser.add_argument("--seed-start", type=int, default=93_000)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--task-seed", type=int, default=9_300)
    parser.add_argument("--task-count", type=int, default=40)
    parser.add_argument("--task-candidate-pool", type=int, default=1_000)
    parser.add_argument("--mle-ridge", type=float, default=0.0)
    parser.add_argument("--warmup-budgets", type=int, nargs="+", default=(0, 50, 200))
    parser.add_argument(
        "--lambda-multipliers", type=float, nargs="+", default=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    )
    parser.add_argument(
        "--candidate-pairs",
        nargs="+",
        default=None,
        metavar="BUDGET:MULTIPLIER",
        help="optional sparse grid; overrides --warmup-budgets/--lambda-multipliers",
    )
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    if args.candidate_pairs is None:
        warmup_budgets = tuple(args.warmup_budgets)
        lambda_multipliers = tuple(args.lambda_multipliers)
        protocols = _protocols(warmup_budgets, lambda_multipliers)
    else:
        try:
            candidate_pairs = tuple(
                (int(raw.split(":", maxsplit=1)[0]), float(raw.split(":", maxsplit=1)[1]))
                for raw in args.candidate_pairs
            )
        except (IndexError, ValueError) as error:
            raise ValueError("candidate pairs must use BUDGET:MULTIPLIER") from error
        warmup_budgets = tuple(sorted({budget for budget, _ in candidate_pairs}))
        lambda_multipliers = tuple(sorted({multiplier for _, multiplier in candidate_pairs}))
        protocols = _protocols_from_pairs(candidate_pairs)
    tasks = StructuredTaskSet(
        name="active_expert_warmup_audit",
        count=args.task_count,
        seed=args.task_seed,
        candidate_pool=args.task_candidate_pool,
    )
    design = PilotDesign(
        seeds=seeds,
        tasks=tasks,
        protocols=protocols,
        warmup_budgets=warmup_budgets,
        lambda_multipliers=lambda_multipliers,
        mle_ridge=args.mle_ridge,
    )
    source_config = args.source_config if args.source_config is not None else args.source_run / "resolved_config.yaml"
    print(run(source_config, args.output_dir, args.total_samples, design))


if __name__ == "__main__":
    main()
