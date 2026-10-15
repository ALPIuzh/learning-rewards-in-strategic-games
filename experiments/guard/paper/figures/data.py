"""Shared run loading, protocol names, and transfer-task data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.rewards import (
    is_unit_interval_reward_metadata,
    suboptimality_in_unit_reward,
    unit_interval_leader_reward,
)
from mairl.core.problem import FloatArray
from mairl_exp.evaluation.statistics import mean_confidence_interval
from mairl_exp.execution.artifacts import sha256_file

MethodKey = tuple[str, bool]

METHOD_SLUGS: dict[MethodKey, str] = {
    ("active", True): "active_warm",
    ("active", False): "active_no_warm",
    ("fixed_random", True): "fixed_random_warm",
    ("fixed_random", False): "fixed_random_no_warm",
    ("random_each_round", True): "online_random_warm",
    ("random_each_round", False): "online_random_no_warm",
    ("fixed_expert_mle", False): "fixed_expert_mle",
    ("zero_sum_ols", False): "zero_sum_ols",
    ("follower_ols", False): "follower_ols",
}


def run_source_suboptimality_is_unit_reward(run_dir: Path) -> bool:
    """Read the per-job marker that distinguishes historical and unit-reward metrics."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    markers: list[dict[str, object] | None] = []
    for entry in manifest["expected_jobs"]:
        summary = json.loads((run_dir / "jobs" / entry["job_id"] / "summary.json").read_text())
        marker = summary.get("leader_reward_normalization")
        if marker is not None and not isinstance(marker, dict):
            raise ValueError("invalid leader-reward metric provenance in job summary")
        markers.append(marker)
    present = [marker is not None for marker in markers]
    if any(present) and not all(present):
        raise ValueError("run mixes historical and unit-reward source metrics")
    if not any(present):
        return False
    if any(not is_unit_interval_reward_metadata(marker) for marker in markers if marker is not None):
        raise ValueError("run contains an unsupported leader-reward metric normalization")
    return True


METRIC_LABELS: dict[str, str] = {
    "stackelberg_suboptimality": "Target SubOpt",
    "fisher_transfer_error": "Target Fisher error",
    "coefficient_l2_error": r"$\|\hat\omega - \omega^\star\|_2$",
}


@dataclass(frozen=True, slots=True)
class SeedTrace:
    seed: int
    checkpoints: FloatArray
    metrics: dict[str, FloatArray]
    omega_history: FloatArray


def source_suboptimality_values_in_unit_reward(traces: list[SeedTrace],
        *, raw_reward_range: float, already_unit_reward: bool) -> FloatArray:  # fmt: skip
    """Return a seed-by-checkpoint matrix in canonical source-reward units."""

    values = np.asarray([trace.metrics["stackelberg_suboptimality"] for trace in traces], dtype=np.float64)
    return suboptimality_in_unit_reward(
        values, raw_reward_range=raw_reward_range, already_unit_reward=already_unit_reward
    )


def source_suboptimality_in_unit_reward(traces: dict[MethodKey, list[SeedTrace]],
        *, raw_reward_range: float, already_unit_reward: bool = False) -> dict[MethodKey, list[SeedTrace]]:  # fmt: skip
    """Copy traces while expressing source SubOpt in canonical reward units."""

    result: dict[MethodKey, list[SeedTrace]] = {}
    for key, method_traces in traces.items():
        normalized = source_suboptimality_values_in_unit_reward(
            method_traces, raw_reward_range=raw_reward_range, already_unit_reward=already_unit_reward
        )
        result[key] = []
        for index, trace in enumerate(method_traces):
            metrics = {name: values.copy() for name, values in trace.metrics.items()}
            metrics["stackelberg_suboptimality"] = normalized[index]
            result[key].append(
                SeedTrace(
                    seed=trace.seed,
                    checkpoints=trace.checkpoints.copy(),
                    metrics=metrics,
                    omega_history=trace.omega_history.copy(),
                )
            )
    return result


def load_run_traces(run_dir: Path) -> dict[MethodKey, list[SeedTrace]]:
    """Load every completed job's checkpoints/metrics, verified against the manifest."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"run at {run_dir} is not complete (status={manifest['status']!r})")

    traces: dict[MethodKey, list[SeedTrace]] = {}
    for entry in manifest["expected_jobs"]:
        job_dir = run_dir / "jobs" / entry["job_id"]
        status = json.loads((job_dir / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError(f"job {entry['job_id']} is not complete")
        result_path = job_dir / "result.npz"
        digest = sha256_file(result_path)
        if digest != status["result_sha256"]:
            raise ValueError(f"job {entry['job_id']} result.npz failed its sha256 check")

        spec = entry["spec"]
        key: MethodKey = (spec["method"], bool(spec["warmup_enabled"]))
        with np.load(result_path) as arrays:
            checkpoints = arrays["evaluation_sample_counts"].copy()
            metrics = {
                name.removeprefix("metric__"): arrays[name].copy()
                for name in arrays.files
                if name.startswith("metric__")
            }
            # omega_history is stored at trace_sample_counts (the full per-round trace for
            # active/fixed_random/random_each_round, or already exactly evaluation_sample_counts
            # for the OLS methods) -- always re-align to the evaluation checkpoints explicitly
            # rather than assume the two coincide.
            trace_sample_counts = arrays["trace_sample_counts"]
            positions = np.searchsorted(trace_sample_counts, checkpoints)
            if not np.array_equal(trace_sample_counts[positions], checkpoints):
                raise ValueError(f"evaluation checkpoint missing from stored trace for job {entry['job_id']}")
            omega_history = arrays["omega_history"][positions].copy()
        traces.setdefault(key, []).append(
            SeedTrace(seed=spec["seed"], checkpoints=checkpoints, metrics=metrics, omega_history=omega_history)
        )

    for seed_traces in traces.values():
        seed_traces.sort(key=lambda item: item.seed)
    return traces


def infer_warmup_budget(traces: dict[MethodKey, list[SeedTrace]]) -> int:
    """Read the common warm-up budget from the warm Active-IRL checkpoints."""

    active_warmup = traces.get(("active", True))
    if not active_warmup:
        raise ValueError("cannot infer warm-up budget without warm Active-IRL traces")
    budgets = {int(trace.checkpoints[0]) for trace in active_warmup}
    if len(budgets) != 1:
        raise ValueError(f"inconsistent warm-up budgets across seeds: {sorted(budgets)}")
    return budgets.pop()


def _select_diverse(candidates: FloatArray, num_tasks: int) -> tuple[FloatArray, FloatArray]:
    """Greedily select ``num_tasks`` rows of ``candidates`` with least pairwise similarity."""

    flat = candidates.reshape(candidates.shape[0], -1)
    unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)

    selected = [0]
    for _ in range(num_tasks - 1):
        similarity = np.abs(unit[selected] @ unit.T).max(axis=0)
        similarity[selected] = np.inf
        selected.append(int(np.argmin(similarity)))

    tasks = candidates[selected]
    pairwise = unit[selected] @ unit[selected].T
    return tasks, pairwise


def draw_diverse_structured_tasks(coverage: FloatArray, *, num_tasks: int, seed: int,
        scale: float, candidate_pool: int) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Draw ``num_tasks`` leader-reward matrices with the real game's coverage form.

    Before canonical normalization, ``reward[a, b] = value[b]`` if patrol
    ``a`` covers target ``b``, else
    ``-value[b]`` -- the same functional form as the real anti-poaching reward
    (rewarding coverage of valuable targets, penalizing leaving them exposed),
    but with an independently random per-target ``value`` standing in for
    whatever a downstream task happens to prioritize (a different constituency,
    a different conservation concern) -- still unrelated to the true omega or
    elephant-density values, but not literally unstructured per-cell noise. This
    keeps the resulting optimum a genuine mixture in typical cases, so
    sub-optimality reflects estimation quality rather than which pure action an
    i.i.d.-noise draw happened to favor. Returned rewards are the positive-
    affine representatives in ``[0, 1]``, so every task has ``R_max = 1``.
    """

    rng = np.random.default_rng(seed)
    values = rng.uniform(-scale, scale, size=(candidate_pool, coverage.shape[1]))
    candidates = (2.0 * coverage[None, :, :] - 1.0) * values[:, None, :]
    selected, _ = _select_diverse(candidates, num_tasks)
    tasks = np.stack([unit_interval_leader_reward(task) for task in selected])
    flattened = tasks.reshape(num_tasks, -1)
    unit = flattened / np.linalg.norm(flattened, axis=1, keepdims=True)
    return tasks, unit @ unit.T


def mean_interval(values: FloatArray, confidence_level: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-column mean and a Student-t confidence band across seeds."""

    return mean_confidence_interval(values, confidence_level)


def load_runtime(run_dir: Path):
    """Rebuild the exact application/problem the run used, from its own stored config."""

    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    return application.build(numerics)
