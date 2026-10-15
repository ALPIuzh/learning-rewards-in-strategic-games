"""Explain small-budget source regret through local task sensitivity and Fisher geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.guard.paper.figures.downstream import load_runtime
from mairl.core.math import cumulative_fisher_information, fisher_information
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    leader_value_and_gradient,
    optimize_leader_policy_batch,
)
from mairl_exp.artifacts import (
    atomic_write_json,
    collect_git_provenance,
    find_repository,
)


def _source_regret_function(runtime: Any, leader_config: LeaderOptimizationConfig):
    leader_reward = np.asarray(runtime.metadata["defender_reward"], dtype=np.float64)
    true_solution = optimize_leader_policy_batch(
        runtime.problem,
        runtime.true_omega[None, :],
        leader_reward,
        leader_config,
        (0,),
    )[0]
    true_value, _ = leader_value_and_gradient(
        runtime.problem,
        runtime.true_omega,
        leader_reward,
        true_solution.policy,
    )

    def regret(error: np.ndarray) -> float:
        estimated_solution = optimize_leader_policy_batch(
            runtime.problem,
            (runtime.true_omega + error)[None, :],
            leader_reward,
            leader_config,
            (1,),
        )[0]
        deployed_value, _ = leader_value_and_gradient(
            runtime.problem,
            runtime.true_omega,
            leader_reward,
            estimated_solution.policy,
        )
        raw = float(true_value - deployed_value)
        if raw < -1.0e-9:
            raise ValueError(f"source regret is unexpectedly negative: {raw:.3e}")
        return max(0.0, raw)

    return regret


def finite_difference_hessian(regret: Any, dimension: int, step: float) -> np.ndarray:
    """Return a centered finite-difference Hessian at zero."""

    basis = np.eye(dimension)
    zero = np.zeros(dimension)
    baseline = regret(zero)
    hessian = np.empty((dimension, dimension), dtype=np.float64)
    for left in range(dimension):
        hessian[left, left] = (
            regret(step * basis[left])
            + regret(-step * basis[left])
            - 2.0 * baseline
        ) / step**2
        for right in range(left):
            value = (
                regret(step * (basis[left] + basis[right]))
                - regret(step * (basis[left] - basis[right]))
                - regret(step * (-basis[left] + basis[right]))
                + regret(-step * (basis[left] + basis[right]))
            ) / (4.0 * step**2)
            hessian[left, right] = value
            hessian[right, left] = value
    return hessian


def _entry_geometry(
    runtime: Any,
    path: Path,
    sensitive_direction: np.ndarray,
    fixed_weak_direction: np.ndarray,
) -> dict[str, Any]:
    with np.load(path) as arrays:
        policies = arrays["policies"].copy()
        estimates = arrays["estimates"].copy()
    information = np.stack(
        [
            cumulative_fisher_information(runtime.problem, seed_policies, runtime.true_omega)
            for seed_policies in policies
        ]
    )
    eigenvalues = np.linalg.eigvalsh(information)
    errors = estimates - runtime.true_omega[None, :]
    sensitive_errors = errors @ sensitive_direction
    weak_errors = errors @ fixed_weak_direction
    sensitive_information = np.einsum(
        "d,sde,e->s",
        sensitive_direction,
        information,
        sensitive_direction,
        optimize=True,
    )
    weak_information = np.einsum(
        "d,sde,e->s",
        fixed_weak_direction,
        information,
        fixed_weak_direction,
        optimize=True,
    )
    return {
        "path": str(path.resolve()),
        "seed_count": int(estimates.shape[0]),
        "mean_fisher_eigenvalues": np.mean(eigenvalues, axis=0).tolist(),
        "mean_sensitive_information": float(np.mean(sensitive_information)),
        "mean_fixed_weak_information": float(np.mean(weak_information)),
        "sensitive_error_bias": float(np.mean(sensitive_errors)),
        "sensitive_error_rmse": float(np.sqrt(np.mean(sensitive_errors**2))),
        "fixed_weak_error_bias": float(np.mean(weak_errors)),
        "fixed_weak_error_rmse": float(np.sqrt(np.mean(weak_errors**2))),
        "mean_coefficient_l2_error": float(np.mean(np.linalg.norm(errors, axis=1))),
    }


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite geometry audit: {output}")
    run_dir = args.reference_run.resolve()
    runtime = load_runtime(run_dir)
    numerics = instantiate(OmegaConf.load(run_dir / "resolved_config.yaml").numerics)
    leader_config = LeaderOptimizationConfig(numerics.evaluation_policy)
    hessian = finite_difference_hessian(
        _source_regret_function(runtime, leader_config),
        runtime.problem.feature_dimension,
        args.step,
    )
    hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(hessian)
    sensitive_direction = hessian_eigenvectors[:, -1]

    fixed_information = fisher_information(
        runtime.problem,
        runtime.fixed_mle_leader_policy,
        runtime.true_omega,
    )
    fixed_eigenvalues, fixed_eigenvectors = np.linalg.eigh(fixed_information)
    fixed_weak_direction = fixed_eigenvectors[:, 0]

    entries = {}
    for specification in args.entry:
        if "=" not in specification:
            raise ValueError("entries must use LABEL=PATH syntax")
        label, raw_path = specification.split("=", 1)
        if not label or label in entries:
            raise ValueError("entry labels must be nonempty and unique")
        entries[label] = _entry_geometry(
            runtime,
            Path(raw_path).resolve(),
            sensitive_direction,
            fixed_weak_direction,
        )

    payload = {
        "schema_version": 1,
        "reference_run": str(run_dir),
        "finite_difference_step": args.step,
        "source_regret_hessian": hessian.tolist(),
        "source_regret_hessian_eigenvalues": hessian_eigenvalues.tolist(),
        "source_sensitive_direction": sensitive_direction.tolist(),
        "fixed_expert_per_sample_fisher_eigenvalues": fixed_eigenvalues.tolist(),
        "fixed_expert_weak_direction": fixed_weak_direction.tolist(),
        "absolute_sensitive_weak_inner_product": float(
            abs(sensitive_direction @ fixed_weak_direction)
        ),
        "entries": entries,
        "evaluation_git": collect_git_provenance(find_repository(Path.cwd())).manifest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=1.0e-2)
    parser.add_argument("--entry", action="append", required=True)
    args = parser.parse_args()
    if args.step <= 0.0:
        parser.error("step must be strictly positive")
    run(args)


if __name__ == "__main__":
    main()
