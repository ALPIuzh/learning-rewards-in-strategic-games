"""Export the compact Lobeke environment data used by the appendix schematic.

The exporter owns the GUARD dependency and the optimization. The central
``reproducibility`` package consumes only the resulting small arrays and never
imports or modifies the public GUARD repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from experiments.guard.geographic.map_lobeke import (
    _GUARD_ROOT,
    _grid_array,
    build_game,
    build_lobeke_zero_sum_game,
    home_base_position,
)
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.planning.leader import LeaderOptimizationConfig, optimize_leader_policy
from mairl_exp.execution.artifacts import sha256_file


def _guard_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_GUARD_ROOT, text=True).strip()


def _grid(gsg, target_nodes: list[object], values: np.ndarray) -> np.ndarray:
    return _grid_array(gsg, dict(zip(target_nodes, values, strict=True)))


def export_environment(output_dir: Path, *, eta: float = 1.25) -> tuple[Path, Path]:
    """Rebuild GUARD's Lobeke game and freeze one source-task deployment."""

    gsg = build_game()
    game, true_omega, ranger_reward, target_nodes, coverage = build_lobeke_zero_sum_game(gsg, eta=eta)
    optimization = PolicyOptimizationConfig(method="multistart", random_starts=16, batch_refine_starts=2)
    solution = optimize_leader_policy(
        game.problem, true_omega, ranger_reward, LeaderOptimizationConfig(optimization), seed=0
    )

    target_values = np.asarray([gsg.graph.nodes[node]["score"] for node in target_nodes], dtype=np.float64)
    arrays = {
        "target_value": _grid(gsg, target_nodes, target_values),
        "patrol_coverage": _grid(gsg, target_nodes, coverage.T @ solution.policy),
        "poacher_attack": _grid(gsg, target_nodes, solution.follower_policy),
    }
    if not np.isclose(arrays["patrol_coverage"].sum(), 1.0, atol=1e-10):
        raise RuntimeError("the exported patrol coverage must sum to one")
    if not np.isclose(arrays["poacher_attack"].sum(), 1.0, atol=1e-10):
        raise RuntimeError("the exported poacher distribution must sum to one")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.npz"
    np.savez_compressed(data_path, **arrays)
    digest = sha256_file(data_path)
    provenance = {
        "schema_version": 1,
        "environment": "Lobeke National Park",
        "source": {
            "dependency": "GUARD",
            "guard_revision": _guard_revision(),
            "grid_shape": [5, 5],
            "target_count": len(target_nodes),
        },
        "model": {
            "eta": float(game.problem.eta),
            "feature_names": ["coverage", "target value", "coverage x target value"],
            "true_omega": true_omega.tolist(),
            "source_game_is_zero_sum": True,
            "ranger_base_position": list(home_base_position()),
        },
        "deployment": {
            "description": "Stackelberg-optimal patrol for the source ranger reward",
            "optimization_seed": 0,
            "optimization_method": optimization.method,
            "random_starts": optimization.random_starts,
            "batch_refine_starts": optimization.batch_refine_starts,
            "leader_value": float(solution.value),
        },
        "arrays": {
            name: {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "sum": float(values.sum()),
            }
            for name, values in arrays.items()
        },
        "data_sha256": digest,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return data_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eta", type=float, default=1.25)
    args = parser.parse_args()
    for path in export_environment(args.output_dir, eta=args.eta):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
