"""Describe the first confirmation's bias and variance in the expert Fisher basis."""

import json
from pathlib import Path

import numpy as np

from mairl.core.math import fisher_information
from mairl.core.problem import compile_problem

root = Path(__file__).parent
data = np.load(root / "compact-data.npz")
problem = compile_problem(data["features"], eta=float(data["eta"]), radius=float(data["radius"]))
policy = np.load(root / "raw/confirmation/batch-00/fixed_expert_matched_ridge.npz")["policies"][0, 0]
eigenvalues, basis = np.linalg.eigh(fisher_information(problem, policy, data["true_omega"]))
result = dict(
    eigenvalues=eigenvalues.tolist(),
    true_coefficients_in_fisher_basis=(data["true_omega"] @ basis).tolist(),
    eigenvectors_as_columns=basis.tolist(),
    scope="descriptive first confirmation; truth used only for evaluation",
)
names = list(data["protocols"]) + [f"fixed_expert_ridge{ridge:g}" for ridge in data["ridge_values"]]
omegas = np.concatenate((data["omegas"], data["ridge_omegas"]))
result["methods"] = {
    name: dict(
        coefficient_rmse=float(np.sqrt(np.mean(np.sum((omega - data["true_omega"]) ** 2, axis=1)))),
        weak_direction_mean=float((omega @ basis[:, 0]).mean()),
        weak_direction_std=float((omega @ basis[:, 0]).std(ddof=1)),
    )
    for name, omega in zip(names, omegas, strict=True)
}
(root / "ridge-geometry.json").write_text(json.dumps(result, indent=2) + "\n")
