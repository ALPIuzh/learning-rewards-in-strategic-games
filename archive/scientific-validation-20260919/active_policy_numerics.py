"""Check selected training decisions with full multistart information optimization."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from validate import PROTOCOLS, ROOT, runtime

from mairl.core.math import cumulative_fisher_information
from mairl.core.randomness import RandomPurpose, experiment_rng
from mairl.exploration.policy import information_gain_and_gradient, select_information_policy

started = time.monotonic()
game, numerics = runtime()
config = replace(numerics.active_policy, random_starts=64, tolerance=1e-12)
records = []
for protocol in PROTOCOLS[:4]:
    stored = np.load(ROOT / "confirmation" / "batch-00" / f"{protocol.slug}.npz")
    counts = stored["trace_sample_counts"][0]
    for index, seed in enumerate(stored["seeds"][:5]):
        for sample in (max(200, int(counts[0])), 400, 800, 1599):
            omega = stored["omega_history"][index, np.where(counts == sample)[0][0]]
            information = (
                protocol.lambda_information_multiplier * game.problem.eta**2 * np.eye(game.problem.feature_dimension)
            )
            information += cumulative_fisher_information(game.problem, stored["policies"][index, :sample], omega)
            gain = information_gain_and_gradient(
                game.problem, omega, np.linalg.inv(information), stored["policies"][index, sample]
            )[0]
            full = select_information_policy(
                game.problem, omega, information, config, experiment_rng(int(seed), RandomPurpose.ACTIVE_POLICY, sample)
            )
            records.append(
                dict(
                    protocol=protocol.slug,
                    seed=int(seed),
                    sample=sample,
                    original_gain=float(gain),
                    full_gain=float(full.information_gain),
                    relative_improvement=float((full.information_gain - gain) / gain),
                )
            )
    print(f"checked {protocol.slug}", flush=True)
Path("archive/scientific-validation-20260919/active-policy-numerics.json").write_text(
    json.dumps(dict(seconds=time.monotonic() - started, checks=records), indent=2) + "\n"
)
