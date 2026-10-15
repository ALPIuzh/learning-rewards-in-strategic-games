# Final Lobéké evaluation — 21 September 2026

All 450 training runs and 1,719 task/protocol evaluations completed. The run uses 50 fresh seeds, 1,600 observations, and MLE ridge 1e-6. Active IRL uses multiplier 0.03 with 50 warm-up observations, or multiplier 0.1 without warm-up. These settings were frozen before evaluation. The tuning heatmap remains unchanged.

Training used the clean cluster snapshot of repository commit
`a12b046a0a71d48d68089145aa9070414845d09b`; the run manifests retain the
cluster snapshot's own Git commit.

The numerical audit checked 472,650 saved policy deployments: finite arrays, simplex constraints, checksums, and consistency between reported loss and saved values. For each of 27 suite/protocol combinations, the first case and the largest-loss case were reevaluated with 256 optimizer starts instead of 64. The maximum loss change was 1.15e-8. These are numerical checks, not a proof that every nonconvex solve is globally optimal.

At behavioral checkpoints absent from the random warm-up traces, MLE was refitted from the original saved observations; no observations were resampled. All refit residuals were below 1e-6. Exact details are in `validation.json`.

Final mean transfer loss on the 180-task suite:

| Method | Mean | 95% CI across learning seeds |
|---|---:|---:|
| Active, no warm-up | 0.000274 | [0.000185, 0.000364] |
| Active, warm-up | 0.000285 | [0.000193, 0.000377] |
| Online random, warm-up | 0.001800 | [0.001383, 0.002217] |
| Fixed-expert MLE | 0.003380 | [0.002533, 0.004226] |

Intervals are conditional on the fixed task suite. The paired warm-minus-no-warm difference is 0.0000109, with 95% CI [-0.0001194, 0.0001413]; this evaluation does not establish a warm-up advantage.

- `PROTOCOL.md`: frozen campaign specification.
- `summary.json`: all methods, suites, seedwise results, and confidence intervals.
- `final-checks.json`: numerical audit and hashes of all evaluation files.
- `evaluations/`: original taskwise policies, values, losses, and Fisher errors.
- `compact.npz`: audited compact arrays consumed by the exporter.
- Current figure inputs: `reproducibility/data/lobeke/` from the repository root. Superseded inputs are in `previous/`.
- Four unaffected figure bodies remain unchanged: environment, tuning, sparse information, and lower bound.
- `runs/`: all nine training runs, preserving their original manifests and result checksums. Cluster and local working copies are also retained.

From the repository root, `make plots` rebuilds all figures from the compact data.
To replay the original campaign scripts in a separate checkout, copy the files in
this directory (except `runs/` and `previous/`) to `output/final-20260921/`, and
copy `runs/lobeke-final-20260921-*` to `artifacts/runs/`. Run `export.py` with
`PYTHONPATH=.` after removing only that scratch checkout's `exports/` directory.
The archived `train.sh` records the original cluster command and absolute paths;
adapt those paths when running elsewhere. Original scripts are preserved unchanged
so their hashes remain verifiable.

`export.py` builds compact figure inputs using the audited distributed results through the existing export schemas. It refuses to overwrite existing export folders. `evaluate.py` and `finish.py` retain the evaluation and verification procedures.

The fresh ten-task illustration suite selects task indices 5, 6, and 1 by oracle-policy distance alone; their distances are 0.366, 0.857, and 1.000. Repository captions now reflect these values, warm-up 50, the selected multipliers, and ridge 1e-6. The external manuscript needs the same edits.
