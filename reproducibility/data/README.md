# Figure data

These exports contain the numeric arrays used by the paper figures. JSON files
record the source runs, seeds, metric definitions, and checksums; NPZ files store
the arrays. Full trajectories are saved separately under `artifacts/`.

`lobeke/source_convergence_unit_reward_v2/` contains all per-seed source SubOpt
checkpoints. Rebuild it with
`experiments.guard.paper.figures.export_source_convergence`.
`lobeke/downstream_figures.json` identifies the policy grids, scores, and
environment data used by the downstream maps.

`lobeke/paired_ols_correction/` is still needed when exporting historical runs.
Superseded evaluation and tuning exports and the synthetic zero-sum results
are preserved under `archive/reproducibility/data/`.
