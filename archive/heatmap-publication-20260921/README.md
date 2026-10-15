# Heatmap publication archive

The live paper figure uses the complete 32-seed grid in
`reproducibility/data/lobeke/sensitivity_32seeds/` (repository root).

- `output/tuning-32seeds-20260919/`: original full-grid results, raw runs,
  scripts, provenance, numerical checks, and plotting summaries.
- `output/tuning-50seeds-20260919/`: 50-seed refinement of the first three
  columns. Its first 32 seeds are exactly those of the full grid.
- Other `output/tuning-*` folders: earlier grids and exploratory checks.
- `output/pdf/`: superseded exported heatmaps.
- `reproducibility/data/lobeke/ridge_zero_tuning/`: previous eight-seed grid
  and confirmation replay, with its original source configuration.
- Other repository-relative paths: former renderer, exporter, tests, and
  Makefile commands for the retired grid.

Original scripts retain their original paths and may need a historical checkout
or restored output layout for execution. No simulations were rerun or relabeled.
The standalone `summarize.py` scripts beside the compact data regenerate their
original plots without access to training runs.
