# Archived research material

Historical code, results, and notes, retained in the private repository and excluded
from Git source archives. Paths are preserved so related code, configs, and tests
can be located together.

## Find material

| Looking for | Location |
| --- | --- |
| Final 50-seed evaluation, raw runs, numerical checks, and superseded paper data | [Final evaluation](final-evaluation-20260921/README.md) |
| Earlier heatmaps, 50-seed refinement, and full tuning runs | [Heatmap archive](heatmap-publication-20260921/README.md) |
| Scientific claim audit and fresh Lobeke confirmation | [Validation report](scientific-validation-20260919/REPORT.md), with the frozen protocol, endpoints, and manuscript corrections |
| Efficiency benchmarks and paired-data checks | [First pass](efficiency-review-20260919/REPORT.md), [second pass](efficiency-round2-20260919/REPORT.md) |
| Moved raw results and their original locations | [Retired data](retired-data-20260918/README.md), with checksums in `relocations.json` |
| Which local results are current or replaced | [Data audit](data-audit-20260918/REPORT.md), with every run in [inventory.json](data-audit-20260918/inventory.json) |
| Original configs and provenance before import-path migration | [Migration record](import-path-migration-20260918/README.md); `migration.json` maps original runs to loadable copies |
| Earlier Lobeke evaluation, tuning, and warm-up factorial data | [reproducibility/data/lobeke/](reproducibility/data/lobeke/) |
| Structured toy sensitivity and zero-sum data | [reproducibility/data/](reproducibility/data/) |
| Historical Lobeke renderers and full-grid workflow | [Lobeke cleanup](lobeke-cleanup-20260918/README.md) |
| Old figure recipes and alternative layouts | [reproducibility/figures/recipes/](reproducibility/figures/recipes/), [alternatives/](reproducibility/alternatives/) |
| Pilot experiments, exported plots, and geographic assets | [experiments/](experiments/) |
| Exploratory analyses and their findings | [investigations/](investigations/) |
| Historical experiment documentation | [previous_docs/experiments/](previous_docs/experiments/) |
| Test setup before simplification | [Test simplification](test-simplification-20260918/README.md) |
| Earlier presentation and configuration assertions | [Test cleanup](test-cleanup-20260918/README.md) |
| Configs, library code, and tests for archived studies | [conf/](conf/), [src/](src/), [tests/](tests/) |

## Retrieve a file

`manifest.json` lists each preserved file's path relative to `archive/`, size,
SHA-256 checksum, and archival reason. `original_path`, when present, records its
former repository location; otherwise the archive-relative path is the former
location. `previous_docs/` contains documentation snapshots from the corresponding
paths without that prefix. Dated audit reports are new records, not former source files.

From the repository root, search filenames or archival reasons:

```sh
rg --files archive | rg 'warmup|structured|calibration'
rg -n 'warmup|structured|calibration' archive/manifest.json
```

Copy historical files into a separate checkout when recovering a study. Scripts
may depend on their contemporary code and environment; use `git log --all -- path`
to find that history. Migration records retain the original source commits and
execution provenance. A metadata migration is not an experiment rerun.

Older raw trajectories remain local and Git-ignored under `artifacts/` and
`archive/retired-data-20260918/artifacts/`. The new scientific validation also
preserves its raw outputs in the private repository under
`scientific-validation-20260919/raw/`.
Migration backups here preserve metadata, not another copy of those trajectories.
README files used for navigation, the manifest itself, and disposable local caches
are not checksummed in the manifest.
