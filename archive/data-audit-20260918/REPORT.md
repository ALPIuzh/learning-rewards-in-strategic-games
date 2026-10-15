# Data audit — 18 September 2026

This is the pre-move audit. See [the relocation guide](../retired-data-20260918/README.md) for subsequently archived results. This inventory covers 180 original runs and 47 metadata-migrated copies outside `archive/`. An older Git commit does not by itself make a result obsolete.

## Needed for the current paper

- Rendering: the frozen exports in `reproducibility/data/`, recipes, and figure registry. `paired_ols_correction` supports export regeneration rather than direct rendering.
- Rebuilding exports: the four runs below, the two `ridge-zero-*-publication-v1` derived datasets, and the transfer/behavioral evaluation caches. The caches can be recomputed but save work.
- Historical export support: `artifacts/runs/guard-lobeke-paper/a4b1b2a1cdf886dd` supplies the paired-OLS correction. Keep it even though the final plots use corrected-v2 runs.

| Source run | Completed jobs |
| --- | ---: |
| `artifacts/runs/guard-lobeke-paper-ols-corrected-v2/cf7b0f494e850f2d` | 100 |
| `artifacts/runs/sparse-information-action-sweep/219b99056804830c` | 1500 |
| `artifacts/runs/guard-lobeke-paper-mle-corrected-v2/349676e5922332fd` | 350 |
| `artifacts/runs/minimax-lower-bound/0ce1a906d05aa86d` | 160 |

## Replaced or historical

| Older data | Current replacement / role |
| --- | --- |
| Earlier `guard-lobeke-paper` runs, excluding `a4b1b2a1cdf886dd` | Corrected MLE `349676e5922332fd` and OLS `cf7b0f494e850f2d`. Includes pilots and failed attempts; not a claim that all were published or numerically equivalent. |
| Pilot `1c191d2dec0ca839` | Explicitly superseded: its zero-sum estimator used the exact expert policy. The archived paper README documents this protocol issue. |
| Grid run `544713bd317e400d` and `derived/guard-lobeke-full-grid-1000` | The current ridge-zero sensitivity and confirmation exports replace this 1,000-sample grid as figure input. |
| Partial lower-bound run `2a9da80adca6ce3d` | Current source `0ce1a906d05aa86d`; different execution protocol, not an interchangeable copy. |
| Structured-temperature, structured-warmup, and toy-warmup runs | Six retained runs from studies whose workflows/figures are archived. |
| `experiments/guard/paper/results/figure2_assets_native_v2` | Older downstream export; current rendering reads `reproducibility/data/lobeke/downstream_assets`. |

The two earlier `active-expert-warmup-ridge-zero-*` derived datasets remain sensitivity intermediates. Zero-sum studies remain supported optional experiments. Neither group is classified as dispensable.

## Inventory

Sizes are file bytes, excluding filesystem allocation overhead. Counts include incomplete runs.

| Category | Runs | MiB |
| --- | ---: | ---: |
| archived-study-source | 6 | 176.0 |
| calibration-history | 7 | 3.7 |
| current-paper-source | 4 | 203.6 |
| development-check | 101 | 228.3 |
| historical-export-support | 1 | 41.6 |
| historical-or-unclassified | 17 | 63.1 |
| migration-copy | 47 | 172.7 |
| optional-study-history | 7 | 38.6 |
| selection-or-diagnostic-intermediate | 25 | 182.5 |
| superseded-paper-input | 12 | 150.0 |

The 17 unclassified historical runs are baseline-comparison, sparse-information comparison, and a small sparse sweep. They are not current figure sources; no exact replacement is established. Matching semantic configs are listed separately in the inventory and are not treated as duplicates across code versions.

Selection, confirmation, calibration, failed jobs, and smoke results are retained. Migration copies contain identical data with updated import/configuration metadata; original provenance remains available.

## Verification and limits

The nine recorded run-provenance check groups passed. The current figure check verifies all ten tracked figure checksums; no expensive simulations were rerun. The migration independently verified 6,019 completed-job checksums and 6,827 copied data files.

`inventory.json` records every run path, commit, completion status, config summary, classification, replacement evidence, migrated-copy location, frozen-data checksum, and derived-data family. Evidence comes from current export provenance and `archive/previous_docs/experiments/guard/paper/README.md`.

Local `tmp/` renderings, design drafts, generated TeX, previews, and the reference PDF are inventoried separately. Dependency checkouts, environments, tool caches, and clones outside this repository are excluded. Raw `artifacts/` data remain local and Git-ignored; this report does not constitute a remote backup.
