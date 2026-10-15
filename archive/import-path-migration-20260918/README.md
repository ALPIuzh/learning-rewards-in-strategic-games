# Import-path migration

`migration.json` records the verified original runs, class mappings, and new run/plot IDs.
Migrated copies are under `artifacts/migrated-imports/`; original runs remain unchanged.
`originals/` preserves the original metadata and removed compatibility modules.

All recorded run identities, Git fingerprints, source snapshots, completed result hashes,
and plot hashes were checked before migration. Only import paths and dependent metadata
were updated. Original execution provenance is retained; no experiment was rerun.

Thirteen older toy configs also required an earlier application-module rename. Their
generated games match the historical implementation. Four calibration configs used active-only job schemas. One needed
`warmup_enabled` converted to `warmup_modes` with `methods: [active]`, matching its
original executor. All four now explicitly select active learning; 140 job identities
were regenerated without changing arrays.

Some original runs and their migrated copies were subsequently moved together.
See [retired data](../retired-data-20260918/README.md) for their current locations.
