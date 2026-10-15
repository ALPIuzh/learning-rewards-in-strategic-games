# Retired data — 18 September 2026

Moved 119 superseded, archived-study, and development runs, 21 corresponding
migration copies, the old tuning-grid cache, and the older downstream export.
All 37,093 files (666,853,834 bytes) passed SHA-256 verification after the move.

`relocations.json` maps each original directory to its archive location and lists
its files, sizes, and checksums. Directory contents and embedded provenance are
unchanged. To restore a result, copy its `archive_path` to its `original_path`
in a separate checkout; restore any referenced runs from the same mapping.

`artifacts/` here remains local and Git-ignored, as it was before the move.
The two previously tracked downstream export files remain tracked under
`experiments/`. The mapping and this guide are committed to the private repository.

Current paper sources, paired-OLS correction support, current derived data,
tuning/calibration intermediates, optional-study results, and uncertain cases
remain at their existing locations. The earlier data audit is a snapshot from
before this move; use this mapping for current locations.
