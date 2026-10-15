# Earlier test assertions

Snapshots from commit `dd48937` before removing eight redundant or presentation-specific
tests: hardcoded figure registration, TeX wrapper tokens, output paths, exact map
coordinates, duplicated figure verification, and frozen source metadata literals.

The active suite retains numerical identities, execution and seed behavior,
checksum rejection, export preservation, data consistency, and TeX compilation.
`make check-figures` owns the tracked figure comparison.

Paths below this directory match the original test paths. Restore into a separate
checkout when inspecting the old assertions.
