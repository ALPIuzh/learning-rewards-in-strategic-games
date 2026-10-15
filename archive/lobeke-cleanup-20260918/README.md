# Historical Lobeke workflows

Original files from commit `022847045714df817568c2ef1e6bff7cb93e2f77`.
Paths below this directory match their former repository locations.

- `experiments/guard/paper/figures/`: alternate Matplotlib presentations, old tuning exports, and the superseded 1,000-sample grid exporter.
- `experiments/guard/paper/watch_full_grid.py`: the old grid watcher.
- `conf/`: experiment and launcher recipes for that grid.
- `tests/`: its dedicated tests.
- `snapshots/`: original mixed modules and Makefile before historical sections were removed.

Restore files into a separate checkout of the recorded commit when revisiting
these workflows. The archived Makefile preserves their launch commands.
The [data relocation index](../retired-data-20260918/README.md) locates retired raw results;
[earlier exports](../reproducibility/data/lobeke/) preserve superseded figure data.

Current paper rendering uses `make plots`. Current run-to-figure exports,
transfer evaluation, sensitivity experiments, and selection/confirmation studies
remain outside the archive.
