# Lobéké hyperparameter sensitivity

`make plots` renders the full 35-cell grid from `data.npz`: 32 paired learning
seeds, 1,600 observations, 100 fixed transfer tasks, and MLE ridge 1e-6.
The array axes are warm-up, information multiplier, learning seed, and task.
Average tasks within each seed, then average seeds. Standard errors are across
learning seeds, conditional on this task suite.

`source_config.yaml` preserves the original base configuration; the run overrides
its MLE ridge with 1e-6. The minimum among zero-warm-up cells selects multiplier 0.1; among positive-warm-up
cells it selects 50 observations and multiplier 0.03. Final evaluation uses
50 fresh learning seeds, separate transfer tasks, and the same MLE ridge. The 50-seed refinement, earlier grids, raw runs,
validation records, and original regeneration scripts are preserved in
`archive/heatmap-publication-20260921/` in the private repository.
