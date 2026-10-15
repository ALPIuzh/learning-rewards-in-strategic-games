# Extend the left three columns to 50 seeds

Only the 15 cells at warm-up budgets 0,25,50,100,200 and multipliers .001,.003,.01 are in scope. Reuse the original 32 fresh seeds and add seeds 2026094032–2026094049, bringing each selected cell to 50. New seeds are trained in batches of eight, eight and two consistently across all 15 cells. Do not rerun or alter any old result.

Keep 1600 observations, MLE ridge1e-6, the identical 100 tasks (seed9400, candidate pool2000), and the strict 64-start evaluation with tolerance1e-12. Reuse exact task oracles. The other 20 grid cells remain at 32 seeds and are not presented as 50-seed results. Deliver a 5-by-3 mean heatmap and companion SE heatmap; preserve the earlier full-grid figure.

Prior measured cost predicts roughly 20–30 minutes with eight single-threaded workers. This adds 270 training seed runs and 27000 task evaluations. Fixed scope and seeds; no score-based stopping or pruning. Review progress if execution exceeds40minutes. The prior 1120-run validation and timing make an additional pilot unnecessary. Check progress after about15minutes, as requested by the user, rather than constantly polling.

Validate all new arrays and their provenance, preserve prior-file hashes, and merge the existing 32-seed data with the 18 added seeds. Report whether the means and uncertainty changed; do not infer an optimal setting merely from the lowest grid mean.
