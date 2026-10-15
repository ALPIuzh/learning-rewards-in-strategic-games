# Frozen Lobeke final evaluation, 2026-09-21

Select the minimum 32-seed heatmap mean separately for zero and positive warm-up:
no warm-up, information multiplier 0.1; warm-up 50, multiplier 0.03.
Use MLE ridge 1e-6 for all MLE methods; retain the existing OLS estimators.
Run 50 paired fresh learning seeds 2026097000..2026097049, 1600 observations,
report checkpoints every 40 observations, and retain full Active trajectories.
These seeds are disjoint from the tuning and cluster timing pilots.
Nine protocols: Active, fixed random, and online random, each with and without
50-observation warm-up; fixed-expert MLE, paired zero-sum OLS, follower-only OLS.
Each protocol runs in its own process, with all 50 seeds vectorized when supported.
Freeze fresh downstream task seeds 202609700 (primary diagnostics/maps) and
202609701 (180-task behavioral-transfer diagnostic) before inspecting outcomes.
Use existing task-generation rules; uncertainty is conditional on the fixed suites.
Evaluate each protocol on the same tasks. Keep selected settings unchanged.
Report means and paired seed uncertainty; do not reselect using final outcomes.
Validate numerical convergence and stricter deployment optima before figure export.
Preserve every original dataset. New data do not overwrite current paper figures.
Heatmap, environment, sparse-information, and lower-bound data remain unchanged.
