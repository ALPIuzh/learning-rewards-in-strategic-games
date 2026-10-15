# Targeted tuning refinement

Preserve every original file. This is exploratory selection, not independent confirmation. Reuse the 15 ridge-zero cells (eight learning seeds 94000–94007; 100 fixed tasks, seed 9400, candidate pool 1000), explicitly marked as historical in the plot. New cells use MLE ridge 1e-6, the same seeds/tasks and 1600 total observations. Historical cells are not relabelled as exact ridge-1e-6 runs.

First run two full-length seeds at (warm-up, information multiplier): (0,.03), (50,.003), (200,.003), (200,.01), (25,.01), (100,.01), (50,.001). These check execution cost and adaptive changes against saved ridge-zero runs where available. Stop to investigate if the pilot takes over eight minutes or fails numerical checks. No data is overwritten. Pilot results count toward the final eight seeds.

Planned final new/exact candidates: warm-up 25 and 100 crossed with multipliers .003, .01 and .03; additionally (50,.001) to extend the lower end and (100,.1) to fill the existing decade gap. Exact anchor reruns: (0,.03), (50,.003), (200,.003), (200,.01). Total 12 cells / 96 seed runs, including pilot runs. Do not extend to larger warm-up budgets without evidence; do not use two noisy pilot seeds to declare a winner.

Before the full run, compare measured pilot time with a 45-minute training budget; reduce breadth rather than exceed it without a documented reason. Evaluate new cells on the original 100 fixed tasks with full scalar 64-start deployment optimization and tolerance 1e-12, reusing exact task oracles. Estimate 10–15 minutes for 9600 deployments. Use at most four worker processes. Save each cell/batch and each task; resume rather than repeat.

Plot means with numerical cell values and provenance markers. Gray cells are unrun, never interpolated. Show paired seed uncertainty for the best new setting against exact reruns of retained settings; these are descriptive selection comparisons, not confirmatory tests. A lower grid mean is not evidence of a warm-up benefit without a matched-multiplier comparison. Select a provisional setting by lowest mean among newly/exactly evaluated ridge-1e-6 cells; ties prefer less warm-up then smaller multiplier. Do not replace main experiment results or claim an independently validated winner.

## Pilot decision

The seven two-seed cells completed in 125.5 seconds. Extrapolation gives roughly 12–20 minutes for the remaining 82 seed runs; proceed with the fixed 12-cell plan. No candidate was discarded based on noisy pilot rankings. Adaptive trajectories can differ materially from historical results, so reuse remains explicitly historical rather than claiming exact ridge-1e-6 equivalence.

The original task provenance specifies candidate pool 2000, not 1000. Correct this before any evaluation. The unused pilot task matrix and original script are preserved under provenance; training is unaffected because downstream tasks do not enter training. All comparisons will use the exact original seed-9400 / pool-2000 suite. A two-seed ridge-zero control for the no-warm anchor will distinguish the tiny-ridge perturbation from other execution differences.
