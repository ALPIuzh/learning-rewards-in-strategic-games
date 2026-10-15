# Uniform 32-seed heatmap

Purpose: replace the mixed-provenance eight-seed display with a fully observed, uniform experiment and communicate uncertainty without smoothing or removing unfavourable data. Preserve all previous runs and figures. Do not edit the manuscript while the user is editing it.

Fixed design: all 35 combinations of warm-up budgets [0,25,50,100,200] and information multipliers [.001,.003,.01,.03,.1,.3,1]. Every cell uses MLE ridge 1e-6, 1600 total observations, and 32 fresh learning seeds 2026094000–2026094031. Four identical-size batches of eight seeds are used for every cell. Original 100 evaluation tasks (seed9400, candidate pool2000) and strict reference optima are reused exactly. No historical ridge-zero cell enters the new figure.

Pilot: run the first eight seeds for (25,.003), the previously noisy cell, and (0,.001), an earlier promising cell. These runs are retained in the final grid. Check finite outputs, simplex constraints, coefficient radius, optimizer residuals, provenance, and measure end-to-end cost including 100-task evaluation. No score-based pruning.

Runtime estimate from prior runs: roughly 45–90 minutes for all 1120 seed runs including strict evaluation with six worker processes. Each worker trains one eight-seed cell and evaluates its eight endpoints. Use one BLAS/OpenMP thread per process and lowered scheduling priority. Pilot uses two workers. Inspect if either pilot batch exceeds 8 minutes or numerical checks fail before expanding. For the full run, set a 100-minute review budget; if exceeded, inspect progress and explicitly record an updated estimate before extending. Never delete failed or expensive cases to beautify the plot.

Freeze all 35 cells and 32 seeds before observing fresh scores. Save each batch and each seed evaluation atomically; completed outputs are validated and reused on resume. Reuse existing strict oracles. Keep diagnostics and all per-task scores. Use 64 starts and tolerance1e-12 for evaluation. Further spot checks may use more starts on the highest-loss cases, without replacing saved values silently.

Deliver a clean main heatmap of means and a companion uncertainty heatmap or table. Use actual categorical grid locations, no interpolation or smoothing. Report SE over independent learning seeds and descriptive split-half stability. Thirty-two seeds improve precision but do not guarantee reliable ranking; selection on this grid remains exploratory. Do not declare warm-up beneficial or alter paper defaults solely because a cell has the smallest mean.

The sandbox disallowed lowering scheduling priority (`nice`), so processes run at normal priority. The full run remains capped at six single-threaded workers on the 12-core machine.

## Pilot decision

Both eight-seed pilot cells passed training and all 100-task evaluations. End-to-end times were 224.2 and 228.0 seconds. The observed cost projects approximately 67 minutes for 140 batches at eight workers, before contention and scheduling overhead. Proceed with the frozen 35-by-32 design using eight single-threaded worker processes on 12 cores, leaving four cores available for interactive work. This revises the earlier six-worker cap based on measured cost. Allow roughly 70–100 minutes; inspect progress at the 100-minute budget rather than silently expanding it. Completed pilot outputs are reused.
