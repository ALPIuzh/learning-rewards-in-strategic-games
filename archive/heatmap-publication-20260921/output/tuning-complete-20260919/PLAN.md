# Complete the displayed tuning grid

Fill only the 12 previously unrun cells. Preserve all previous data, figures and scripts. Use MLE ridge 1e-6, 1600 observations, seeds 94000–94007 in the same two-plus-six batches, and the identical 100-task suite (seed 9400, candidate pool 2000). Reuse the existing strict task optima. Do not reinterpret the 11 historical ridge-zero references as ridge-1e-6 runs.

Missing (warm-up, information multiplier) pairs:

[(0, 0.001), (0, 0.1), (25, 0.001), (25, 0.1), (25, 0.3), (25, 1), (50, 0.1), (100, 0.001), (100, 0.3), (100, 1), (200, 0.001), (200, 0.1)]

Previous measured training was 17.4 minutes for 12 cells plus a control. Expect roughly 15–25 minutes training and 8–12 minutes evaluation. Check the first two-seed cell before starting the six-seed batch. Run the two independent training batches concurrently thereafter; evaluate completed batches with at most two workers while both train. If the first cell takes over three minutes or fails numerical checks, investigate before expanding. If total training exceeds 45 minutes, inspect progress and document any extension before continuing. Do not stop or prune cells based on their scores.

Selection uses the lowest mean among the 24 cells evaluated at ridge 1e-6; ties prefer fewer warm-up observations then smaller multiplier. This is further exploratory tuning on the same seeds/tasks, not independent confirmation. The completed display has 35 cells, including 11 separately marked historical ridge-zero references. All previous results remain intact in the sibling tuning-refinement-20260919 folder.

The first two-seed cell (no warm-up, multiplier .001) took 20.3 seconds and passed finite-value, simplex and coefficient-radius checks. Proceed with both batches.
