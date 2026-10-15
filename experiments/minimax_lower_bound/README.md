# Minimax lower-bound check

This experiment runs Active IRL on the paper's hard family. It is deliberately a
small illustration on a family whose separation shrinks with the budget. A policy
that takes no observations already has loss of order `sqrt(d/K)`, so the plot slope
alone does not demonstrate learning or establish minimax optimality.

For each dimension `d`, sample budget `K`, coordinate `i`, and replicate, the instance
uses

- `phi(a_j, b_1) = 0` and `phi(a_j, b_2) = e_j`;
- leader reward `1{b=b_2}`;
- `eta = S = R_max = 1`;
- `omega = 0.5 sqrt(d/K) e_i`;
- no warm-up and online regularization `lambda = eta^2 = 1`.

The final grid uses `d in {33, 64}`, `K/d in {4, 8, 16, 32, 64}`, and
16 independent replicates of every active coordinate. The smallest dimension is
33, within the theorem's condition `d >= 32`. Coordinates are
balanced and shuffled with a separate design seed, then reused across the
`K/d` values.

## Run and resume

```bash
uv run --extra experiments python -m experiments.run --config-name runs/minimax
```

The generic experiment runner writes a content-addressed, checksummed run under
`artifacts/runs/minimax-lower-bound/`. Re-running the same command resumes
incomplete jobs. Each of the 160 durable jobs contains all coordinates for one
`(d, K/d, replicate)` cell and vectorizes their simulation internally. This
keeps every coordinate-level result while avoiding thousands of tiny files.

Before a final run, profile the heaviest job and extrapolate the complete grid:

```bash
uv run --extra experiments python -m experiments.minimax_lower_bound.profile \
  --output tmp/minimax_lower_bound_profile.json
```

The final run is launched only if this end-to-end projection, which includes
compressed artifact creation, remains below 30 minutes.

## Why this implementation is fast and checkable

On the hard instance, every selected information policy is a simplex vertex. The code
evaluates all vertex information gains in closed form and verifies the selected vertex's
stationarity every round; exact gain ties are broken uniformly with the job's seeded
random-number generator. One-hot policies reduce logistic MLE data to per-coordinate
Bernoulli counts; a vectorized KKT solver computes the same constrained ridge MLE and
checks its projected-gradient residual. Deployment SubOpt and Fisher-transfer error
also use their exact hard-instance formulas.

The artifacts retain follower actions, selected leader actions, full per-round solver
diagnostics, final sufficient statistics, and approximately 20 coefficient checkpoints.
Dense per-round policy matrices are unnecessary because each policy is exactly identified
by its stored action index.

## Average and maximum coordinate risk

After the run completes, export both aggregations with

```bash
uv run --extra experiments python -m experiments.minimax_lower_bound.summarize \
  artifacts/runs/minimax-lower-bound/<run-id> \
  reproducibility/data/lower_bound/summary.json
```

The summary contains the uniform-coordinate mean, every coordinate's replicate mean,
and their maximum for SubOpt, Fisher error, and coefficient error. The maximum is named
`maximum_coordinate_sample_mean` deliberately: it estimates the theorem's maximum
expected risk, and its finite-replicate uncertainty must remain visible in the plot.
