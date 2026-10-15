# Sparse-information experiment

Only the first leader action affects the follower reward. In zero-based code
indices, the scalar features are `phi(0, 1) = 1` and zero everywhere else.
There are two follower actions, with `omega = 1` and inverse temperature `eta = 2`.

If the leader places probability `x` on the informative action, the follower
chooses action 1 with probability `p = sigmoid(eta * omega * x)`. Its Fisher
information is `eta^2 * x^2 * p * (1 - p)`. All other pure leader actions provide
zero information about the reward coefficient.

## Reproduce

Run the action-count sweep used in the paper:

```bash
uv run python -m experiments.run --config-name runs/sparse_information/action_count
```

It compares 2, 4, 8, 16, and 32 leader actions over 100 paired seeds, with 200
observations per seed and no warm-up. The separate warm-up comparison runs with:

```bash
uv run python -m experiments.run --config-name runs/sparse_information/comparison
```

Plot a completed comparison with:

```bash
uv run python -m experiments.plot run_dir=<run-directory>
uv run python -m experiments.plot run_dir=<run-directory> plot=stackelberg_suboptimality
```

Results and plots are saved under `artifacts/`. Run `make plots` to rebuild the
paper figures from the included data.
