# Toy example

The toy game compares Active IRL, fixed random policies, and per-round random
policies with the same observation budget.

```bash
uv run python -m experiments.run --cfg
uv run python -m experiments.run experiment.seed_count=10
```

Results are saved under `artifacts/`. Plot a completed run with
`uv run python -m experiments.plot run_dir=<run-directory>`.

The warm-up and temperature sensitivity studies are preserved in `archive/`.
