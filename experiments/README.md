# Experiments

Each folder contains its game definition, studies, and result exports.
Run any study through the shared launcher:

```bash
uv run python -m experiments.run --config-name runs/lobeke/comparison
```

Recipes live in `conf/runs/`. Add `--cfg` to inspect settings or pass Hydra overrides.
Results are saved under `artifacts/`.

| Experiment | Description |
|---|---|
| [Toy](toy/README.md) | Small baseline comparison |
| [Sparse information](sparse_information/README.md) | A game with one informative leader action |
| [Zero-sum comparison](zero_sum_comparison/README.md) | Active IRL and fixed-leader OLS estimators |
| [Lower bound](minimax_lower_bound/README.md) | Rate check on the minimax hard instance |
| [GUARD](guard/README.md) | Security-game applications |
| [Lobeke paper study](guard/paper/README.md) | Main application and downstream transfer |

Run `make smoke` for short execution checks. See
[reproducibility](../reproducibility/README.md) for full experiment commands and
[the main README](../README.md#run-an-experiment) for configuration, resumption,
and plotting.
