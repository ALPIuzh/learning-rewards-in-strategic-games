# Zero-sum estimator comparison

This experiment compares Active IRL, follower-only OLS, and zero-sum OLS using
200 follower observations. It uses the toy game's three leader actions, four
follower actions, and three reward features. The leader reward is the negative
of the follower reward, with inverse temperature `eta = 1.25`.

Both OLS methods use the same fixed QRE expert and follower samples. Active IRL
chooses its leader policies adaptively. Metrics include Fisher-norm estimation
error, coefficient error, and the error of the reconstructed QRE leader policy.

## Reproduce

From the repository root:

```bash
uv run python -m experiments.run --config-name runs/zero_sum/warmup_tuning
uv run python -m experiments.run --config-name runs/zero_sum/comparison
```

Export plots from the run directories printed by those commands:

```bash
uv run python -m experiments.zero_sum_comparison.export_plots \
  --tuning-run <tuning-directory> --comparison-run <comparison-directory>
uv run python -m experiments.zero_sum_comparison.downstream_tasks \
  --comparison-run <comparison-directory>
```

Plots are saved under `artifacts/plots/zero_sum_comparison/`.
Use `--config-name runs/zero_sum/joint_selection` for a joint warm-up and regularization sweep;
use `--cfg` to inspect its settings.
