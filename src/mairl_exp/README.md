# Experiment utilities

- `execution/`: run orchestration, job records, seed batching, and artifact storage.
- `studies/`: baseline comparisons, calibration, tuning, and confirmation.
- `evaluation/`: metrics, statistics, plotting, and Matplotlib setup.
- `cli/`: configuration loading and command-line entry points.

`application.py` defines the game interface; `config.py` defines shared paths and numerical settings.

## Profiling and calibration

```bash
uv run mairl-profile --json artifacts/profile/report.json --cprofile artifacts/profile/mairl.prof
uv run python -m experiments.run experiment=numerical_calibration
```

Calibration saves solver choices in `calibration_selection.json` for use in `conf/numerics/`.
Set `run.seed_batch_size` to adjust seed batching (default: 64); compare sizes with
`uv run python -m experiments.toy.profile_seed_batches --batch-sizes 16 32 64 100 --workloads warmup/active`.
