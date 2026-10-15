# Learning Rewards in Strategic Games

Code and experiments for [**Learning Rewards in Strategic Games**](https://arxiv.org/abs/2610.21912).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
make install
make guard
```

This installs the Python dependencies and download
[GUARD](https://github.com/CoffeeAndConvexity/GUARD) for the security-game experiments.

We describe below how to reproduce paper figures and run experiments. Use `make help` for a list of all available commands.

## Reproduce the results

```bash
make smoke # runs small experiments to check data generation and plotting
make plots # regenerate paper figures
```

See [reproducibility](reproducibility/README.md) for full experiment commands and
figure exports.

## Run an experiment

For example, run the toy comparison:

```bash
make toy
```

For configuration options, use `uv run python -m experiments.run --cfg`,
or replace `--cfg` with overrides such as `experiment.seed_count=10`.
Recipes are configured in `conf/runs/`, choose one with e.g. `--config-name runs/zero_sum/joint_selection`.

Runs are saved under `artifacts/runs/<study>/<run-id>/` and require a clean Git checkout (use `run.require_clean_git=false` during development).

Inspect progress or plot a run:

```bash
uv run mairl-status artifacts/runs/<study>/<run-id> --watch
uv run python -m experiments.plot run_dir=artifacts/runs/<study>/<run-id>
```

Plots are saved in the run's `plots/` directory.

See [experiment index](experiments/README.md) for more information about the configured experiments.

## License and citation

[MIT license](LICENSE). See [third-party terms](THIRD_PARTY.md) and
[CITATION.cff](CITATION.cff).

If you use this code in your research, please cite our paper:

```bibtex
@misc{bergerault2026learningrewardsstrategicgames,
  title = {Learning Rewards in Strategic Games},
  author = {Antoine Bergerault and Cyrille Kone and Giorgia Ramponi},
  year = {2026},
  url = {https://arxiv.org/abs/2610.21912}
}
```
