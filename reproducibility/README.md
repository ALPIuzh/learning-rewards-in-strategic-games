# Reproducing the results

After [setup](../README.md#setup), run these commands from the repository root:

```bash
make smoke
make plots
```

`make smoke` runs short simulations across seven experiment workflows and saves
logs and results under `artifacts/smoke/`. These checks exercise data generation;
they do not repeat the full paper experiments.

`make plots` rebuilds the nine empirical paper figures and one alternate layout
from the included numeric data and
compiles `output/pdf/preview.pdf`. You need LaTeX with TikZ/PGFPlots and `latexmk`.
To generate and check the TikZ files without compiling a PDF:

```bash
make generate
make check-figures
```

The figure list is in [`figures.toml`](figures.toml), data in [`data/`](data/),
and rendering code in [`figures/`](figures/).

## Generate new data

These commands run the full experiments and can take substantially longer than
`make smoke`. Use `--cfg` on a Python launcher to inspect its settings first.

| Experiment | Command | Details |
|---|---|---|
| Lobeke | `uv run python -m experiments.run --config-name runs/lobeke/comparison` | [Protocol and exports](../experiments/guard/paper/README.md#reproducing) |
| Sparse information | `uv run python -m experiments.run --config-name runs/sparse_information/comparison` | [Sparse-information experiment](../experiments/sparse_information/README.md#reproduce) |
| Lower bound | `uv run python -m experiments.run --config-name runs/minimax` | [Summary export](../experiments/minimax_lower_bound/README.md#average-and-maximum-coordinate-risk) |
| Zero-sum comparison | `uv run python -m experiments.run --config-name runs/zero_sum/comparison` | [Comparison and plots](../experiments/zero_sum_comparison/README.md) |

The included [Lobeke sensitivity data](data/lobeke/sensitivity_32seeds/) regenerate
the heatmap without training.

Use the run directories printed by your commands when exporting results.
Historical run IDs in the experiment notes refer to data collected for the
paper. Export into a new directory and compare with the included data before
replacing it.

## Use figures in LaTeX

Include the shared style in your preamble:

```latex
\input{reproducibility/tex/figure_style}
```

Then include a generated fragment inside a figure environment, for example
`\input{reproducibility/tex/generated/main/m1_source_convergence}`.

To add a figure, add its data and recipe, register it in `figures.toml`, and run
`make plots`. Captions live in `tex/sections/`; shared styles live in
`tex/figure_style.tex`. [FIGURES.md](FIGURES.md) describes the experiments behind
each figure.

To iterate on one figure, edit its recipe and regenerate just that fragment:

```bash
uv run python -m reproducibility.figures generate m1_source_convergence
uv run python -m reproducibility.figures check m1_source_convergence
```

The check reports stale output; it does not freeze the figure. For a hand-written
TikZ draft, register `recipe = "copy"` with a repository-relative `source` path
instead of `data`. Include the generated fragment in your LaTeX document to preview it.
