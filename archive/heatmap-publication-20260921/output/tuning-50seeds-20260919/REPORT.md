# 50-seed refinement of the left three heatmap columns

All 15 selected cells now have 50 seeds: the previous 32 were reused exactly and 18 fresh seeds were added. The remaining 20 cells of the full grid were not rerun and remain at 32 seeds. This figure deliberately shows only the uniform 50-seed region.

The smallest observed mean is warm-up 0, information multiplier 0.003: 0.0001746. This is a ranking of sample means, not a claim of a unique optimum.

## Unchanged design

Warm-up budgets: 0,25,50,100,200. Information multipliers lambda / eta squared: .001,.003,.01. MLE ridge 1e-6, 1600 observations, and the same 100 transfer tasks (seed 9400, candidate pool 2000). Existing strict task optima were reused exactly. New training seeds are 2026094032–2026094049, in batches of eight, eight and two for every cell. The merged dataset has seeds 2026094000–2026094049. Evaluation uses 64 starts and tolerance 1e-12.

The 15-cell region was chosen after inspecting the earlier results; the extension to 50 seeds was fixed before examining the added runs. No seed or cell was removed based on its performance. The original 32-seed dataset and figures, the manuscript, and paper defaults were not modified.

## Comparison

All values below are in units of 10^-4. SE is over 50 learning seeds after averaging 100 fixed tasks. The task suite is held fixed; these errors do not represent uncertainty over new task suites. The added 18 column is descriptive, not a new tuning selection rule.

| Warm-up | Multiplier | 32-seed mean | Added 18 mean | 50-seed mean | SE at 50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.001 | 3.10 | 3.28 | 3.17 | 0.57 |
| 0 | 0.003 | 1.72 | 1.79 | 1.75 | 0.21 |
| 0 | 0.01 | 2.86 | 3.26 | 3.00 | 0.37 |
| 25 | 0.001 | 2.17 | 1.97 | 2.10 | 0.32 |
| 25 | 0.003 | 2.06 | 2.65 | 2.27 | 0.29 |
| 25 | 0.01 | 1.97 | 3.46 | 2.51 | 0.30 |
| 50 | 0.001 | 2.14 | 4.08 | 2.84 | 0.63 |
| 50 | 0.003 | 1.73 | 2.76 | 2.10 | 0.29 |
| 50 | 0.01 | 2.27 | 2.79 | 2.46 | 0.37 |
| 100 | 0.001 | 1.86 | 2.37 | 2.04 | 0.26 |
| 100 | 0.003 | 2.27 | 3.40 | 2.68 | 0.48 |
| 100 | 0.01 | 2.44 | 2.42 | 2.43 | 0.36 |
| 200 | 0.001 | 1.91 | 1.80 | 1.87 | 0.23 |
| 200 | 0.003 | 2.00 | 2.55 | 2.20 | 0.40 |
| 200 | 0.01 | 2.14 | 3.51 | 2.63 | 0.48 |

The best positive warm-up compared with no warm-up at its matched multiplier has mean difference -0.00012996 and unadjusted paired 95% interval [-0.00025281, -0.00000710]. The comparison is selected on these data, so it is not independent confirmation. The first versus second 25-seed rank correlation is 0.350; its interpretation is descriptive, since the region was selected using earlier scores.

## Validation and runtime

Added 270 training runs and 27000 task evaluations. All passed finite-value, policy-simplex, coefficient-radius, MLE-residual and provenance checks. The 1570 protected prior files are unchanged. The merged first 32 seeds are bit-identical to the prior compact data. Execution took 22.4 minutes with eight single-threaded workers. Rechecking the highest-loss new case in each cell with 256 optimizer starts changed losses by at most 3.71e-09. The new heatmaps and statistics were regenerated in an isolated folder using only the compact data and plotting script, with byte-identical PNG and JSON outputs.

## Regeneration

Copy `figure-data.npz` and `summarize.py` together; NumPy, SciPy and Matplotlib suffice to regenerate the figures without experiment code or training:

```sh
python summarize.py
```

From the repository root, raw-result verification and collection are available through `validate.py` and `collect.py` in this folder. Raw new training trajectories and evaluations are preserved under `training/` and `evaluation/`. `results.json` includes per-seed scores and descriptive intervals. `heatmap.pdf` and `uncertainty.pdf` are vector figures, with PNG and SVG versions alongside them.
