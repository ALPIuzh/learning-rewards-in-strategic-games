# Completed Lobéké tuning grid

All 12 formerly blank cells have been computed. The plot now contains all 35 combinations of warm-up budgets 0, 25, 50, 100, 200 and information multipliers .001, .003, .01, .03, .1, .3, 1.

The lowest mean among the 24 ridge-1e-6 cells is warm-up 25, multiplier 0.001, mean transfer SubOpt 0.0001289. The best positive warm-up is 25, multiplier 0.001, mean 0.0001289. These are descriptive rankings on the tuning set, not independent confirmation or proof of a warm-up benefit.

At the same multiplier, the best positive warm-up differs from no warm-up by -0.00000088, with an unadjusted paired 95% interval [-0.00007859, 0.00007683]. This does not establish a warm-up benefit.

## Data and selection

Only the 12 missing configurations were trained: 96 learning seed runs and 9,600 task evaluations. Each uses 1,600 observations and the same eight seeds (94000–94007) in the same two-plus-six batches as the previous refinement. The 100 downstream tasks (seed 9400, candidate pool 2000), game and strict task optima are copied byte-for-byte from that refinement. Deployment uses 64 starts and tolerance 1e-12.

The 12 previously recomputed ridge-1e-6 cells are reused without modification. Eleven historical ridge-zero cells remain marked R and are excluded from selection under ridge 1e-6. Thus the display is complete, but it is not a homogeneous 35-cell ridge-1e-6 experiment. The earlier refinement and all original paper data are preserved. The information multiplier lambda / eta squared affects policy selection; it is distinct from MLE ridge.

No cells were discarded based on pilot scores, and no main-paper result or selected configuration was overwritten. Extending the grid after viewing earlier results makes this exploratory tuning on reused seeds and tasks.

## Ranking of ridge-1e-6 cells

| Warm-up | Multiplier | Mean SubOpt | Seed SE |
|---:|---:|---:|---:|
| 25 | 0.001 | 0.0001289 | 0.0000249 |
| 0 | 0.001 | 0.0001298 | 0.0000187 |
| 200 | 0.001 | 0.0001447 | 0.0000379 |
| 100 | 0.3 | 0.0001740 | 0.0000448 |
| 0 | 0.03 | 0.0001760 | 0.0000485 |
| 100 | 0.001 | 0.0001877 | 0.0000422 |
| 50 | 0.1 | 0.0001979 | 0.0000618 |
| 100 | 0.003 | 0.0002015 | 0.0000648 |
| 50 | 0.001 | 0.0002055 | 0.0000390 |
| 100 | 0.1 | 0.0002113 | 0.0000350 |
| 25 | 0.3 | 0.0002118 | 0.0000571 |
| 200 | 0.1 | 0.0002180 | 0.0000617 |
| 50 | 0.003 | 0.0002286 | 0.0000496 |
| 100 | 0.03 | 0.0002367 | 0.0000464 |
| 100 | 1 | 0.0002371 | 0.0000932 |
| 25 | 0.03 | 0.0002441 | 0.0000958 |
| 25 | 0.01 | 0.0002453 | 0.0000557 |
| 200 | 0.003 | 0.0002459 | 0.0000625 |
| 200 | 0.01 | 0.0002543 | 0.0000780 |
| 25 | 1 | 0.0002736 | 0.0001150 |
| 25 | 0.1 | 0.0002833 | 0.0000589 |
| 100 | 0.01 | 0.0003287 | 0.0001370 |
| 0 | 0.1 | 0.0003482 | 0.0001518 |
| 25 | 0.003 | 0.0004135 | 0.0001506 |

The comparison plot gives paired Student-t 95% intervals over eight learning seeds. They are unadjusted and evaluated on the data used for selection. They do not account for uncertainty over fresh task sets or for choosing a winner among many candidates. Different information multipliers also prevent isolating a causal warm-up effect from a protocol comparison.

## Verification and cost

All 96 additional runs passed finite-value, simplex, coefficient-radius and evaluation-provenance checks. Hash verification confirms 175 protected files are unchanged. Saved fitted rewards match the training endpoints exactly. Each output stores its training source hash and task-model hash.

The two training processes took 4.4 and 15.1 minutes, largely concurrently. Evaluation took 14.9 minutes summed across waves, overlapping training. Completed results and existing task optima were reused.

## Files and regeneration

`results.json` contains the 35 cell values, per-seed scores and paired comparisons. `training/` and `evaluation/` contain the 12 additional configurations. The previous 12 ridge-1e-6 configurations remain in `../tuning-refinement-20260919/`. `PLAN.md` records the fixed candidate list, runtime estimate and pilot check.

From the repository root:

```sh
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-complete-20260919/summarize.py
```
