# Uniform 32-seed Lobéké tuning heatmap

All 35 cells use the same estimator, numerical settings, 32 fresh learning seeds and 100 evaluation tasks. The lowest observed mean is warm-up 50, information multiplier 0.03: 0.0001568. This is a tuning-set ranking, not confirmation that this setting is uniquely optimal.

## Design

Warm-up observations: 0, 25, 50, 100, 200. Information multipliers lambda / eta squared: .001, .003, .01, .03, .1, .3, 1. MLE ridge is 1e-6 in every cell and is distinct from the policy-selection information multiplier. Each run uses 1,600 total observations and known inverse temperature from the unchanged source configuration.

Learning seeds are 2026094000–2026094031, in four batches of eight, with the same batching across every cell. None of these training seeds entered the previous eight-seed grid. The fixed 100 normalized transfer tasks (seed 9400, candidate pool 2000) and numerical reference optima are reused exactly. Every fitted reward is evaluated on every task with 64 optimizer starts and tolerance 1e-12. This gives 1,120 training runs and 112,000 task evaluations.

The grid and 32-seed budget were frozen before reading fresh scores. The two pilot batches contribute to the final data. No cell or seed was discarded based on performance. Historical ridge-zero cells are not mixed into these figures. Earlier data and figures remain unchanged. The manuscript and paper defaults were not modified.

## Precision and interpretation

The main heatmap shows actual cell means, without smoothing, interpolation or score clipping. The companion heatmap shows standard errors over the 32 independent learning seeds. The 100 transfer tasks are held fixed; these errors do not represent uncertainty over newly sampled task suites. Table intervals use Student-t with 31 degrees of freedom and are unadjusted for selecting among 35 configurations.

The first-versus-second 16-seed rank correlation is 0.079. The half-sample winners are [{'warmup': 50, 'multiplier': 0.003}, {'warmup': 100, 'multiplier': 0.003}]. This low correlation indicates that the fine ranking remains unstable. This is a descriptive stability diagnostic, not an additional selection rule.

The best positive warm-up is {'warmup': 50, 'multiplier': 0.03}. Compared with no warm-up at the same information multiplier, its mean difference is -0.00009410, with an unadjusted paired 95% interval [-0.00018333, -0.00000487]. This comparison was selected using these data and must not be presented as independent confirmation.

More seeds reduce sampling uncertainty; they do not imply that small differences in heatmap shading are meaningful. A reliable claim that the selected setting improves over alternatives needs fresh confirmation data. A clean figure should not hide that limitation.

## Cell statistics

| Warm-up | Multiplier | Mean × 10⁴ | SE × 10⁴ | 95% interval × 10⁴ |
|---:|---:|---:|---:|---:|
| 50 | 0.03 | 1.568 | 0.294 | [0.969, 2.168] |
| 200 | 1 | 1.619 | 0.254 | [1.101, 2.137] |
| 0 | 0.1 | 1.660 | 0.294 | [1.060, 2.259] |
| 0 | 0.003 | 1.720 | 0.284 | [1.140, 2.299] |
| 50 | 0.003 | 1.732 | 0.373 | [0.970, 2.494] |
| 50 | 0.3 | 1.752 | 0.454 | [0.826, 2.679] |
| 50 | 1 | 1.765 | 0.286 | [1.182, 2.348] |
| 100 | 0.3 | 1.778 | 0.265 | [1.237, 2.318] |
| 0 | 1 | 1.804 | 0.318 | [1.154, 2.453] |
| 100 | 0.001 | 1.863 | 0.299 | [1.254, 2.471] |
| 200 | 0.03 | 1.895 | 0.311 | [1.260, 2.529] |
| 200 | 0.001 | 1.906 | 0.307 | [1.279, 2.533] |
| 25 | 0.01 | 1.966 | 0.298 | [1.358, 2.574] |
| 200 | 0.003 | 1.997 | 0.500 | [0.977, 3.017] |
| 25 | 0.003 | 2.063 | 0.423 | [1.199, 2.926] |
| 50 | 0.001 | 2.140 | 0.302 | [1.525, 2.756] |
| 200 | 0.01 | 2.144 | 0.399 | [1.330, 2.959] |
| 50 | 0.1 | 2.150 | 0.372 | [1.392, 2.908] |
| 25 | 0.001 | 2.165 | 0.447 | [1.255, 3.076] |
| 100 | 1 | 2.167 | 0.425 | [1.300, 3.033] |
| 25 | 0.3 | 2.192 | 0.460 | [1.253, 3.130] |
| 200 | 0.3 | 2.202 | 0.482 | [1.220, 3.184] |
| 50 | 0.01 | 2.270 | 0.464 | [1.323, 3.218] |
| 100 | 0.003 | 2.273 | 0.576 | [1.099, 3.448] |
| 25 | 0.1 | 2.278 | 0.406 | [1.449, 3.106] |
| 0 | 0.3 | 2.303 | 0.377 | [1.533, 3.073] |
| 25 | 1 | 2.379 | 0.399 | [1.565, 3.193] |
| 100 | 0.01 | 2.437 | 0.378 | [1.666, 3.208] |
| 0 | 0.03 | 2.509 | 0.316 | [1.864, 3.154] |
| 100 | 0.03 | 2.669 | 0.696 | [1.250, 4.088] |
| 0 | 0.01 | 2.858 | 0.440 | [1.959, 3.756] |
| 100 | 0.1 | 2.879 | 0.498 | [1.865, 3.894] |
| 200 | 0.1 | 2.988 | 0.436 | [2.099, 3.877] |
| 0 | 0.001 | 3.103 | 0.717 | [1.642, 4.565] |
| 25 | 0.03 | 3.195 | 0.911 | [1.338, 5.052] |

## Validation and cost

All 1120 training endpoints and 112000 evaluations passed finite-value, policy-simplex, coefficient-radius, MLE-residual and source-link checks. Maximum MLE residual across saved diagnostics is 1e-06; maximum simplex error is 8.88e-16. The 316 protected prior files are byte-identical. Task data and reference optima match the earlier files byte-for-byte. Rechecking the highest-loss case in every cell with 256 starts changed losses by at most 4.75e-09, fitted objectives by 3.44e-15, and reference optima by 2.22e-16.

The full scheduler took 88.6 minutes using 8 single-threaded worker processes, reusing the completed pilots. Per-batch timings, source hashes, software commit and evaluation hashes are recorded under `jobs/`. Raw training trajectories and per-task evaluation scores are retained under `training/` and `evaluation/`.

## Reproduce the figures

From the repository root, without retraining:

```sh
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-32seeds-20260919/validate.py
PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache .venv/bin/python output/tuning-32seeds-20260919/summarize.py
```

`figure-data.npz` contains every task score in a compact file. Copy it with `summarize.py` to regenerate both figures using only NumPy, SciPy and Matplotlib. This was verified in an isolated folder with a fresh Matplotlib cache: both PNG files and the statistics JSON were byte-identical. `results.json` contains every per-seed score. `heatmap.pdf` is the vector main figure; `uncertainty.pdf` is its companion. PNG and SVG versions are also included. `caption.tex` provides a short factual caption without a claim of optimality.
