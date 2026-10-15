# Efficiency measurements — 19 September 2026

Branch: `codex/profile-efficiency`. Baseline: `99f4ab82042e7a504cb095f34afeb56a3a9e066e`.

Changes: omit unused batch gradients during backtracking and candidate ranking;
update progress heartbeats without rescanning all job-status files. Per-job durable
writes, sampling order, solver tolerances, and numerical expressions are retained.
The manifest now consistently shows the job that emitted the latest heartbeat.
Full status reconciliation remains on job start/completion/failure and run exit.

## Measurements

Sequential local measurements, same Python environment; no competing benchmark workers.
The existing component profiler used five repetitions and eight seeds. Selected end-to-end
training runs used cProfile, three warmed repetitions per implementation, alternating order.
These are short-workload measurements, not forecasts for full paper-scale runs.

| Measurement | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Batched policy selection, component median | 5.086 ms | 4.367 ms | 1.16x |
| Active IRL, 160 warm-up + 12 online samples | 50.581 ms | 41.700 ms | 1.21x |
| Four tuning trials | 87.379 ms | 75.455 ms | 1.16x |
| Eight progress updates, 450 running job records | 78.744 ms | 4.997 ms | 15.76x |
| Toy CLI experiment, warmed median | 2.623 s | 2.421 s | 1.08x |
| Sparse-information CLI experiment, warmed median | 3.039 s | 2.884 s | 1.05x |
| Lobeke CLI experiment, warmed median | 7.454 s | 7.220 s | 1.03x |

All component measurements, including unchanged MLE, warm-up and grid operations,
are in `before-profile.json` and `after-profile.json`. Small changes in untouched
components illustrate timing noise. CLI timings include interpreter startup,
configuration, evaluation, durable writes and profiling overhead. The small Lobeke
wall-time gain should not be treated as a precise general speedup estimate.

The initial Lobeke baseline took 18.4 seconds due to cold imports; it is excluded
from the warmed timing comparison. In warmed repetition 2, information-objective
calls took 203 -> 135 ms and leader-objective calls took 108 -> 74 ms under cProfile.
Manifest scans fell from 110 to 38. Constrained MLE/SLSQP remains the dominant cost.

## Generated-data comparison

Each paired repetition contains 54 jobs (18 per application):

- Toy: three seeds, 64 samples, 24-sample warm-up, both warm-up modes, all three methods.
- Sparse information: three seeds, 48 samples, 16-sample warm-up, both modes, all three methods.
- Lobeke: three seeds, 32 samples, no warm-up, all six methods, including both OLS estimators.

Evaluation interval: eight observations. Original numerical recipes and seeds are retained.
Every pair matched bit-for-bit across all 813 saved NPZ arrays (dtype, shape, bytes),
including trajectories, actions, estimates, diagnostics and evaluation metrics. All
non-timing summary fields matched. This passed for the initial run and all three
warmed repetitions. Repetitions reuse seeds; they are timing replicates, not additional
independent statistical trials. Timestamps, run identities and timing fields differ by design.

A separate comparison of 24 complete core result groups also matched exactly:
scalar active-learning runs with/without warm-up, Newton and SLSQP MLE, both batched
policy optimizers, zero random starts/prefilter iterations, and 513-action vertex screening.
This is empirical equivalence on selected cases, not a proof over every possible input.

## Validation

`make check`: Ruff formatting/lint, **206 tests passed**, and all **10 figure checks passed**.
The two additional tests check exact value-only outputs and durable heartbeat consistency
without unrelated job-status reads. Production code grows by 26 lines; these changes
remove repeated work rather than reducing source line count. No new classes or caches.

## Reproduce / retrieve

`run_pairs.py` runs selected bounded CLI experiments and records commands and cProfile output.
`compare_runs.py BEFORE_PHASE AFTER_PHASE` compares their result arrays and summaries.
`progress_bench.py OUTPUT_JSON` measures durable updates at 18 and 450 job records.
`repeat_pairs.sh` records the alternating schedule used for warmed timings.
Scripts expect execution from the repository root and write to `artifacts/efficiency-review/`.
Raw runs, NPZ results, logs and `.prof` files are preserved locally in that directory.

The baseline modules were copied before editing to `/private/tmp/mairl-efficiency-baseline/src/`;
`conf`, `experiments`, `reproducibility` and `artifacts` there point to the checkout.
For reruns, reconstruct the baseline `src/mairl` and `src/mairl_exp` from the recorded
commit in that directory. Set `PYTHONPATH` to the baseline `src` for before-runs and
to the working checkout's `src` for after-runs. Baseline-replay provenance captures
the working checkout, so the recorded baseline revision and PYTHONPATH override here
are necessary to interpret these benchmark runs; they are not publication data.
The initial before-runs were collected directly on the unmodified baseline branch.

`core_compare.py OUTPUT.pkl` captures the complete core result groups; run against each
source version and compare the unpickled payloads for equality.

The broader profile command was:

```sh
uv run mairl-profile --repeats 5 --batch-seeds 8 --json REPORT.json --cprofile RUN.prof
```

Preallocation was deferred: the main batched execution path already preallocates its
arrays, while scalar append cost was below one millisecond in the profiled active run.
Projected-feature caching, OLS-factorization caching, and changes to tensor gathers or
sampling order were not made.
