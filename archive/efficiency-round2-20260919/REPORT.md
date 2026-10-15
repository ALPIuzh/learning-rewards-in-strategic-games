# Efficiency pass 2 — 19 September 2026

Branch: `codex/profile-efficiency`. Baseline: `1f2d7ae` (the first efficiency pass).

## Retained changes

- Compute scalar follower rewards once per local policy optimization instead of once
  per objective evaluation. Keep the original scalar contraction expressions; do not
  substitute batched projections with potentially different floating-point results.
- Remove path planning from the scalar likelihood's `nb,nbd->nd` contraction.
- Keep the manifest in the locked, single-writer `RunStore`; continue writing every
  heartbeat durably. `progress()` returns a deep copy so callers cannot mutate store state.

Net production change: **+12 lines**, no solver, sampling, tolerance or data-schema changes.

## Measurements

Leader-optimization microbenchmarks use seven timed repetitions after a warm-up, with
fixed inputs and random seeds. For 3 and 23 leader actions, scalar optimizations improve
from 2.811 -> 2.410 ms and 34.270 -> 29.715 ms (1.17x and 1.15x). Batched optimization
improves from 2.412 -> 2.237 ms and 7.339 -> 6.727 ms (1.08x and 1.09x).
All ten full result records match exactly, including optimization diagnostics.

Eight durable progress updates across 450 job records improve from 5.374 -> 4.042 ms
(1.33x); across 18 records, 3.359 -> 3.044 ms. This is additional to the first pass's
removal of all-job status scans. Measurements retain real atomic writes and fsync.

The scalar expected-feature contraction was exact across 12 shape/layout cases
(C-contiguous, Fortran-contiguous and strided inputs), with a median 3.04x microbenchmark
speedup. Whole-MLE gains are much smaller. Full seven-repetition component profiles are
recorded, including unchanged components; variation in those components illustrates noise.

Warmed CLI experiments were repeated three times in alternating before/after order,
with cProfile enabled and one benchmark process at a time:

| Workload | Before median | After median | Ratio |
| --- | ---: | ---: | ---: |
| toy | 2.587 s | 2.694 s | 0.96x |
| sparse | 3.055 s | 2.899 s | 1.05x |
| lobeke | 7.705 s | 7.187 s | 1.07x |

Whole-run timings include imports, configuration, evaluation, I/O and profiling overhead.
Do not interpret these short, noisy runs as precise predictions for paper-scale training.
In particular, the profiled toy timings show **no improvement**. A follow-up with
cProfile disabled (five paired runs, same inputs) measured 1.423 -> 1.382 seconds median
(1.03x), with exact data agreement in all five pairs. This does not establish a precise
whole-run gain, but gives no evidence of a consistent toy regression. Targeted operation
benchmarks are clearer than end-to-end timings. The initial cold Lobeke run is excluded.
The separate 128-observation Lobeke pair took 15.25 -> 14.29 s; one pair is a correctness
check and indicative timing, not a reliable speedup estimate.

## Data verification

All three warmed before/after comparisons matched **813/813 NPZ arrays across 54 jobs**:
18 toy, 18 sparse information and 18 Lobeke jobs. These use the same bounded settings
as the first pass: 3 seeds, 64/48/32 observations, eight-observation evaluation intervals,
both warm-up modes for toy/sparse, and all six methods for Lobeke. Repeated runs reuse
seeds and are timing replicates, not additional independent statistical trials.

An additional **12 Lobeke jobs at 128 observations**, two seeds, with and without a
32-observation warm-up, cover active/fixed-random/random-each-round methods. All **196
arrays** match exactly. Non-timing summary fields also match in every comparison.
The comparisons check dtype, shape and raw bytes, including policies, sampled actions,
coefficient trajectories, diagnostics and evaluation metrics.

All 24 complete core result groups from the original pre-simplification harness also
match exactly, including zero-start/prefilter cases and 513-action vertex screening.
This is evidence on selected cases, not proof of equivalence for all inputs.

## Rejected / deferred

- Separating scalar MLE objective and gradient callbacks preserved data but made SLSQP
  component benchmarks take approximately 1.24x the runtime. Reverted; its
  patch, profiles and comparison records are retained.
- Removing planning from four other batched contractions changed floating-point bytes
  for some shapes/layouts. Not applied; the full 60-case probe is saved.
- Incremental feature tensors and OLS factorization caches were not implemented: in the
  short Lobeke profile, likelihood preparation plus cumulative Fisher evaluation took
  about 10 ms and OLS SVD solves about 1 ms, versus ~7 seconds total. These measurements
  do not establish their cost at full paper scale. Their interface/state complexity is
  not justified by this bounded evidence.
- Cholesky gate changes, altered sampling, solver tolerances, true-reward initialization,
  and figure-reader caching were not introduced.

## Validation

`make check`: **206 tests passed**, Ruff lint/format checks passed, and all **10 figure checks
passed**. Existing interruption/resume tests cover the retained manifest; the progress
test now also checks that mutating a returned progress dictionary cannot alter store state.

## Reproduce / retrieve

Raw runs, NPZ files, logs and `.prof` profiles remain local under
`artifacts/efficiency-round2/`. This directory preserves the scripts and JSON measurements.
Run scripts from the repository root; they write to that local artifacts directory.
`run_pairs.py PHASE [SCENARIO ...]` selects toy/sparse/lobeke by default, or `lobeke_long`
for the 128-observation check. `compare_runs.py BEFORE_PHASE AFTER_PHASE` verifies arrays
and summaries. `leader_bench.py PHASE`, `progress_bench.py OUTPUT_JSON` and `probe_einsum.py`
record the targeted measurements. `repeat_pairs.sh` is the alternating timing schedule.
Set `MAIRL_BENCH_PROFILE=0` for unprofiled runs; the recorded command in each run JSON
shows whether cProfile was used.

Baseline packages were copied before editing into
`/private/tmp/mairl-efficiency-round2-baseline/src/`, with repository `conf`, `experiments`,
`reproducibility` and `artifacts` linked beside it. Reconstruct that snapshot from `1f2d7ae`
for future runs. `PYTHONPATH` selects baseline or current `src`. Replay provenance records
the checkout, so interpret it together with this baseline revision and PYTHONPATH override;
these are benchmark artifacts, not publication data.

The full component profiler command was `uv run mairl-profile --repeats 7 --batch-seeds 8
--json REPORT.json --cprofile RUN.prof`. The complete core snapshot script is preserved in
`../efficiency-review-20260919/core_compare.py`.
