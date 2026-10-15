# Warm-up and baseline investigation

Branch: `codex/warmup-baselines-audit`

Source paper: `private paper draft (not distributed)`

Source TeX: `private paper TeX (not distributed)`

## Preservation and execution rules

- Preserve all pre-existing data and artifacts.
- Never overwrite experimental outputs; use new, timestamped or purpose-specific paths.
- Validate hypotheses with small, ETA-bounded runs before scaling.
- Record commands, configurations, seeds, runtimes, and artifact paths.
- Treat implementation, experiment design, tuning, plotting, and metric semantics as separate possible causes.

## Investigation log

### 2026-08-11: setup

- The original worktree was on `codex/public-release-readiness` with three unrelated modified files.
- Created isolated worktree `/private/tmp/mairl-warmup-baselines` at commit `f6f9984` on branch `codex/warmup-baselines-audit`.
- No pre-existing file or result has been modified or deleted.

## Findings

### 1. Why warm-up does not improve Active IRL

The implementation retains warm-up samples in both the MLE and accumulated
Fisher information, and the total sample budget is held fixed. Warm-up is not
silently discarded. It also strongly improves the random baselines at 1,600
samples:

| Method | Warm-up | No warm-up |
|---|---:|---:|
| Fixed random, target SubOpt | `1.348e-3` | `2.381e-3` |
| Online random, target SubOpt | `1.082e-3` | `2.935e-3` |

The Active method is different: at `omega_hat=0`, its online rule already
greedily maximizes Fisher gain with `lambda I`. It therefore performs a
sequential, adaptive D-optimal initialization by itself. At the 200-sample
boundary, source SubOpt is almost identical: `1.152e-3` with explicit WARM and
`1.177e-3` without it. Explicit WARM has a better Fisher certificate after 200
samples (mean true-information eigenvalues `[1.42, 14.37, 29.57]`, versus
`[0.47, 13.37, 32.69]` for no-warm Active), but it consumes 200 rounds that
could otherwise adapt to the observed reward direction. The theorem's WARM
phase is certificate/safety insurance, not an expected average-performance
accelerator.

The final plotted Active curves are not a causal warm-up ablation: warm Active
uses `lambda=12 eta^2`, while no-warm Active uses `3 eta^2`. In the complete
matched-lambda 20-seed grid, the paired `K0=200 minus K0=0` downstream effect
has a 95% interval crossing zero for every one of the ten lambda values. At
lambda 12 the effect is `-3.86e-5`, 95% CI `[-1.08e-4, 3.10e-5]`; at lambda 3
it is `+1.67e-5`, 95% CI `[-4.71e-5, 8.05e-5]`.

The old executor also used one stateful RNG for warm-up samples, optimizer
starts, online policies, and follower actions. Toggling warm-up therefore
shifted every subsequent draw and weakened paired inference. The implementation
now uses purpose-separated streams keyed by seed and absolute sample index, and
regression tests verify that warm/no-warm random baselines use identical online
policies and common follower draws over their overlapping sample indices.

### 2. Why fixed-expert MLE and follower-only OLS are bad

This is primarily an information-geometry limitation, not a solver failure.
The source joint-QRE expert leader is almost uniform (probabilities
`0.04289..0.04654`). Its per-sample follower Fisher eigenvalues are
`[4.11e-10, 4.90e-7, 1.494e-1]`, condition number `3.63e8`. Even after 1,600
samples, the information eigenvalues are only `[6.58e-7, 7.84e-4, 239]`.
The selected MLE ridge `0.1` dominates both weak directions: its local
shrinkage factors are approximately `[6.6e-6, 0.0078, 0.9996]`. It stabilizes
source deployment but structurally biases transfer. The final mean estimate is
`[0.026, 0.962, 0.081]` instead of `[0, 1, -0.667]`. Newton residuals are small
and final fits take two iterations, so this is not optimizer nonconvergence.

Follower-only OLS is algebraically full rank but effectively unidentified. With
the historical reference action 0 its singular values are
`[1.039, 1.50e-3, 5.01e-5]`, condition `20,746`; all 50 final estimates hit the
radius constraint. Feeding exact population probabilities recovers the truth
to `1.5e-11`, which rules out an equation implementation bug. Pseudocount sweeps
do not rescue it, and all actions are observed by 1,600 samples. This is noise
amplification by a nearly singular contrast system.

The frozen config comment claimed conditions about `13,686/174` but actually
set both references to 0. Reference 22 minimizes the follower contrast condition
and produces the documented numbers. Re-fitting the same samples improves
follower-only target SubOpt from `3.046e-2` to `2.122e-2` and Fisher error from
`0.872` to `0.712`, but 100% of estimates still hit the ball. The config now
uses reference 22; this removes an avoidable numerical penalty without hiding
the genuine fixed-policy identification failure.

The scientific conclusion is that a single near-uniform expert policy does not
excite the reward directions needed for general recovery. Ridge/reference
tuning can trade variance for bias, but cannot replace intervention over leader
policies.

### 3. Why the historical zero-sum OLS curve looked better in transfer

There was a serious executed-protocol/manuscript mismatch. The historical
flagship study called `fit_liao_zero_sum_ols`, estimated only follower
frequencies, and inserted `expert.max_policy` exactly into the leader equation
block. It sampled no leader actions. All three fixed-expert methods used the
same byte-identical follower action arrays. Thus the curve did **not** use
"twice as many samples"; it used a noiseless leader marginal, effectively an
infinite-information oracle for that block.

The current study now samples one expert leader action per follower action and
calls `fit_liao_zero_sum_paired_actions`. It stores `leader_actions` and records
the observed action counts and sampling protocol in job metadata. On the same
50 retained follower trajectories, with independently sampled leader actions
and stable reference 22:

| Protocol at 1,600 follower actions | Target SubOpt | Target Fisher | Coefficient L2 | Boundary |
|---|---:|---:|---:|---:|
| Legacy exact leader distribution, ref 0 | `7.03e-5` | `0.0632` | `0.298` | 0% |
| Exact leader distribution, ref 22 | `9.26e-5` | `0.0322` | `0.212` | 0% |
| Corrected paired actions, ref 22 | `2.15e-3` | `0.209` | `1.101` | 40% |
| Follower-only, ref 22 | `2.12e-2` | `0.712` | `1.738` | 100% |

Corrected paired zero-sum OLS averaged transfer SubOpt is `4.02e-4`, 95% CI
`[3.41e-4, 4.64e-4]`, versus `7.73e-5` for Active without warm-up. The paired
corrected-minus-Active difference is `+3.25e-4`, 95% CI
`[2.44e-4, 4.06e-4]`; corrected OLS wins only 7/50 seeds. Its averaged transfer
Fisher error is `0.161`, versus `0.0358` for Active.

Even before correcting the oracle advantage, the claimed mean transfer edge
over Active without warm-up was not significant: legacy zero-sum minus Active
was `-7.31e-6`, 95% CI `[-5.14e-5, 3.68e-5]`, and legacy zero-sum won only
19/50 paired seeds. Its median was worse and its averaged Fisher error was more
than twice Active's. The small mean advantage came from fewer upper-tail
decision losses and nonlinear deployment geometry, not uniformly better reward
estimation. Task averaging itself is correct (ten fixed tasks are averaged
within seed before the 50-seed interval), but the statement that zero-sum OLS
is generally better at averaged transfer is unsupported.

### Additional reproducibility issues fixed

- Transfer caches previously keyed only on estimated coefficients and task
  reward. They now include the problem, true reward, numerical config, code
  implementation hash, and schema version.
- Future final exports retain complete taskwise transfer arrays instead of only
  averages, so task heterogeneity and paired effects are auditable.
- Transfer SubOpt previously clipped a deployed value above the computed "true"
  optimum to zero. It now raises on a material negative gap rather than silently
  converting an optimizer failure into perfect performance.
- The external manuscript copy contains stale warm-up-selection wording and a
  TODO about discrepant hyperparameters. The tracked paper specification is the
  safer source; captions now state the paired leader-sampling cost explicitly.

## Experiment ledger

- Read-only audit of protected run `a4b1b2a1cdf886dd` (50 seeds, 450 jobs) and
  tracked `final_evaluation/data.npz`; no raw result was changed.
- `analyze_final.py`: paired 50-seed final metrics and complete matched-lambda
  grid audit; output `final_analysis.txt`.
- `geometry_diagnostics.py`: exact Fisher/contrast singular values, WARM design,
  and 200-sample information spectra; output `geometry_analysis.txt`.
- `recompute_ols_protocols.py`: final-checkpoint paired-action counterfactual on
  50 retained follower trajectories and all ten protected transfer tasks;
  runtime 19.4 seconds; immutable output under `corrected_ols_reanalysis/`.
- `recompute_ols_curves.py`: all 40 source checkpoints for paired zero-sum and
  stable-reference follower-only OLS; runtime 22.5 seconds; immutable output
  under `corrected_ols_curves_v1/`.
- `render_mechanism_figure.py`: matched-lambda warm-up, Fisher spectrum, transfer
  ECDF, and exact-vs-paired zero-sum task effects; PNG/PDF/JSON under `figures/`.
- `render_corrected_ols_figure.py`: corrected source SubOpt and Fisher curves;
  PNG/PDF/JSON under `figures/`. Renderers force the non-GUI Agg backend.
- Focused regression suite: 64 tests passed after RNG and paired-action changes.
- Canonical publication export: 350 unaffected protected traces plus 100
  corrected OLS traces under `reproducibility/data/lobeke/final_evaluation_corrected/`;
  all main/appendix TikZ fragments and fixed-expert policy-map assets were
  regenerated from this export. The original export and all prior generated
  fragments are preserved for auditability.
- Existing artifacts and data were never deleted or overwritten.

## Final conclusions

1. WARM works and Active already self-warms adaptively, but the matched
   follow-up below shows that explicit warm-up is not always redundant: it
   improves Target SubOpt at the lower tested information regularization. Its
   transfer effect remains inconclusive.
2. The original fixed-expert MLE and follower-only OLS fail because the chosen
   joint-QRE expert has catastrophic information geometry. MLE is repaired by
   the Stackelberg expert below; follower-only OLS remains structurally weak
   because its equilibrium contrast system is nearly singular.
3. The historical zero-sum result is not a fair sampled baseline. Its apparent
   transfer strength comes from the exact leader-distribution oracle and, even
   then, is not significant against no-warm Active on averaged transfer. The
   paired-action correction removes the apparent advantage.
4. Use the native paired-action study and ref-22 config for any new paper run.
   The canonical plots use the all-method native rerun; prior composite exports
   remain only as preserved audit evidence.

## 2026-08-12 native expert-design follow-up

Branch: `codex/expert-design-audit`

### Fixed-expert MLE redesign

The poor MLE curve was caused by using the joint-QRE leader as its expert even
though MLE does not require the QRE equilibrium equations. That leader is nearly
uniform and has per-sample Fisher condition number `3.63e8`. The primary MLE
baseline now uses the source Stackelberg-optimal leader instead. Its policy has
three nonzero actions with probabilities `[0.1303, 0.1432, 0.7265]`, sacrifices
no source value, and has Fisher eigenvalues
`[7.89e-5, 3.21e-2, 1.86e-1]` (condition number about `2,356`). OLS remains on
the joint-QRE leader because both OLS estimating equations require it.

Ridge tuning used source data only: 64 selection seeds (`10000..10063`), 64
confirmation seeds (`20000..20063`), and the grid
`[1e-4, 1e-3, 1e-2, .03, .05, .1, .2, .3, .5, 1, 2, 3, 10]` at 1,600
samples. Ridge 2 minimized the selection mean; the frozen one-standard-error
rule retained 1, 2, and 3 and selected the smallest, ridge 1. On confirmation,
ridge 1 had mean source SubOpt `4.33e-4`, versus `8.57e-4` for `1e-4`; the
paired 95% interval was `[-6.68e-4, -1.79e-4]`. The full per-seed record is
`reproducibility/data/lobeke/fixed_expert_mle_stackelberg_tuning.json`.

In the definitive 50-seed native run, Stackelberg-expert MLE reaches target
SubOpt `2.53e-4 +/- 1.52e-4` and target Fisher error
`0.0285 +/- 0.0057`. It is statistically indistinguishable from no-warm Active
on target SubOpt (paired MLE-minus-Active interval
`[-1.52e-4, 1.42e-4]`). Averaged-transfer SubOpt is `7.94e-5 +/- 1.91e-5`,
slightly worse than no-warm Active's `5.17e-5 +/- 1.68e-5`; the paired
difference interval is `[5.95e-6, 4.94e-5]`. A single source-optimal design is
therefore now a strong source baseline without being made artificially
transfer-optimal using the unknown true reward.

### Native sampled OLS results

The definitive paired-action zero-sum curve is not better than Active:

| Protocol at 1,600 follower actions | Target SubOpt | Avg. transfer SubOpt | Fisher | Boundary |
|---|---:|---:|---:|---:|
| Active, no warm-up | `2.58e-4` | `5.17e-5` | `0.0325` | 0% |
| Stackelberg-expert MLE | `2.53e-4` | `7.94e-5` | `0.0285` | 0% |
| Paired zero-sum OLS | `3.06e-3` | `4.41e-4` | `0.210` | 46% |
| Follower-only QRE OLS | `2.78e-2` | `7.55e-4` | `0.892` | 100% |

Paired zero-sum minus no-warm Active averaged-transfer SubOpt is
`3.89e-4`, 95% CI `[2.90e-4, 4.89e-4]`. Follower-only remains poor because
its contrast system is effectively singular; every final estimate reaches the
parameter ball.

### Protocol and numerical fixes

- The native export contains 99 arrays: all nine methods, all target metrics,
  all ten taskwise transfer arrays, averages, coefficient error, and the
  simulation-only self-concordance distance. It comes from one 450-job run at
  commit `0046334`, run ID `8c4ab05e9b2fe388`; no corrected or copied traces
  are mixed in.
- Transfer cache identity now includes array shape/dtype, trace seeds,
  checkpoints, task seed/index, problem/config context, and implementation
  hash. Cache hits must also pass shape, floating-dtype, and finiteness checks.
- Source SubOpt now rejects a materially negative raw gap instead of silently
  clipping an underestimated optimum to zero.
- The invalid warm-up sweep that mixed pilot/final seeds and lambda values is
  retained but removed from reproduction targets and requires an explicit
  acknowledgement flag.
- Random baselines now collect the identical random trajectory and solve MLE
  only at reported checkpoints. A 4-seed, 400-sample all-method preflight fell
  from 115.0 to 27.5 seconds; sampled actions/policies were byte-identical and
  maximum metric change was `4.6e-7`, below solver tolerance.
- One follower-OLS seed exposed an absolute root-tolerance failure in the
  ball-constrained secular solve. The solver now retries at machine precision
  only when the norm postcondition fails. All 2,000 final seed/checkpoint fits
  pass; estimates agree with an always-ultratight solve within `2.6e-7`.
- The appendix now shows the self-concordance certificate distance and
  coefficient error alongside target/transfer SubOpt and Fisher error. At the
  last checkpoint, the certificate holds for 98% of Stackelberg-expert MLE,
  96% of no-warm Active, 90% of warm Active, 42% of paired zero-sum OLS, and
  0% of follower-only OLS seeds.

### Preserved artifacts

- Definitive native run: `artifacts/runs/guard-lobeke-paper/8c4ab05e9b2fe388`.
- Native derived data: `reproducibility/data/lobeke/final_evaluation_native/`.
- Failed run that exposed the OLS numerical issue:
  `artifacts/runs/guard-lobeke-paper/7cbb5891d30cf4b5` (424 complete, four
  failed, 22 pending); it was not resumed, edited, or deleted.
- Preflight runs `c3fe6128463e35e5`, `f38d07c32ed0a003`, and
  `a9700d2999e01521`, plus all temporary diagnostic scripts under
  `/private/tmp`, remain preserved.

### Matched warm-up/lambda factorial

Run `8513aac9cbe3189a` swaps the final Active multipliers on the first 20 final
seeds: warm-up uses `lambda/eta^2=3` and no-warm uses 12. It completed 40/40
jobs without failure in 547 seconds, with four-seed batches and single-threaded
numerical libraries. Combining it with the corresponding arms of native run
`8c4ab05e9b2fe388` gives a paired
`warm-up in {0,200} x lambda/eta^2 in {3,12}` factorial at 1,600 observations.

Paired warm-minus-no-warm final effects (negative favors warm-up):

| Metric | `lambda/eta^2=3` | `lambda/eta^2=12` |
|---|---:|---:|
| Target SubOpt | `-2.30e-4 [-4.10e-4, -5.04e-5]` | `-3.92e-5 [-1.41e-4, 6.22e-5]` |
| Target Fisher error | `-7.09e-3 [-1.86e-2, 4.44e-3]` | `-4.45e-3 [-1.59e-2, 6.95e-3]` |
| Averaged-transfer SubOpt | `-2.25e-5 [-6.10e-5, 1.59e-5]` | `3.94e-6 [-5.78e-5, 6.57e-5]` |
| Certificate distance | `-2.12e-2 [-8.80e-2, 4.57e-2]` | `-2.63e-2 [-1.24e-1, 7.16e-2]` |

This changes the original simple story. No-warm Active does perform a
reward-independent, sequential Fisher design from its zero estimate, which
explains why explicit WARM is often redundant. But at lower online
regularization the 200-sample offline design prevents early reward-adaptive
policies from narrowing exploration too soon and significantly improves the
source objective on these paired seeds. The result does not establish a
transfer benefit: both averaged-transfer intervals cross zero, conditional on
the fixed ten-task suite. The target interaction between multipliers is also
borderline rather than significant: `1.91e-4`, 95% CI
`[-1.14e-5, 3.93e-4]`.

Tracked arrays, run hashes, taskwise transfer values, and every paired effect
are in `reproducibility/data/lobeke/warmup_factorial_native/`. The raw run is
preserved append-only under
`artifacts/runs/guard-lobeke-paper/8513aac9cbe3189a`.

#### 50-seed extension

Because the 20-seed transfer intervals remained wide, a disjoint extension was
run only for seeds `30020..30049`. Run `2d6a6b2f2d58eebd` completed 60/60 jobs
without failure in 1,050 seconds. It was combined with the preserved first
20-seed result into a new, non-overwriting `warmup_factorial_native_v2` export.

The 50-seed paired warm-minus-no-warm effects are:

| Metric | `lambda/eta^2=3` | `lambda/eta^2=12` |
|---|---:|---:|
| Target SubOpt | `-1.10e-4 [-2.12e-4, -8.19e-6]` | `9.12e-6 [-5.81e-5, 7.63e-5]` |
| Target Fisher error | `-3.24e-3 [-9.36e-3, 2.89e-3]` | `-1.34e-3 [-6.99e-3, 4.31e-3]` |
| Averaged-transfer SubOpt | `-1.66e-6 [-2.50e-5, 2.17e-5]` | `-8.34e-6 [-5.69e-5, 4.02e-5]` |
| Certificate distance | `2.87e-3 [-3.96e-2, 4.54e-2]` | `-2.77e-2 [-1.06e-1, 5.09e-2]` |

The larger run preserves the qualitative conclusion while shrinking the source
effect: explicit warm-up helps Target SubOpt at multiplier 3, is neutral at 12,
and has no detected effect on the fixed-suite transfer average. The factorial
interaction in Target SubOpt remains borderline (`1.19e-4`, 95% CI
`[-6.91e-6, 2.45e-4]`), so the safe claim is effect heterogeneity consistent
with lambda dependence—not a proven statistical interaction.
