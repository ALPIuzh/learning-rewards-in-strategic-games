# Behavioral-transfer pilot findings

## Scope and safeguards

- This is a small exploratory pilot, not publication evidence.
- The two task metrics are fixed before evaluating any learned estimates:
  leader-policy total variation from the source optimum, and the downstream
  value gap of deploying the source-optimal leader.
- Candidate rewards use the existing structured Lobeke coverage form and are
  normalized to unit Frobenius norm. The dense extension evaluates every
  candidate, so its bucket means involve no within-bucket task selection.
- Training is deliberately limited to Active IRL with and without warm-up and
  fixed-expert MLE, eight seeds, 400 follower samples, and four evaluation
  checkpoints. Numerical libraries are restricted to one thread.
- Raw run outputs and pilot-derived artifacts are preserved under ignored
  `artifacts/` directories and are never overwritten by the pilot script.

## Findings

### Exact pilot protocol

- Branch: `codex/behavioral-transfer-pilot`.
- Training run: `34a5920bc1c298d1`, created from clean commit `96644af`.
- Training arms: Active IRL with warm-up, Active IRL without warm-up, and
  fixed-expert MLE. Each arm uses seeds `60000..60007`, 400 follower samples,
  and checkpoints `[100, 200, 300, 400]`. The warm-up budget is 100. The
  fixed-expert MLE arm still learns from 400 sampled follower actions per seed.
- Training took 82.1 seconds for 24 jobs. The numerical libraries were limited
  to one thread and the executor used batches of four seeds.
- Two independent downstream candidate pools use task seeds 63001 and 63002.
  Each contains 24 candidates. Three tasks per low/medium/high bucket were
  selected independently for TV and gap, and the union was evaluated using the
  same retained learned estimates. The corrected derived evaluations took 21.4
  and 19.7 seconds including task construction, evaluation, and plotting.
- The evaluator at commit `4726087` aligns all protocols on their common
  positive checkpoints and keeps numerically tied metric values in the same
  bucket.
- The dense extension reuses the same trained estimates and evaluates every
  reward in two independent 180-candidate pools (task seeds 64001 and 64002).
  The evaluations took 200.7 and 196.4 seconds, including oracle task
  construction, all three methods at all four checkpoints, diagnostics, and
  plotting. No algorithm was retrained.

### The two metrics are related but not interchangeable

- Spearman correlation between TV and gap is 0.753 and 0.719 in the two dense
  pools. Thus both measure meaningful source-to-task displacement, but one is
  not merely a rescaling of the other.
- Only three of the nine sampled bucket tasks overlap between TV and gap in
  each pool. Selection is uniform within each bucket and never observes any
  method result.
- The gap values are smooth and support exactly balanced 60/60/60 terciles.
  Their internal boundaries reproduce closely: `(0.00416, 0.00998)` and
  `(0.00446, 0.01032)` after unit-Frobenius reward normalization.
- TV's tie-aware internal boundaries are also stable: `(0.691, 0.871)` and
  `(0.712, 0.872)`. Bucket counts are not forced to equality when a tied TV
  value crosses an empirical tercile.

### TV has a real saturation issue in this game

- The source-optimal leader has support on only three of 23 actions. Many
  downstream optimal policies have support disjoint from those three actions.
  Every such policy is at TV distance one, regardless of its mixture weights
  or how consequential the behavior change is. This corrects the initial pilot
  interpretation: most saturated policies are mixed, not pure.
- TV is numerically one for 45/180 and 42/180 tasks in the dense pools. In the
  first pool, only 3/180 downstream optimal policies are actually pure; the
  remaining support sizes are 2--6.
- The first pilot implementation used naive empirical quantiles and could split
  mathematically tied `TV = 1` tasks because of floating-point noise. Those
  original artifacts are preserved. The corrected implementation is tie-aware,
  producing TV bucket sizes 8/7/9 and 8/9/7 rather than pretending every bucket
  has the same number of distinct distances.
- TV remains easy to explain and purely behavioral, but a future figure must
  report its saturation fraction. A categorical distinction such as
  same-support versus disjoint-support may also be more honest than treating
  all TV-one tasks as finely ordered.
- Re-optimizing 18 metric-boundary/extreme tasks six times each produced a
  maximum objective spread of `8.2e-12`, maximum pairwise policy TV of
  `3.28e-5`, and maximum TV from the originally stored policy of `4.46e-5`.
  Thus the observed TV saturation is not an optimizer-seed artifact in this
  probe.

### Dense algorithmic validation

The table averages the two dense pools and reports mean transfer SubOpt at 400
samples. Each dense pool contributes every one of its 180 rewards, with about
60 tasks per bucket. There are still only eight training seeds, so this remains
diagnostic rather than paper-ready.

| Metric | Method | Low | Medium | High | High / low |
|---|---|---:|---:|---:|---:|
| TV | Active, warm-up | 1.54e-4 | 2.37e-4 | 3.13e-4 | 2.03 |
| TV | Active, no warm-up | 1.78e-4 | 1.68e-4 | 2.19e-4 | 1.23 |
| TV | Fixed-expert MLE | 9.07e-5 | 7.59e-5 | 1.03e-4 | 1.13 |
| Gap | Active, warm-up | 1.62e-4 | 2.61e-4 | 2.70e-4 | 1.67 |
| Gap | Active, no warm-up | 1.72e-4 | 1.91e-4 | 1.95e-4 | 1.13 |
| Gap | Fixed-expert MLE | 8.75e-5 | 8.94e-5 | 8.85e-5 | 1.01 |

TV exposes three distinct levels for warm Active IRL. At 400 samples, the
paired high-minus-low differences are `1.55e-4` (95% Student-t interval
`[1.00e-4, 2.10e-4]`) and `1.63e-4` (`[1.06e-4, 2.20e-4]`) in the two pools;
all eight seed-pairs are positive in both pools. The adjacent TV contrasts are
also positive in all or all but one seed and their intervals exclude zero.

Gap has a different, reproducible shape: it separates the low bucket from the
other two, but medium and high are not distinguishable. In pool 2,
high-minus-medium is `-2.44e-6` with interval `[-3.83e-5, 3.35e-5]`. This is a
reason to retain both metrics rather than rename one as the unique difficulty.

For no-warm Active and fixed-expert MLE, every dense high-minus-low paired
interval crosses zero for both metrics. TV and gap quantify source-to-task
displacement, not how a particular estimator's coefficient error is oriented
relative to a task. Calling the rows **transfer-distance buckets** is more
accurate than calling them universal **difficulty buckets**.

### What happens inside a bucket

- The distributions overlap substantially, so a heatmap should be accompanied
  by task-level points or boxplots. In dense pool 64001, warm-Active TV medians
  are `1.46e-4`, `2.33e-4`, and `2.96e-4`, while the respective ranges are
  `[3.73e-5,4.22e-4]`, `[1.03e-4,4.58e-4]`, and
  `[1.28e-4,5.89e-4]`. The bulk shifts even though individual ranges overlap.
- Extremes do not explain away the cell means. Across both dense pools, the
  largest single task contributes 2.4--6.5% of a final cell sum; the worst 10%
  contribute 14.6--25.5%. No cell is a one-task result.
- Extremes often lie near the bucket boundary or are large on the other metric.
  For example, the worst warm-Active task in the low-gap bucket has gap
  `0.00402`, just below the `0.00416` boundary, and TV `0.870`. This explains
  some within-bucket spread but not the population-level shift.
- Oracle policy shape explains another large part of the spread. At 400
  samples, SubOpt's Spearman correlation with the oracle maximum action
  probability is `0.79/0.81` for no-warm Active and `0.78/0.79` for
  fixed-expert MLE in the two pools. Their direct TV and gap correlations are
  near zero. Peaked, low-entropy optimal policies are empirically harder for
  those two estimators in this reward family. This is descriptive, not yet a
  claim that peakedness causally defines difficulty.
- Warm Active is different: its final SubOpt correlates strongly with TV
  (`0.71/0.67`), moderately with gap (`0.54/0.51`), and only weakly with policy
  peakedness (`0.16/0.12`). Its method-specific Fisher error has correlation
  `0.72/0.75` with SubOpt. This supports an error-orientation explanation for
  why the same task distance affects the methods differently.
- Across-seed variability is larger than across-task variability, which is why
  paired seed contrasts are reported above. Task-only bootstrap bars would be
  too optimistic with eight training seeds.

At this short 400-sample horizon, the *configured* fixed-expert MLE has much
smaller mean coefficient error (`0.445` in L2 versus `1.178` for warm Active and
`0.989` for no-warm Active). It beats warm Active on 94.4% and 91.1% of task
means in the two dense pools, and no-warm Active on 100% and 99.4%. The ridge
audit below shows that this is not an intrinsic advantage of fixed collection:
the plotted estimators use very different regularization, and the apparent
fixed-expert dominance largely disappears when that is controlled.

Gap has a complementary caveat to TV. It captures the value consequence of a
behavior change, but that makes it closer to the downstream SubOpt quantity by
construction. Unit-Frobenius reward normalization removes arbitrary reward
scaling, but it does not remove task-specific policy sensitivity. This is why
the pilot supports retaining both views rather than selecting a winner.

### Fixed-expert dominance is mostly an MLE-tuning artifact

The paper configuration does not use the same estimator for the MLE-based
curves. Active IRL uses `numerics.mle.ridge = 1e-4`, whereas fixed-expert MLE
uses the separately protected-selected `fixed_expert_mle_ridge = 1`. This is a
10,000-fold regularization difference. The Active information multiplier
(`12 eta^2` with warm-up and `3 eta^2` without) is not an MLE ridge: it only
regularizes the information matrix used to select the next leader policy.

A fixed-design audit refit the exact same saved policies and follower actions
at ridges `[1e-4, 1e-2, 1e-1, 1, 3]`, then evaluated all 180 rewards in dense
pool 64001. It therefore isolates estimator regularization, although it does
not replay the counterfactual adaptive collection loop.

| MLE ridge | Active, warm-up | Active, no warm-up | Fixed-expert MLE |
|---:|---:|---:|---:|
| 1e-4 | 2.25e-4 | 1.82e-4 | 6.35e-4 |
| 1e-1 | 2.05e-4 | 1.78e-4 | 2.25e-4 |
| 1 | 9.73e-5 | 1.08e-4 | 8.72e-5 |
| 3 | 7.58e-5 | 7.88e-5 | 9.22e-5 |

These are final mean transfer SubOpt values at 400 samples. There is no
ridge-invariant fixed-expert advantage: Active is better at shared ridge `0.1`,
fixed-expert is slightly better at shared ridge `1`, and Active is better at
shared ridge `3`. The source-task criterion also prefers different ridges on
these eight diagnostic seeds (`0.1`, `3`, and `1`, respectively), so a fair
paper comparison needs an identical, disjoint tuning rule for every MLE arm.
Those post-hoc minimizers are diagnostic and are not proposed final settings.

The information geometry points in the opposite direction from the original
performance plot. At the true reward and 400 samples, the mean Fisher
eigenvalues under the originally saved policies are approximately
`[0.94, 28.42, 69.68]` for warm Active, `[0.88, 26.75, 65.66]` for no-warm
Active, and `[0.032, 12.86, 74.34]` for fixed-expert MLE. The corresponding
condition numbers are about `75`, `109`, and `2356`. Active therefore collects
far more informative data; fixed-expert does not win because its sampling
design has better Fisher geometry. Its strong ridge shrinks the weakly
identified directions that dominate coefficient and transfer error at this
short horizon.

Finally, a separate held-out run replayed the adaptive loop rather than merely
refitting it. It used new seeds `62000..62007`, the same ridge `1` for all three
MLE arms, 400 samples, and the same independently generated 180-task pool. The
final transfer means were `8.97e-5` for warm Active, `8.24e-5` for no-warm
Active, and `9.54e-5` for fixed-expert MLE. Fixed-minus-Active paired
differences were `+5.73e-6` and `+1.30e-5`; their seed-level 95% Student-t
interval half-widths were `7.45e-5` and `7.81e-5`. With only eight seeds, this
does not establish a small ordering, but it does independently reject the
large, everywhere-fixed advantage suggested by the unmatched plot.

The held-out source metrics tell the same nuanced story. Mean source SubOpt was
`2.44e-4` for warm Active, `5.52e-4` for no-warm Active, and `5.70e-4` for
fixed-expert MLE; coefficient L2 errors were `0.572`, `0.427`, and `0.474`.
Fixed-expert remains a legitimate and competitive sample-based baseline, but
the present evidence supports **mostly tuning artifact**, not an inherent
algorithmic advantage. A final experiment should tune the MLE ridge for every
method on the same protected selection/confirmation splits, freeze the choices,
and only then regenerate the 50-seed curves.

### Preserved artifacts

- Reduced raw run:
  `artifacts/runs/behavioral-transfer-pilot/34a5920bc1c298d1/`
- Corrected pool 63001:
  `artifacts/derived/behavioral-transfer-pilot/34a5920bc1c298d1-tasks63001-n24-k3-tie-aware/`
- Corrected pool 63002:
  `artifacts/derived/behavioral-transfer-pilot/34a5920bc1c298d1-tasks63002-n24-k3-tie-aware/`
- Dense pool 64001:
  `artifacts/derived/behavioral-transfer-pilot/dense-1cfb138-tasks64001-n180-all/`
- Dense pool 64002:
  `artifacts/derived/behavioral-transfer-pilot/dense-6def5a1-tasks64002-n180-all/`
- Fixed-design common-ridge audit:
  `artifacts/derived/behavioral-transfer-pilot/ridge-refit-4fec82e-n180/`
- Held-out matched-ridge raw run:
  `artifacts/runs/behavioral-transfer-ridge-matched/b5c55e3738e96e24/`
- Held-out matched-ridge dense evaluation:
  `artifacts/derived/behavioral-transfer-pilot/ridge-matched-b5c55e3738e96e24-tasks64001-n180-all/`
- The two pre-correction derived directories are intentionally retained beside
  them; no artifact was overwritten or deleted.

Each dense directory contains every candidate reward, exact oracle policy,
metric and bucket assignment, per-task/per-seed evaluation, heatmap, task-level
dispersion and scatter plots, paired seed contrasts, task-structure diagnostics,
and exact evaluator Git provenance. The first dense directory also contains the
18-task repeated-oracle stability probe. The raw run manifest records the clean
training commit, full resolved configuration, job status, hashes, and elapsed
time.

### Reproduction commands

Run commands assume this repository is the working directory and the numerical
thread-limit variables are set to one.

```bash
python experiments/guard/paper/run.py \
  experiment.name=behavioral-transfer-pilot \
  experiment.seed_start=60000 experiment.seed_count=8 \
  experiment.total_samples=400 experiment.warmup_budget=100 \
  experiment.evaluation_interval=100 \
  'experiment.methods=[active,fixed_expert_mle]' \
  'experiment.warmup_modes=[true,false]' \
  run.root=artifacts/runs run.seed_batch_size=4

python -m investigations.behavioral_transfer.pilot \
  --run-dir artifacts/runs/behavioral-transfer-pilot/34a5920bc1c298d1 \
  --output-dir artifacts/derived/behavioral-transfer-pilot/<new-empty-directory> \
  --candidate-count 180 --task-seed 64001 --evaluate-all

python -m investigations.behavioral_transfer.ridge_refit_audit \
  --run-dir artifacts/runs/behavioral-transfer-pilot/34a5920bc1c298d1 \
  --tasks-dir artifacts/derived/behavioral-transfer-pilot/dense-1cfb138-tasks64001-n180-all \
  --output-dir artifacts/derived/behavioral-transfer-pilot/<new-empty-ridge-directory>

python experiments/guard/paper/run.py \
  experiment.name=behavioral-transfer-ridge-matched \
  experiment.seed_start=62000 experiment.seed_count=8 \
  experiment.total_samples=400 experiment.warmup_budget=100 \
  experiment.evaluation_interval=100 \
  'experiment.methods=[active,fixed_expert_mle]' \
  'experiment.warmup_modes=[true,false]' \
  experiment.fixed_expert_mle_ridge=1 numerics.mle.ridge=1 \
  run.root=artifacts/runs run.seed_batch_size=4

python -m investigations.behavioral_transfer.pilot \
  --run-dir artifacts/runs/behavioral-transfer-ridge-matched/b5c55e3738e96e24 \
  --output-dir artifacts/derived/behavioral-transfer-pilot/<new-empty-matched-directory> \
  --candidate-count 180 --task-seed 64001 --evaluate-all
```

The derived command refuses to write into a nonempty directory.

## Small-budget estimator, transfer, and inverse-temperature audit

### Ridge must not be tuned in the flagship comparison

The paper defines both Active IRL and the fixed-policy MLE through the same
constrained estimator,

`argmax_{omega in Omega} sum_k log p_omega(b_k | mu_k)`,

with no coefficient penalty. The implementation instead supports

`sum_k log p_omega(b_k | mu_k) - ridge * ||omega||^2 / 2`.

A positive ridge is therefore not merely a solver tolerance. It defines a
zero-centered Gaussian prior and a different estimator. If the experimental
claim is about the algorithms in the paper, ridge must be fixed identically for
all MLE arms, preferably at exactly zero. Tuning it on target SubOpt is tuning a
task-favorable statistical prior, not tuning numerical stability.

This distinction is large in the present game. The Stackelberg expert's
per-sample Fisher eigenvalues are `[7.89e-5, 3.21e-2, 1.86e-1]`. At `K=40`,
its data curvatures are `[0.0032, 1.29, 7.43]`: ridge `1` overwhelms the weakest
direction by a factor of about 317. Even at `K=1600`, the weakest data
curvature is only `0.126`, so ridge `1` still dominates it by about eight.

The ridge was applied correctly by the code. Direct refits reproduced retained
Active (ridge `1e-4`) and fixed-expert (ridge `1`) estimates to below `1e-15`,
with verified KKT residuals. The error is in the scientific protocol, not in
configuration plumbing or the optimizer.

The earlier ridge searches remain useful as bias--variance mechanism
diagnostics, but they must not be described as fair algorithm tuning. The
original QRE expert selected ridge `0.1`; the replacement Stackelberg expert
selected ridge `1`; Active remained at `1e-4`. The resulting plot compares
different estimators. `reproducibility/FIGURES.md` also still describes the
superseded `0.1` selection and is stale.

The MLE configuration has now been corrected on this investigation branch to
permit ridge `0`, which the experiment class already advertised as valid. A
targeted test verifies that the unregularized Newton/constrained solution agrees
with the independent SLSQP solve at an active ball boundary.

### Matched-estimator `K=40` confirmation

On 32 disjoint paired seeds `63300..63331` and a fixed second pool of all 24
transfer rewards, using the same ridge `1e-4` for both estimators gives:

| Protocol | `lambda/eta^2` | Target SubOpt | Averaged transfer SubOpt |
|---|---:|---:|---:|
| Active, no warm-up | 3 | `2.458e-3` | `5.089e-4` |
| Active, no warm-up | 12 | `2.713e-3` | `5.250e-4` |
| Fixed-expert MLE | n/a | `5.943e-3` | `8.327e-4` |

Thus the currently configured Active design is `2.42x` better on target SubOpt
and `1.64x` better on averaged transfer at `K=40`. For transfer, Active is
better on 20 of 32 paired seeds; the Active-minus-fixed mean is `-3.24e-4`
with 95% Student-t interval `[-6.68e-4, 2.05e-5]`. This pilot is close but not
quite statistically resolved.

The exact paper estimator (ridge `0`) yields the same qualitative conclusion:
Target SubOpt is `2.567e-3` for Active versus `5.944e-3` for fixed-expert, and
averaged transfer SubOpt is `6.261e-4` versus `8.412e-4`. The transfer paired
interval still crosses zero. Ridge `1e-4` is nearly irrelevant for the fixed
trajectory, but it can perturb early Active estimates and therefore later
adaptive policies; exact zero is the clean final choice rather than trying to
declare any positive value "numerical only."

For comparison, applying ridge `1` to **both** methods on the selection pool
nearly erases the sampling-design difference: Active gives target/transfer
`2.131e-3 / 3.937e-4`, while fixed-expert gives
`1.829e-3 / 3.701e-4`. Their mean task-sensitive coefficient RMSEs are
`0.33875` and `0.33874`. This is direct evidence that the apparent fixed-expert
advantage was predominantly the prior, not its samples.

### Why the fair result also makes sense on transfer

The source deployment-regret Hessian is almost rank one. Its dominant direction
is approximately `(-0.730, -0.227, -0.644)`, and that direction is nearly
orthogonal (absolute inner product `0.071`) to the fixed expert's weakest
Fisher direction. This explains why an oracle source expert can remain strong
on target SubOpt despite poor generic identifiability.

Transfer is different: different leader rewards rotate the task-sensitive
directions. Across the 24 held-out rewards, their absolute overlap with the
fixed expert's weak direction ranges from `0.002` to `0.560` (median `0.216`).
With matched ridge `1e-4`, Active has lower mean SubOpt on every one of the 24
tasks. The fixed estimator puts `93.75%` of estimates on the reward-ball
boundary and has mean coefficient error `1.579`, versus `46.9%` and `1.020`
for Active. Its weakly identified component contaminates deployment across the
suite even though the suite was not constructed as a worst-case weak-direction
test.

This also explains why looking only at `q^T G q` for each task was incomplete.
The fixed design often has more direct information in a task's dominant local
direction, but its nearly singular full information matrix produces large,
correlated, nonlocal coefficient errors. Ridge `1` replaces that missing
information with a prior and makes the two methods look nearly identical. It
does not demonstrate superior transfer from fixed-policy samples.

### What may still be tuned

The information regularizer `lambda` in Active's policy-selection matrix is not
the MLE ridge. It is part of the sampling design and appears in the analysis;
it may be selected, together with warm-up budget, on protected selection seeds.
The estimator itself should remain the same constrained MLE for Active,
fixed-random, online-random, and fixed-expert MLE.

For a learning-curve claim, freeze ridge `0`, predeclare a curve-level objective
(rather than tuning every checkpoint), select only sampling-design parameters,
confirm once on disjoint seeds, and then generate the final curves on another
split. A regularized MaxEnt/MAP estimator can be shown only as a separately
labeled ablation with the same prior applied to every MLE collection design.

### Inverse-temperature analysis

The configured application value `eta=1.25` is in the original GUARD payoff
units. Because features are divided by their maximum norm `sqrt(3)`, the
compiled problem uses `eta=2.165`; the product `eta * phi` and hence behavior
are unchanged by that normalization.

The value `1.25` has no documented empirical calibration in the repository; it
was introduced with the first Lobeke experiment. It is a valid interior setting,
but a rather diffuse one. Under the fixed Stackelberg expert, follower action
probabilities span `0.0311..0.0701` over 23 actions and normalized entropy is
`0.988`. Under the joint QRE they span `0.0297..0.0983`, with entropy `0.980`.
The game is therefore not exactly uniform, but it is close.

A bounded sensitivity held features, rewards, `K=40`, 16 paired seeds, all 24
transfer tasks, Active `lambda/eta^2=3`, and ridge `0.0001` fixed. No algorithm
parameter was retuned at any temperature:

| Raw `eta` | Fixed-expert entropy | Weakest per-sample Fisher | Target SubOpt Active / fixed | Transfer SubOpt Active / fixed |
|---:|---:|---:|---:|---:|
| 0.625 | 0.997 | `3.55e-6` | `1.339e-3 / 0.843e-3` | `0.430e-3 / 0.485e-3` |
| 0.9375 | 0.993 | `2.67e-5` | `2.056e-3 / 1.607e-3` | `0.797e-3 / 0.669e-3` |
| 1.25 | 0.988 | `7.89e-5` | `2.509e-3 / 2.205e-3` | `0.653e-3 / 0.780e-3` |
| 1.5625 | 0.981 | `1.68e-4` | `3.287e-3 / 2.716e-3` | `0.570e-3 / 0.941e-3` |
| 2.5 | 0.950 | `7.34e-4` | `2.655e-3 / 14.130e-3` | `0.340e-3 / 1.274e-3` |

The 16-seed rows are mechanism pilots, not winner claims; the paired intervals
are wide. They do establish that `eta` materially changes both Fisher geometry
and task sensitivity. Higher `eta` makes actions more reward-sensitive and
creates a regime in which adaptive collection can exploit its better global
coverage; at `eta=2.5`, Active also has much lower coefficient error (`0.670`
versus `1.446`) and far fewer boundary estimates (`25%` versus `81%`). Absolute
SubOpt values should not be compared across rows because changing `eta` changes
the environment and deployment map.

`eta` should therefore **not** be tuned on which algorithm wins. Keep `1.25` in
the main result unless there is an external behavioral calibration, and add a
predeclared sensitivity panel. If a less-uniform follower is scientifically
preferred, define that choice before looking at algorithm outcomes using an
interpretable target such as expert-policy entropy or a max/min action-odds
ratio, then report the resulting `eta`. Selecting `2.5` merely because it makes
Active look better would be post-hoc tuning.

### Preserved artifacts

- Matched `1e-4` confirmation:
  `small-k40-common-ridge-confirm-{active-lambda3,active-lambda12,fixed}/`
- Exact unregularized confirmation: `small-k40-zero-ridge-confirm/`
- Matched-ridge task geometry:
  `small-k40-common-ridge-transfer-geometry.json`
- Common ridge-`1` task geometry: `small-k40-ridge1-transfer-geometry.json`
- Temperature pilots: `eta-sensitivity-k40-eta*-seeds63400-n16/`
- Original ridge mechanism grids: `small-k40-*-840e5c7*/`
- Source-sensitivity decomposition: `small-k40-geometry-1832566.json`

## Canonical reward units for ordinary SubOpt

The main metric remains the standard deployment regret

`SubOpt(mu_hat; r) = V_r(mu_r^*) - V_r(mu_hat)`.

No relative-regret denominator or `RelSubOpt` metric is introduced.  Instead,
each evaluation leader reward is represented canonically as

`r_unit = (r - min(r)) / (max(r) - min(r))`.

This positive-affine transformation leaves every leader-policy ordering and
every optimal leader policy unchanged.  It gives every source and transfer
task the same interpretable reward range `[0, 1]` and makes `R_max = 1` exactly.
An additive convention such as “the uniform leader has value 1” was rejected:
adding a constant cannot affect regret, and that convention would therefore
pretend to normalize scale without actually doing so.

The old source reward had range `0.9959677419`, so its final SubOpt curves only
increase by the factor `1 / 0.9959677419 = 1.0040486`.  Fisher, coefficient,
and certificate metrics are unchanged.  The old transfer rewards, however,
had task-dependent ranges much smaller than one under unit-Frobenius
normalization.  Their reported regrets were therefore compressed by unrelated
task scales.  At `K=1600`, the old/new averaged transfer SubOpt means are:

| Method | Unit-Frobenius reward | Unit-interval reward |
|---|---:|---:|
| Active IRL, w/ warm-up | `9.20e-5` | `6.41e-4` |
| Active IRL, w/o warm-up | `5.17e-5` | `3.53e-4` |
| Fixed-expert MLE | `7.94e-5` | `5.48e-4` |
| Zero-sum OLS | `4.41e-4` | `3.09e-3` |
| Follower-only OLS | `7.55e-4` | `5.28e-3` |

This is a change of reward units, not retraining: retained estimates and
collection trajectories are reused, and each policy is reevaluated against
the canonically normalized reward.  The export includes all 99 all-method,
taskwise arrays under schema 4 and records the normalization in provenance.
The previous data and figure assets remain present under their original names;
new outputs use the `_unit_reward` suffix or a `*_unit_reward/` directory.
