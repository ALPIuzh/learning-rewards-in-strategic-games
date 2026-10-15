# Active IRL, fixed-expert MLE, and warm-up audit

This file is the append-only scientific record for the bounded audit started on
2026-08-12.  Protected paper arrays and existing run artifacts are read-only.
New diagnostic outputs are written under `results/`; no existing artifact is
deleted or overwritten.

## Questions

1. Why is Active IRL's transfer advantage over fixed-expert MLE smaller and
   later than expected?
2. How much of fixed-expert MLE's strength comes from its separately tuned
   estimator ridge rather than its sampling design?
3. Is warm-up ineffective, harmful, or merely confounded by the separately
   tuned online information multiplier?
4. Does Active IRL optimize its intended information objective reliably, and
   does that objective emphasize the reward directions used by the fixed
   downstream suite?

## Established from the protected final arrays

- At 1,600 poacher observations, no-warm Active IRL **does beat** fixed-expert
  MLE in averaged transfer SubOpt: `3.5306e-4` versus `5.4775e-4`.  The paired
  Active-minus-expert difference is `-1.9470e-4`, with 95% CI
  `[-3.4513e-4, -4.4265e-5]`; Active is better on 35/50 paired seeds.  The
  improvement is 35.5% (1.55x), not the much larger separation one might hope
  for.
- Warm Active IRL does not beat the expert at the final checkpoint:
  `6.4051e-4` versus `5.4775e-4`; the paired CI crosses zero.  This plotted
  arm uses `lambda/eta^2=12`, whereas no-warm Active uses `3`, so the two blue
  curves are not a causal warm-up comparison.
- In the matched factorial, warm-up at multiplier 3 improves target SubOpt
  (`1.4885e-4` versus `2.5925e-4`; paired 95% CI for warm-minus-no-warm
  `[-2.1258e-4,-8.2205e-6]`) but has no detectable averaged-transfer SubOpt
  effect (`3.4374e-4` versus `3.5305e-4`; CI crosses zero).
- The larger information multiplier is harmful for transfer.  At fixed warm-up
  200, multiplier 12 minus multiplier 3 increases averaged transfer SubOpt by
  `2.9678e-4` (paired 95% CI `[6.7799e-5,5.2576e-4]`) and increases averaged
  transfer Fisher error by `8.5199e-3` (CI `[3.2919e-3,1.3748e-2]`).  The same
  qualitative effect occurs without warm-up.
- Fixed-expert MLE currently uses ridge `1.0`; Active IRL and random-design MLEs
  use ridge `1e-4`.  The fixed-expert ridge was selected on source SubOpt only.
  This is a real estimator asymmetry and must be separated from the collection
  design before interpreting the algorithm comparison.

## Working hypotheses (not conclusions)

- **Tuning-target mismatch:** multiplier 12 was selected using source SubOpt,
  a noisy and direction-specific endpoint.  A large `lambda` makes the online
  criterion closer to one-step total-information maximization and less
  sensitive to underexplored directions; this can preserve source deployment
  while harming reward identification and transfer.
- **Self-warming:** no-warm Active starts from `omega_hat=0` and an isotropic
  information matrix.  Its first decisions therefore already implement a
  sequential reward-agnostic information design close to the offline WARM
  design.  An explicit 200-sample WARM phase may improve source stability but
  need not add new transfer information.
- **Shrinkage advantage:** ridge 1 may suppress the fixed design's weak,
  high-variance coefficient direction.  If that shrinkage is favorably aligned
  with these downstream tasks, the displayed fixed-expert curve combines a
  sampling design with a strong prior toward zero and is not an estimator-matched
  baseline.

## Direct counterfactual checks

### Fixed-expert ridge is the dominant apparent advantage

`refit_fixed_expert.py` refits the exact 50 protected fixed-expert trajectories
without changing a policy, follower action, seed, checkpoint, or evaluation
task. The immutable result is
`results/fixed_expert_ridge_refit_v1.json`.

At 1,600 samples:

| MLE ridge | Boundary fits | Target SubOpt | Target Fisher | Coefficient L2 | Averaged transfer SubOpt | Averaged transfer Fisher |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48% | 4.654e-4 | 3.217e-2 | 1.136 | 2.983e-3 | 9.867e-2 |
| 1e-4 | 46% | 4.653e-4 | 3.217e-2 | 1.135 | 2.983e-3 | 9.864e-2 |
| 0.01 | 48% | 4.501e-4 | 3.203e-2 | 1.107 | 2.824e-3 | 9.516e-2 |
| 0.1 | 28% | 3.874e-4 | 3.109e-2 | 0.901 | 2.275e-3 | 8.040e-2 |
| 1 | 0% | 2.541e-4 | 2.853e-2 | 0.427 | 5.478e-4 | 3.903e-2 |

The plotted fixed expert therefore combines its collection policy with very
strong shrinkage. At the estimator-matched ridge `1e-4`, no-warm Active is
8.45x better in averaged transfer SubOpt (`3.531e-4` versus `2.983e-3`), and
the paired 95% interval excludes zero. Ridge 1 was selected on source SubOpt;
it was not a numerical-stability constant and cannot be attributed to the
fixed-expert sampling design.

### Warm-up data are retained and improve early conditioning

The scalar algorithm initializes its online state with `warmup.dataset`; the
batched runner fills the leading policy/action slice with WARM observations.
Both compute online
`G_k = lambda I + cumulative_fisher_information(all policies so far, omega_hat)`,
so WARM samples are present in every subsequent information matrix.

At the true coefficient and 200 total samples, the mean per-sample Fisher
eigenvalues are:

- offline WARM: `[0.007086, 0.071871, 0.147861]`, condition number 20.9;
- no-warm Active at multiplier 3: `[0.002927, 0.068663, 0.159502]`, with
  median condition number 69.2.

Thus WARM materially improves the weak direction at the boundary. No-warm
Active is nevertheless already a sequential information design: it starts at
`omega_hat=0` and greedily explores from an isotropic information matrix, so it
partially self-warms.

### The old warm curve is primarily a lambda artifact

The protected matched factorial separates warm-up from the information
multiplier. At multiplier 3 and 1,600 samples, WARM improves target SubOpt
(`1.488e-4` versus `2.592e-4`, paired 95% CI excluding zero) and has no
detectable transfer effect (`3.437e-4` versus `3.531e-4`). At multiplier 12,
the corresponding differences are not detectable.

More importantly, multiplier 12 damages final information geometry. At 1,600
samples:

- WARM + multiplier 3: eigenvalues `[0.007190, 0.072526, 0.151014]`, median
  condition number 19.7;
- WARM + multiplier 12: `[0.003513, 0.072645, 0.167455]`, median condition
  number 51.6;
- no-WARM + multiplier 3: `[0.006800, 0.072287, 0.151703]`, median condition
  number 20.3;
- no-WARM + multiplier 12: `[0.002792, 0.072491, 0.168433]`, median condition
  number 91.3.

The separately selected warm multiplier 12 therefore trades away the weak
direction that WARM initially repaired. It was chosen using noisy source SubOpt,
not transfer, coefficient error, or a matched warm-up comparison.

### Optimization is not the explanation

All protected MLE projected-gradient residuals are below tolerance. A targeted
64-start/8-refinement replay at early and late active checkpoints found a
maximum acquisition-objective gap of `3.01e-4` (2.8% relative) at isolated
early rounds; gaps were zero to numerical precision at all inspected late
checkpoints. The stored-policy replay mismatch was below `4e-18`. This is too
small and too transient to explain either final curve.

### Sub-theoretical lambda pilot

The information regularizer only needs to be strictly positive for the
algorithm to be numerically defined. The former executable restriction
`lambda >= eta^2` encoded a sufficient theorem condition and prevented an
empirical search below it. A bounded coarse pilot now tests multipliers
`{0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}` with warm-up budgets `{0, 50, 200}`
on four disjoint seeds, 800 samples, and 40 new normalized transfer rewards.
Its immutable output is `results/lambda_grid_800_v1/`.

Preliminary rankings are noisy but decisive about the missing search region:

- best no-warm cell: multiplier 0.03, transfer SubOpt `2.806e-4`;
- best 200-warm cell: multiplier 1, `2.228e-4`; multiplier 0.003 is close at
  `2.721e-4` and is better on Target SubOpt and transfer Fisher error;
- estimator-matched fixed expert: `4.223e-3`.

This is a diagnostic grid, not a final selection: four seeds are insufficient,
and the warm ranking is not smooth in lambda. An independent narrowed
confirmation uses new seeds and reward tasks before any paper protocol changes.

### Independent selection and held-out confirmation

A narrowed eight-seed study at 1,600 samples used new seeds `94000..94007`
and 100 new normalized rewards from task seed `9400`. A boundary refinement
tested smaller warm-up multipliers and a denser no-warm neighborhood. It froze
the following transfer-selected protocols without inspecting the later
held-out suite:

- no warm-up: `lambda/eta^2 = 0.03`;
- 200-sample warm-up: `lambda/eta^2 = 0.003`.

The selection-suite transfer means were `7.813e-5` and `1.658e-4`,
respectively. Going below the warm boundary did not improve it; multiplier
`0.0003` gave `2.128e-4`. The apparently sharp no-warm optimum was noisy, so a
separate confirmation used seeds `95000..95019` and 100 new tasks at seed
`9500`, retaining both frozen choices and the matched-lambda counterfactuals.

At ridge `1e-4`, the held-out means were:

| Protocol | Averaged transfer SubOpt | Transfer Fisher | Target SubOpt |
|---|---:|---:|---:|
| No WARM, multiplier 0.003 | 1.598e-4 | 2.744e-2 | 1.310e-4 |
| No WARM, multiplier 0.03 (frozen choice) | 2.156e-4 | 3.153e-2 | 1.855e-4 |
| WARM 200, multiplier 0.003 (frozen choice) | 1.906e-4 | 2.628e-2 | 1.693e-4 |
| WARM 200, multiplier 0.03 | 1.794e-4 | 3.012e-2 | 1.929e-4 |
| Fixed expert, matched ridge | 2.833e-3 | 9.319e-2 | 6.183e-4 |

Neither matched warm-minus-no-warm transfer interval excludes zero. Every
Active-minus-fixed-expert interval does: for the frozen no-warm choice the
paired difference is `-2.618e-3`, 95% CI `[-3.914e-3,-1.322e-3]`; for the
frozen warm choice it is `-2.643e-3`, CI
`[-3.902e-3,-1.384e-3]`.

An exact unpenalized-MLE replay of the two frozen Active choices and fixed
expert on the same held-out split gives transfer means `2.160e-4`, `1.592e-4`,
and `2.834e-3`. Thus ridge `1e-4` is not driving the corrected Active result.
The final protocol uses ridge zero because that is the estimator defined in
the paper; all MLE collection designs use it identically.

## Complete ridge-zero post-freeze audit

The final estimator was subsequently checked on a complete `3 x 5` surface at
the final 1,600-sample budget: warm-up thresholds `[0, 50, 200]`, information
multipliers `[0.003, 0.01, 0.03, 0.3, 1]`, eight paired seeds
`94000..94007`, and 100 fixed normalized downstream tasks from seed `9400`.
All 15 cells completed without a solver failure. The frozen no-warm choice
`0.03` had mean transfer SubOpt `2.141e-4`; the frozen warm-200 choice `0.003`
had mean `3.071e-4`. Every candidate-minus-corresponding-frozen 95% paired
interval contained zero, so this noisy post-freeze audit does not justify
reselection.

An independent confirmation used 20 paired seeds `95000..95019` and a new
100-task suite from seed `9500`. It recovered both frozen row choices in mean:

| Row | Frozen | Alternative | Frozen mean | Alternative mean | Paired 95% CI, frozen minus alternative |
|---|---:|---:|---:|---:|---:|
| No warm-up | `0.03` | `0.003` | `2.160e-4` | `2.613e-4` | `[-1.730e-4, 8.243e-5]` |
| Warm-up 200 | `0.003` | `0.03` | `1.592e-4` | `2.662e-4` | `[-2.217e-4, 7.679e-6]` |

The matched warm-minus-no-warm intervals still contain zero. In contrast, the
frozen no-warm and warm protocols beat ridge-zero fixed-expert MLE by paired
means `-2.618e-3` and `-2.675e-3`, with intervals entirely below zero. The
tracked publication export retains every sensitivity and confirmation
per-seed score plus source-artifact checksums; the figure outlines the frozen
protocols rather than selecting outlines from these audit means.

The correct scientific interpretation is therefore:

1. estimator-matched, transfer-tuned Active is roughly 13--18x better than
   fixed-expert MLE on independent 100-task suites;
2. WARM is implemented and provides much better early weak-direction
   information;
3. no-warm Active self-warms, so after tuning its online regularizer the final
   transfer difference from explicit WARM is not detectable;
4. the old apparent expert advantage and useless-warm curve were caused by
   estimator-ridge and information-multiplier asymmetries, respectively.

### Figure organization

The appendix diagnostic renderer now places active-design methods in the left
column and fixed-expert methods in the right column, one shared-scale row per
metric. Separate legends and column titles make the different data-access
assumptions explicit. Generated publication fragments contain none of the
forbidden audit term `legacy`.

## Frozen current-protocol final evaluation

The final MLE/random run `349676e5922332fd` completed 350/350 jobs and the
sampled-OLS run `cf7b0f494e850f2d` completed 100/100 jobs. Both were produced
from clean commit `37cc482`; every stored result hash was rechecked. The latter
run stores equally many empirical leader and follower actions for Zero-sum OLS.
The two runs have byte-equivalent data-generating configurations after removing
only scheduling fields (`name`, requested methods, and requested warm-up modes).

The immutable composed export is
`reproducibility/data/lobeke/final_evaluation_unit_reward_v2/`. It contains all
99 expected arrays: source, certificate, coefficient, and taskwise/averaged
transfer diagnostics for nine protocols. At 1,600 observations:

| Protocol | Target SubOpt | Averaged transfer SubOpt | Averaged transfer Fisher |
|---|---:|---:|---:|
| Active IRL, WARM 200, multiplier 0.003 | 1.912e-4 | 2.413e-4 | 3.066e-2 |
| Active IRL, no WARM, multiplier 0.03 | 3.576e-4 | 2.925e-4 | 3.563e-2 |
| Fixed-expert MLE, ridge 0 | 4.650e-4 | 2.978e-3 | 9.857e-2 |
| Zero-sum OLS, paired actions | 3.070e-3 | 3.088e-3 | 1.741e-1 |
| Follower-only OLS | 2.787e-2 | 5.276e-3 | 8.464e-1 |

WARM Active is 12.3x better than Fixed-expert MLE in mean final transfer
SubOpt; no-WARM Active is 10.2x better. Their paired Active-minus-expert 95%
intervals are respectively `[-3.581e-3,-1.892e-3]` and
`[-3.538e-3,-1.832e-3]`. The paired WARM-minus-no-WARM transfer difference is
`-5.12e-5`, with interval `[-1.61e-4,5.89e-5]`, so the final transfer effect of
explicit WARM remains statistically unresolved even though its mean, target
SubOpt, Fisher error, and coefficient error are all lower.

The paper preview was regenerated and visually inspected. The appendix uses
six shared-scale metric rows, with active-design protocols only in the left
column and fixed-expert protocols only in the right. Intermediate x tick labels
and long vertical labels were removed after PDF inspection showed collisions;
the final render has no overlaps, keeps separate legends, and tightens the
Averaged Transfer SubOpt floor from `2e-4` to `1e-4` without clipping an upper
confidence endpoint.

## Dense behavioral-transfer heatmaps

The completed diagnostic evaluates all nine current protocols on all 180
normalized structured rewards drawn from the frozen seed `9600`, across the 50
final learning seeds and checkpoints `200, 400, 800, 1200, 1600`. It replays
retained coefficient histories and therefore does not retrain, resample, or
alter any collection trajectory. Every method/task evaluation is cached in an
atomic file; all 1,620 cache entries are present. The six-reward smoke suite and
all dense caches remain preserved.

The two algorithm-blind distance definitions give:

- optimal-policy TV: edges `[0.137, 0.725, 0.877, 1.000]`, bucket counts
  `60/71/49`, with 42 tasks tied at distance one;
- source-policy downstream value gap: edges
  `[0.00488, 0.02913, 0.06748, 0.13334]`, bucket counts `60/60/60`;
- TV/gap Spearman correlation: `0.745`.

At the final checkpoint, averaging all 180 tasks gives:

| Protocol | Mean transfer SubOpt |
|---|---:|
| Active IRL, w/ warm-up | `2.440e-4` |
| Active IRL, w/o warm-up | `3.067e-4` |
| Fixed random, w/ warm-up | `1.136e-3` |
| Fixed random, w/o warm-up | `3.416e-3` |
| Online random, w/ warm-up | `1.057e-3` |
| Online random, w/o warm-up | `2.814e-3` |
| Fixed-expert MLE | `3.028e-3` |
| Zero-sum OLS | `3.199e-3` |
| Follower-only OLS | `5.520e-3` |

This independently reproduces the corrected conclusion from the ten-task
suite: Active with and without warm-up is respectively `12.4x` and `9.9x`
better than Fixed-expert MLE, so the result is not a small-suite or reward
selection artifact.

The bucket trends require care. Active's final SubOpt increases mildly from
low to high TV (`1.99e-4`, `2.23e-4`, `3.30e-4` with warm-up), but several weak
baselines decrease. This is not an implementation reversal: TV and source
policy gap measure displacement from the source task, not universal deployment
difficulty. The downstream oracle policies vary from pure to seven-action
support, and task error also depends on how their sensitive directions align
with each estimator's coefficient error. For example, final taskwise SubOpt is
strongly associated with task Fisher error for both OLS estimators, whereas
TV alone is not. A peaked downstream optimum can be comparatively easy for a
poor estimate if recovering its dominant action is enough.

Within-bucket extremes do not drive the displayed means. Across all methods,
metrics, and buckets, the single largest task contributes only `2.4--6.0%` of
a cell's final total, and the worst 10% contribute `13.6--23.4%`. The tracked
provenance records taskwise quantiles, the minimum/median/maximum representative
task indices, both distance ranges, and these concentration shares for every
cell. The two heatmaps deliberately retain both TV and gap: neither is chosen
post hoc as the metric that makes Active look best.
