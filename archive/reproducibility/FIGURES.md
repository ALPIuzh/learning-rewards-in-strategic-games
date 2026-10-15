# Paper experiment and figure specification

This is the reviewable source of truth for the paper figures. It describes the
scientific claim and required data before a plot is generated. A figure enters
the tracked paper assets only after its specification and rendered layout have
both been approved.

## Shared rules

- The horizontal budget is always the number of observed poacher actions.
- Curves show the mean and a two-sided 95% Student-t confidence interval across
  independent seeds. Every caption states the seed count.
- Comparisons use the same follower-sample budget and paired application seeds.
- Hyperparameter selection, confirmation, and final evaluation use disjoint
  seeds. Confirmation never reselects a candidate.
- Every evaluation leader reward is represented positive-affinely in `[0,1]`,
  making `R_max=1` without changing policy orderings. The metric remains
  ordinary SubOpt; no relative-regret denominator is introduced.
- Downstream rewards are fixed before final evaluation and reused for every
  method. An averaged transfer metric is permitted only for an explicitly
  named fixed suite, with equal task weights after the common `[0,1]`
  normalization.
- Main-text downstream maps are explicitly illustrative rather than a claim of
  average performance over a task distribution. Plot all three rewards emitted
  by the fixed generator and retain every definition and outcome; do not select
  displayed tasks based on method results.
- Raw run directories stay gitignored. Only approved figures and the smallest
  derived arrays/JSON needed to compile them are tracked, together with source
  run IDs, configuration fingerprints, and Git commits.
- Python produces title-free figure components. Panel labels, captions, and
  paper narration belong in LaTeX. All plots use `figures/style.py` and all TikZ
  plots use the same color and line-style definitions.

## Frozen Lobeke choices

The selection stage used seeds `10000..10063`; confirmation used
`20000..20063`; final evaluation reserves the untouched seeds `30000..30049`.
The earlier run on `100..149` is pilot evidence because it was inspected before
the final hyperparameter decisions. The complete cross-study registry is in
`reproducibility/SEEDS.md`.

| Active IRL variant | Warm-up samples | Online lambda | Status |
|---|---:|---:|---|
| With warm-up | 200 | `0.003 eta^2` | transfer-selected, frozen, and evaluated on untouched final seeds |
| Without warm-up | 0 | `0.03 eta^2` | transfer-selected, frozen, and evaluated on untouched final seeds |

`lambda_warm=0.01` remains fixed and is used only inside the warm-up phase.
Every likelihood-based method, including Fixed-expert MLE, uses the same
unpenalized constrained MLE (`ridge=0`) defined in the paper. The earlier
source-selected ridge-1 expert result is retained only as a documented tuning
artifact and is not a current figure input.
The final flagship comparison must support the two different online lambdas;
the existing single-lambda run is exploratory and cannot be reused as final
paper evidence.

## Main text

### M1 — Source-task convergence

**Claim.** Active IRL learns from adaptively chosen leader policies under the
general MLE assumptions; fixed and online-random exploration are weaker
general-purpose alternatives. Fixed-expert estimators are shown in a separate
panel because they assume access to an expert leader policy for the source task.

**Layout.** One full-width TikZ figure with a shared vertical scale.

- Left: Active IRL, fixed random, and online random, each with and without
  warm-up. Color denotes algorithm; solid/dashed denotes warm-up status.
- Right: Fixed-expert MLE (the one-step MaxEnt IRL analogue), zero-sum
  OLS, and follower-only OLS. MLE uses the source Stackelberg-optimal fixed
  leader. The OLS methods instead use a joint-QRE expert required by their
  estimating equations; zero-sum OLS additionally observes one independently
  sampled expert leader action per follower action and uses both QRE equation
  blocks. Thus its horizontal axis counts follower actions, while its total
  number of observed player actions is twice as large.
- Horizontal axis: number of observed poacher actions.
- Vertical axis: last-iterate `SubOpt(hat mu)` on a log scale.
- The first 200 samples are marked as the warm-up region.

**Data.** A checksum-verified native current-code run on 50 paired
final-evaluation seeds and 1,600 samples, with checkpoints every 40 samples.
All nine protocols are regenerated together; zero-sum OLS uses reference action
22 and one sampled expert leader action per follower action. Active IRL uses its
separately tuned lambdas. Each expert policy is computed once from the source
reward and is never selected using downstream tasks.

### M2 — Concrete downstream transfer

**Claim.** A behavioral reward estimate can be reused to optimize patrols for
new leader objectives. Active IRL and Fixed-expert MLE use the same
general logit response model but different collection designs; their relative
performance can vary with the downstream objective.

**Layout.** Reference task plus three fixed downstream tasks as columns; oracle,
tuned Active IRL with warm-up, and Fixed-expert MLE as rows. Every map has the
same size and coverage color scale. Each learned row contains its task-specific
mean SubOpt in a small box. The main comparison deliberately does not use OLS:
those estimators require additional fixed-expert assumptions and remain in M1
and the appendix as assumption-dependent baselines.

The three columns span a behavioral shift rather than merely a reward-matrix
distance. From the predeclared ten-task transfer suite, select the tasks nearest
to the 0%, 50%, and 100% quantiles of total-variation distance between each
task's true optimal leader policy and the source true optimal leader policy.
Ties are resolved by the fixed suite index. The resulting distances are 0.23,
0.86, and 1.00, and the selected downstream policies have pairwise distances at
least 0.92. The selection rule never reads an estimated reward or algorithm
score.

**Data.** Post-processing of the final M1 run at 1,600 samples; no new follower
samples. Generate the fixed ten-task structured suite, apply the algorithm-blind
behavioral-quantile rule above, and plot the three selected rewards. Generation
fails unless the plotted comparison is
Fixed-expert MLE. Every likelihood-based method uses the same unpenalized
constrained MLE; downstream tasks select only the Active information
multiplier, never the fixed expert or an estimator-specific prior.

## Appendix

The appendix follows the paper's intended hierarchy. Lobeke is the complete
application study, two synthetic setups isolate complementary mechanisms, and
the final subsection checks the rate on the proof's hard instance.

### I.1 — Lobeke environment from the GUARD benchmark

This subsection defines the application and protocol, then reports all evidence
underlying M1 and M2. GUARD remains an external dependency; this repository
contains only the adapter, configurations, and derived paper assets.

#### I.1.1 — Environment and protocol

Explain the map, patrol coverage, poacher response, observable action,
behavioral features, reference leader reward, and fixed downstream rewards.
The compact environment schematic uses the exact policy-map component from M2:
target value, one source-task Stackelberg patrol, and its induced poacher
distribution are shown as a three-stage sequence, with the source reward model
written underneath. Its arrays are rebuilt through the pinned external GUARD
dependency and frozen with provenance; central figure generation never imports
GUARD. State the budget, warm-up accounting, seed partitions, and assumptions
of every method in one table.

#### I.1.2 — Target and averaged-transfer diagnostics

**Claim.** The target-task ranking is not an artifact of one metric, and the
same estimates can be inspected across a fixed deterministic suite of
downstream rewards.

**Layout.** A two-column by six-row grid. Every row is one metric: Target
SubOpt, Target Fisher error, the simulation-only self-concordance certificate
distance, coefficient L2 error, Averaged Transfer SubOpt, and Averaged Transfer
Fisher error. The left column contains active-design methods (Active IRL, fixed
random, and online random, each with and without warm-up); the right contains
Fixed-expert MLE, Zero-sum OLS, and Follower-only OLS. Matching row axes and
separate legends make the different data-access assumptions explicit.
For each seed and checkpoint, the averaged panels first compute each metric on
ten fixed leader rewards, each mapped by the same positive-affine convention to
`[0,1]` so `Rmax=1`, and then average the ten scalar values;
confidence intervals are computed across final seeds only afterward and remain
conditional on this task suite.

**Data.** Reuse the native current-code M1 export. The fixed transfer suite uses
task seed `9200` and is disjoint from the hyperparameter-tuning set at seed
`9100`.

#### I.1.2b — Matched warm-up/lambda factorial

**Claim.** The tuned warm/no-warm curves do not identify the warm-up effect,
because they use different information multipliers. A paired
`warm-up in {0,200}` by `lambda/eta^2 in {3,12}` factorial shows the causal
final-checkpoint effects. Warm-up improves Target SubOpt at multiplier 3 but
not detectably at multiplier 12; no transfer, Fisher, or certificate effect is
detected at either multiplier on 50 paired seeds.

**Layout.** Four warm-minus-no-warm effect panels: Target SubOpt, Averaged
Transfer SubOpt, Target Fisher error, and the simulation-only certificate
distance. Plot paired means and paired 95% confidence intervals with a zero
reference line; negative values favor warm-up.

**Data.** Combine the first 20 seeds of native reference run
`8c4ab05e9b2fe388` with swapped-lambda runs `8513aac9cbe3189a` (seeds
`30000..30019`) and `2d6a6b2f2d58eebd` (seeds `30020..30049`). All runs use
purpose/index-keyed random streams, 1,600 total follower observations, and the
same fixed ten-task transfer suite. The first 20-seed export remains preserved;
the canonical `v2` export contains the planned 50-seed extension.

#### I.1.3 — Full downstream-policy grid

**Claim.** Show where every evaluated method succeeds or fails on the same
concrete transfer tasks without crowding the main text.

**Layout.** Keep the reference task and the three behaviorally stratified
downstream tasks from M2 and
split the comparison into two full-width figures. The general-exploration
figure contains the oracle plus Active IRL, fixed random, and online random,
each with and without warm-up. The fixed-expert figure contains the oracle plus
fixed-expert MLE, zero-sum OLS, and follower-only OLS. Repeating the oracle makes
each group directly readable and keeps every map exactly the same size as M2.
Both figures use the shared policy-grid component, normalization, label
placement, and task-specific SubOpt boxes.

Do not add arbitrary heatmap columns. A broader transfer claim requires a
separately defined task distribution or meaningful one-parameter task family,
not additional hand-picked examples.

**Data.** Reuse the corrected M1 composition and exact M2 task definitions.

#### I.1.3b — Behavioral transfer-distance heatmaps

**Claim.** The ten-task average and the three policy maps are not hiding a
fixed-expert advantage on a broader task set. Across a separate frozen suite,
Active IRL remains substantially better than Fixed-expert MLE; source-relative
policy displacement changes the error profile, but is not itself a universal
notion of task difficulty.

**Layout.** Report two matched $3\times3$ heatmaps, one stratified by total
variation between the true source- and downstream-optimal leader policies and
one by the downstream value lost by the true source-optimal policy. Rows are
low/medium/high tie-aware distance buckets and columns are the common
checkpoints $200,400,800,1200,1600$. The first two method columns contain the
six protocols with general data-collection access; the third contains
Fixed-expert MLE, Zero-sum OLS, and Follower-only OLS. Both figures use the
same logarithmic color scale.

**Data.** Evaluate all 180 i.i.d. signed-priority rewards from seed `9600`, not
a performance-selected subset. Every reward is positive-affinely normalized
to `[0,1]`. Buckets are frozen from true optimal policies and values before
reading any estimate; each cell averages all tasks in the bucket and all 50
final learning seeds. TV has bucket counts `60/71/49` because all 42 saturated
distance-one tasks remain tied; the value-gap buckets are `60/60/60`. The
tracked export stores every taskwise SubOpt and Fisher value, oracle policy,
distance, assignment, checksum, and within-bucket extreme diagnostic.

#### I.1.4 — Hyperparameter selection and confirmation

**Claim.** The frozen Active IRL settings remain statistically competitive in
an estimator-matched, post-freeze sensitivity audit. All likelihood-based
collection designs use the same unpenalized constrained MLE defined in the
paper. Earlier unequal-ridge results are retained only as a bias--variance
diagnostic and are not part of the current comparison.

**Layout.** A complete $3\times5$ ridge-zero surface at the final 1,600-sample
budget. The five discrete lambda settings use equal-width columns labeled by
their actual multipliers; color uses a zero-anchored fixed $[0,4\times10^{-4}]$
scale. Outlines identify the protocols frozen before the audit. All other
candidate-minus-frozen paired intervals contain zero, so this shared result is
stated once in the caption instead of repeated inside every cell. Missing cells
are omitted, never imputed or rendered as results. The
caption distinguishes an unresolved difference from equivalence and states
that the audit did not reselect the final protocols. The tracked payload also
records an independent 20-seed ridge-zero confirmation: its means recover both
outlined row choices, its warm-up contrasts remain unresolved, and both frozen
Active protocols outperform matched fixed-expert MLE.

### I.2 — Performance in synthetic environments

The EWRL-style template is used consistently: when a controlled parameter is
swept, show representative learning curves against observed follower actions
and a compact final-budget comparison. Every sweep reports both
Target SubOpt and Target Fisher error. Relative-performance
ratios may be secondary summaries but never replace the two absolute metrics.

#### I.2.1 — Structured nontrivial game

**Question.** How sensitive are the algorithms to ordinary game geometry and
to allocating samples between common warm-up and adaptive exploration?

**Core warm-up ablation.** Refresh the existing 100-seed toy sweep at
`lambda=eta^2`, total budget 1,000, and warm-up budgets
`[0,24,60,120,240,500,800,1000]`. Plot final SubOpt and Fisher error against
warm-up fraction for Active IRL, fixed random, and online random. One or two
representative learning-curve insets may show the phase transition; do not keep
eight nearly redundant full plots.

**Expected interpretation.** Active IRL should be insensitive over a broad
range, weak baselines should improve as they inherit informative common data,
and every method must coincide when warm-up consumes the entire budget.

**Controlled parameter ablations.** Following the EWRL organization, vary only
axes with a clear scientific interpretation: number of leader actions, number
of follower actions, follower temperature, feature dimension, and online
regularization around the theorem default. The temperature sweep keeps the
reward and feature scale fixed, varies the known inverse temperature `eta`, and
scales information regularization as `lambda/eta^2`; this avoids silently
conflating response sharpness with an unrelated regularization change. Begin
with a small preflight grid around the canonical instance. Keep an axis only
when it reveals a stable qualitative effect; relegate redundant/null results
to a table or omit them.
The regularization sweep here measures sensitivity on a fixed synthetic game;
the protected Lobeke selection in I.1.4 serves the separate purpose of choosing
the application configuration without using final-evaluation seeds.

#### I.2.2 — Sparse-information game

**Question.** Can active exploration be dramatically better when informative
leader policies occupy a small part of the simplex?

**Instance.** Use the existing one-dimensional sparse-information construction.
With `m` leader actions, only action zero affects the follower response. At the
initial estimate, Active IRL assigns it probability one, while a uniformly
random simplex policy has expected squared mass `2/(m(m+1))`; the initial
information ratio is exactly `m(m+1)/2` (136 at `m=16`).

**Layout.** Show SubOpt and Fisher error versus observed follower actions for
the existing `m=16`, no-warm-up instance, then final SubOpt and Fisher error
versus `m in {2,4,8,16,32}`. Annotate the analytical information ratio on the
action-count sweep. Include 64 only if a runtime/optimization preflight is clean.

Keep budget, seed count, true coefficient, and decision threshold fixed. No
warm-up is used in the mechanism panel because common D-optimal warm-up removes
the information scarcity being studied. A small warm-up comparison may be used
to demonstrate exactly that effect.

### I.3 — Active IRL on the minimax lower-bound instance

**Claim.** Check whether implemented Active IRL exhibits the `sqrt(d/K)` scaling
predicted by the upper and lower bounds on the paper's hard instance. This
validates a rate, not the loose universal constant `1/44`.

**Instance.** `A=[d]`, `B={b1,b2}`, `phi(ai,b1)=0`, `phi(ai,b2)=ei`, and leader
reward `1{b=b2}`. For each `(d,K)`, set `eta=1`, `Rmax=1`, `S=1`, and
`omega=epsilon ei` with `epsilon=0.5 sqrt(d/K)`. Use theorem-valid points
`d>32` and `K>=4d`. Disable warm-up so `K` is the total observed follower
actions, and use the theorem default `lambda=eta^2` rather than Lobeke-tuned
values.

**Grid.** `d in {33,64}` and `K/d in {4,8,16,32,64}`, with 16 independent
replicates of every active coordinate. The coordinate order is balanced, shuffled by
a domain-separated design seed, and reused across `K/d` values. The implementation
uses the hard instance's verified sufficient-statistic MLE and closed-form policy and
evaluation solutions; the complete grid is required to remain below 30 minutes on the
reference machine.

**Layout and checks.** Compare both required metrics, while keeping the
theoretical claim attached only to SubOpt:

- log-log final SubOpt and Fisher error versus `K/d`, with 95% confidence
  intervals; the SubOpt panel includes a reference slope `-1/2`;
- retain both the uniform-coordinate mean and every coordinate's replicate mean,
  including the maximum coordinate sample mean, so the main text can decide which
  aggregation is most informative without rerunning the experiment;
- SubOpt normalized by `sqrt(d/K)`, which should be approximately flat and
  comparable across dimensions if the algorithm matches the lower-bound rate;
- empirical Fisher-error slopes as an estimation diagnostic, without assigning
  them the lower bound's normalization unless that rate is derived separately;
- retain prespecified weighted SubOpt slope estimates and confidence intervals
  in the numerical artifact for auditing, without adding a separate figure panel.

Show the theoretical lower-bound line only as a floor. Claim matching empirical
scaling only if the tail slope is compatible with `-1/2` and normalized risk has
no material trend.

## Experiment sharing

| Experiment/data source | Figures supplied |
|---|---|
| Final tuned Lobeke comparison | M1, M2, I.1.2, I.1.3 |
| Protected Lobeke tuning artifacts | I.1.4 |
| Refreshed structured-toy study | I.2.1 |
| Refreshed sparse-information study | I.2.2 |
| Minimax hard-instance study | I.3 |
| Deterministic environment build | optional application illustrations |

The final comparison supports distinct Active IRL lambdas for each warm-up
mode. Every native export records these settings, the fixed-MLE expert
definition and ridge, the OLS references and sampling protocol, and the RNG
schema in machine-readable provenance.

## Historical data that must be refreshed

The tracked zero-sum synthetic data remain available under
`reproducibility/data/zero_sum/` as exploratory and implementation-validation
evidence, but are intentionally not registered as publication figures because
the Lobeke experiment already contains the relevant zero-sum estimator
comparison.
