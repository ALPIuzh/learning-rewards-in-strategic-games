# Lobeke flagship experiment

Tier-3 Lobeke anti-poaching game (5x5 grid, 23 leader/23 follower actions, built
live from the GUARD submodule's elephant-tracking CSVs — same game as
`../geographic/map_lobeke.py`),
compiled as a genuine zero-sum game (`leader_features = -phi`) so both the
Stackelberg-exploration methods and Liao's zero-sum OLS estimator can be
compared on one follower-sample-count axis.

## Methods compared

- **Active IRL** (own D-optimal warm-up, on/off)
- **Fixed-expert MLE** — standard fixed-policy MaxEnt-style likelihood fit
  under the source-task Stackelberg-optimal leader
- **Liao zero-sum OLS** — "aware" of the zero-sum structure; estimates both
  QRE marginals from paired expert actions and requires a genuine joint-QRE
  expert (`mairl.zero_sum.solve_qre`), not a Stackelberg-committed leader
- **Follower-only OLS** — "unaware" baseline, same samples, only the follower's
  own equations
- **Fixed random** / **Random each round** (own warm-up, on/off)

The fixed-policy methods have no warm-up variant. MLE keeps the
Stackelberg-optimal leader fixed, while both OLS methods keep their joint-QRE
leader fixed as required by their equations (see `study.py`'s module docstring).

Fixed-expert MLE uses the same constrained follower likelihood as Active IRL.
The current comparison sets `ridge=0` for every likelihood-based method, which
is the estimator defined in the paper. A prior source-only calibration assigned
ridge `1` only to Fixed-expert MLE. Re-fitting the identical samples showed that
this shrinkage reduced its averaged-transfer SubOpt from `2.98e-3` to `5.48e-4`;
it was therefore a substantive prior, not a numerical-stability constant. That
audit remains preserved in
`reproducibility/data/lobeke/fixed_expert_mle_stackelberg_tuning.json`, but ridge
`1` is not used by current figures.

## Current final evaluation

The current evidence comprises two disjoint clean runs at commit `37cc482`,
composed only after verifying every result checksum and their identical
data-generating configuration:

- MLE and random-design run `349676e5922332fd`: 350/350 jobs;
- sampled OLS run `cf7b0f494e850f2d`: 100/100 jobs.

Both use 50 paired untouched seeds (`30000..30049`) and 1,600 observed poacher
actions. Active IRL uses the frozen transfer-selected choices: a 200-sample
warm-up with `lambda/eta^2=0.003`, or no warm-up with
`lambda/eta^2=0.03`. Every likelihood fit is unpenalized (`ridge=0`).

The raw runs remain gitignored and preserved unchanged. The canonical paper
figures use the checksum-verified 99-array export in
`reproducibility/data/lobeke/final_evaluation_unit_reward_v2/`. Every evaluation
leader reward is represented positive-affinely in `[0,1]`, making `R_max=1`
without changing any leader-policy ordering; the metric remains ordinary
SubOpt. Zero-sum OLS stores one sampled leader action per follower action, and
both OLS estimators use reference action 22. Regenerate the export into a new
empty directory with:

```bash
uv run python -m experiments.guard.paper.figures.export_final \
  --run-dir artifacts/runs/guard-lobeke-paper-mle-corrected-v2/349676e5922332fd \
  --run-dir artifacts/runs/guard-lobeke-paper-ols-corrected-v2/cf7b0f494e850f2d \
  --output-dir /tmp/lobeke-final-evaluation-unit-reward
```

The earlier compact, correction, and composite exports remain unchanged for
auditability but are not canonical figure inputs.

At the last checkpoint, mean target SubOpt is `1.91e-4` for Active IRL with
warm-up, `3.58e-4` without warm-up, `4.65e-4` for Fixed-expert MLE,
`3.07e-3` for Zero-sum OLS, and `2.79e-2` for Follower-only OLS.
Mean averaged-transfer SubOpt is respectively `2.41e-4`, `2.93e-4`,
`2.98e-3`, `3.09e-3`, and `5.28e-3`. Thus the current Active protocols are
roughly 10--12 times better than estimator-matched Fixed-expert MLE. The paired
WARM-minus-no-WARM interval crosses zero at the final transfer checkpoint, so
the plot does not claim a detectable final WARM effect. These unit-reward values are rendered in the
main convergence figure, appendix diagnostics, and policy maps.
The retained reward estimates and collection trajectories are unchanged; only
their deployment values are expressed in the canonical reward units.

### Dense behavioral-transfer evaluation

The appendix also evaluates every current protocol on a separate fixed suite
of 180 i.i.d. signed-priority rewards (seed `9600`) at checkpoints
`200, 400, 800, 1200, 1600`. No task is selected using method output. Rewards
are bucketed independently by (i) total variation between the true source- and
downstream-optimal leader policies and (ii) the downstream value lost by the
true source-optimal policy. Ties remain together, producing TV bucket sizes
`60/71/49`; value-gap buckets are exactly `60/60/60`.

At 1,600 observations, the mean over all 180 tasks is `2.44e-4` for Active IRL
with warm-up, `3.07e-4` without warm-up, and `3.03e-3` for Fixed-expert MLE.
Thus the broader suite preserves an approximately 10--12x Active advantage.
The TV and gap rankings are related but not interchangeable (Spearman
correlation `0.745`). They are described as transfer-distance buckets rather
than universal difficulty buckets: weak no-warm and OLS estimators can have
lower SubOpt on more distant tasks when those tasks have a more peaked optimal
policy or align more favorably with the estimator's error.

The complete checksum-verified arrays and within-bucket extreme diagnostics
are tracked in `reproducibility/data/lobeke/behavioral_transfer_v1/`. Atomic
per-method/task caches make the export resumable. Rebuild it into a new empty
directory with:

```bash
uv run python -m experiments.guard.paper.figures.export_behavioral_transfer \
  --run-dir artifacts/runs/guard-lobeke-paper-mle-corrected-v2/349676e5922332fd \
  --run-dir artifacts/runs/guard-lobeke-paper-ols-corrected-v2/cf7b0f494e850f2d \
  --output-dir /tmp/lobeke-behavioral-transfer
```

The main transfer illustration uses three tasks from the fixed deterministic
ten-task evaluation suite. They are selected without reading any algorithm
output: compute total variation between the source and each downstream true
optimal leader policy, then retain the tasks nearest to the 0%, 50%, and 100%
distance quantiles (fixed-suite tasks 6, 3, and 1). Their distances are 0.23,
0.86, and 1.00, and their pairwise distances are all at least 0.92. The old
reward-cosine selection covered only source-policy distances 0.73--0.87. These
task-specific values illustrate failure modes across behavioral shifts; they
are not an uncertainty statement over a downstream-task population.

With reference action 22, the follower-only OLS equations have condition number
approximately 13,700, versus approximately 174 for Liao's zero-sum-aware equations.
The historical ref-0 run instead used conditions 20,746 and 190. The current configuration therefore
uses `FixedLeaderOLSConfig.ball_tolerance=1e-6` and retains the `1e-12` root tolerance;
the solver automatically retries the secular root at machine precision if its
norm postcondition fails.
The complete values are tracked in `conf/experiment/guard_lobeke_paper.yaml`.

## Historical pilot numbers (superseded)

Run `1c191d2dec0ca839` used 50 seeds, 1,600 follower samples, and a warm-up
budget of 100. It predates the corrected sampled-action OLS protocol: its
zero-sum estimator was given the exact expert leader distribution rather than
estimating that distribution from paired leader actions. The zero-sum row below
is therefore retained only as an audit record and must not be compared with the
current algorithms.

| Method | Final Stackelberg sub-opt (mean / median) |
|---|---|
| Active IRL (warm-up) | 0.00026 / 0.00008 |
| Active IRL (no warm-up) | 0.00018 / 0.00009 |
| Liao zero-sum OLS | 0.00009 / 0.00004 |
| Follower-only OLS (unaware) | 0.02715 / 0.02739 |
| Fixed random (warm-up) | 0.00135 / 0.00099 |
| Fixed random (no warm-up) | 0.00273 / 0.00169 |
| Random each round (warm-up) | 0.00133 / 0.00081 |
| Random each round (no warm-up) | 0.00231 / 0.00259 |

The follower-only OLS baseline never gets substantially better with more
samples: its equations have condition number ~13,700 vs. ~174 for Liao's
zero-sum-aware equations (verified by SVD in `application.py`'s validation),
so it's stuck ~100-300x worse than every other method regardless of budget.
This required loosening `FixedLeaderOLSConfig.ball_tolerance` from its 1e-10
default to 1e-6 and retaining the solver's 1e-12 root tolerance (see
`conf/experiment/guard_lobeke_paper.yaml`) to stop its ball-projection
postcondition check spuriously raising `ConvergenceError`
— verified this changes nothing about converged solution quality, only
numerical robustness in this specific ill-conditioned regime.

These inspected pilot numbers predate the protected tuning protocol and are not
final paper evidence.

The tracked flagship protocols use the frozen protected choices: warm-up budget
200 with `lambda=0.003 eta^2`, and no warm-up with `lambda=0.03 eta^2`. Their plotted
contrast is therefore a tuned-protocol comparison, not a causal warm-up
ablation.

## Lambda-only tuning

### Coarse joint discovery

Warm-up length and online lambda interact through the accumulated information
matrix, so lambda-only tuning at `warmup_budget=100` is conditional rather than
globally conclusive.  Before protected selection, the joint discovery study
crosses warm-up budgets `[0, 50, 100, 200, 400]` with lambda multipliers
`[1, 2, 4, 8, 16]` at the same total budget of 1,600 observed poacher actions.
All 25 candidates use the same 32 fresh seeds (`40000..40031`).

This stage deliberately produces a guarded shortlist rather than a chosen
hyperparameter.  It compares candidates with paired differences, excludes
large Fisher-error regressions relative to the current `(warmup=100,
lambda=eta^2)` configuration, and explicitly reports whether the shortlist
touches either grid boundary.  The final paper seeds remain protected.

```bash
make -C experiments/guard/paper joint-discovery
uv run mairl-status <run_dir> --watch
uv run python -m experiments.guard.paper.figures.joint_discovery <run_dir> \
  --output tmp/joint_discovery.png
```

The machine-readable result is `<run_dir>/joint_discovery.json`.  It must be
reviewed before defining the smaller protected selection/confirmation grid.

### Historical complete transfer-oriented grid (superseded figure input)

The earlier appendix figure used a complete Cartesian grid at a cheaper tuning horizon: 1,000
observed poacher actions, 20 paired seeds, warm-up thresholds
`[0, 50, 100, 200, 400]`, and information multipliers
`[1, 2, 3, 4, 6, 8, 12, 16, 24, 32]`. Existing protected trajectories are
reused at their exact 1,000-sample prefix. Only the missing cells are run:

```bash
make -C experiments/guard/paper full-grid-completion
```

The heatmap score is computed after learning: for every seed, SubOpt is
averaged over ten fixed structured downstream rewards, each positive-affinely
normalized to $[0,1]$ so that $R_{\max}=1$, that are
disjoint from the main-text tasks; uncertainty remains across paired seeds.
While the completion run is active, the watcher merges every newly completed
cell, refreshes the TikZ fragment, and recompiles the preview:

```bash
uv run python -m experiments.guard.paper.watch_full_grid <completion_run_dir>
```

The preserved compact audit data are written to
`reproducibility/data/lobeke/tuning_grid.{json,npz}`; the full resumable job
artifacts remain under `artifacts/runs/`.

Run `868e27df31c3f608` completed all 800 trajectories without failure in about
3.1 hours.  The current figure configuration `(warmup=100, lambda=eta^2)` had
mean final SubOpt `3.35e-4`.  The raw guarded winner `(400, 16 eta^2)` reached
`1.15e-4` with lower Fisher error, but it won on only 16/32 paired seeds and
lies on both upper grid boundaries.  The guarded shortlist is `(50, 8)`,
`(200, 1)`, `(200, 8)`, `(400, 8)`, and `(400, 16)`.  These are discovery
results—not a final selection—and require an upper-region expansion before
protected confirmation.

The sparse upper-region expansion, run `188d855f83ea9511`, completed all 256
trajectories on the same paired pilot seeds.  The raw optimum remained at
`(400, 16 eta^2)`: increasing lambda to `32 eta^2` gave mean SubOpt `1.32e-4`,
and warm-up budgets 600 or 800 gave means between `2.07e-4` and `2.65e-4` over
the tested lambda values.  Thus the apparent pilot optimum is no longer a raw
grid-boundary optimum.  The differences remain noisy, so these pilot seeds are
used only to define a conservative fresh-seed selection set.

Fresh paired selection uses seeds `10000..10063`, separately for the warm-up
and no-warm-up variants.  This separation gives each variant its own Fisher
reference and avoids imposing an arbitrary ordering between warm-up length and
lambda.  The run produces shortlists only: confirmation seeds `20000..20063`
and final paper seeds `30000..30049` remain untouched until a selection rule is
reviewed and frozen.

```bash
make -C experiments/guard/paper protected-selection
```

Both selections completed without failure.  Warm-up run `2ce0376fe643084a`
used 832 trajectories and found the interior raw winner `(200, 12 eta^2)`:
mean SubOpt fell from `2.38e-4` for the `(100, eta^2)` reference to `1.50e-4`,
while mean Fisher error fell from `0.0315` to `0.0272`.  The paired SubOpt
improvement was `8.80e-5` with standard error `6.78e-5`.

No-warm-up run `9e985f76d0b8ed3a` used 576 trajectories and found the interior
raw winner `lambda=3 eta^2`: mean SubOpt fell from `2.37e-4` to `1.62e-4`,
while mean Fisher error fell from `0.0325` to `0.0305`.  The paired SubOpt
improvement was `7.54e-5` with standard error `4.99e-5`.  These are selection,
not final evaluation, results; neither winner may be changed after inspecting
the untouched confirmation split.

### Current ridge-zero sensitivity figure

The current appendix heatmap matches the final estimator and budget exactly:
all likelihood fits use `ridge=0`, all cells use 1,600 samples, and the grid is
warm-up budgets `[0, 50, 200]` crossed with multipliers
`[0.003, 0.01, 0.03, 0.3, 1]`. It evaluates 100 fixed normalized downstream
tasks on eight paired post-freeze seeds (`94000..94007`). These seeds and tasks
are disjoint from final evaluation. The two final protocols are outlined from
their frozen definitions; the audit never chooses an outline from its observed
means.

Run or resume all 15 cells with:

```bash
make -C experiments/guard/paper ridge-zero-sensitivity
make -C experiments/guard/paper ridge-zero-confirmation
make -C experiments/guard/paper ridge-zero-export
```

The runner writes each complete candidate cell atomically. To split a long run
across local or cloud workers, give every worker a disjoint
`--candidate-pairs` list and a different `--output-dir`. Then pass every bundle
to the exporter with repeated `--input-dir` arguments. The exporter requires
the same ridge, sample budget, seeds, task policy, and source-configuration
checksum; it rejects conflicting duplicates and partial cells. Missing cells
are omitted from diagnostic exports and are never averaged or imputed. The
canonical paper export refuses an incomplete grid.

The tracked `reproducibility/data/lobeke/ridge_zero_tuning/` artifact contains
all 15 cells, every per-seed downstream score, paired intervals against the
corresponding frozen protocol, array checksums, and source-artifact checksums.
The heatmap treats the five tested multipliers as a discrete grid with
equal-width columns and uses a zero-anchored `[0, 4e-4]` color scale. The same
tracked payload includes an independent
20-seed, 100-task ridge-zero confirmation (`95000..95019`, task seed `9500`).
Its means recover `0.03` for no warm-up and `0.003` for warm-up 200; paired
warm-up contrasts cross zero, whereas both frozen Active-minus-fixed-expert
intervals lie below zero.

![Joint warm-up/lambda discovery](results/lobeke_joint_discovery.png)

### Protected selection and confirmation

Only the online information regularizer is varied:

```text
lambda = lambda_information_multiplier * eta^2
```

`lambda_warm=0.01` remains fixed and is recorded in every artifact as a
proof/certificate parameter; it does not alter finite-budget WARM execution.
For this normalized Lobeke problem, `eta^2=4.6875`;
the multiplier grid `[1, 1.5, 2, 3, 4, 6, 8, 12, 16]` therefore corresponds to
actual lambda values `[4.6875, 7.03125, 9.375, 14.0625, 18.75, 28.125, 37.5,
56.25, 75]`.

The study tunes the warm-up and no-warm-up variants separately.  It uses 64
selection seeds (`10000..10063`) and 64 disjoint confirmation seeds
(`20000..20063`); the final paper seeds (`30000..30049`) are rejected if they
overlap either split.  Selection minimizes final source-task
`stackelberg_suboptimality`, but a candidate is eligible only if its final
`fisher_transfer_error` is statistically no worse than the `lambda=eta^2`
reference.  Every candidate uses the same seeds within a split, so all tests use
paired per-seed differences rather than pretending the runs are independent.
Among candidates within one paired standard error of the eligible winner, the
smallest lambda is chosen.  Confirmation cannot reselect it, and a winner
must remain statistically no worse than the reference on both source SubOpt and
Fisher error.  A winner at the largest grid value is rejected rather than
silently accepting a likely boundary optimum.

Before leaving the full study unattended, exercise every candidate at the full
1,600-sample horizon on four fresh smoke-test seeds:

```bash
make -C experiments/guard/paper lambda-tuning-preflight
```

Then launch or resume the full study.  On macOS, `caffeinate` prevents ordinary
idle sleep while the command is running (keep the machine powered):

```bash
caffeinate -dimsu make -C experiments/guard/paper lambda-tuning
```

The run is content-addressed, records the resolved Hydra configuration and Git
state, persists each seed atomically, and resumes completed jobs after an
interruption.  The 2,304 trajectories are executed as 36 vectorized seed
batches.  A 6-seed, final-checkpoint benchmark took 50--65 seconds per warm-up
mode on the development machine, giving a deliberately conservative overnight
estimate of roughly 7--11 hours for the full grid.  The all-candidate preflight
itself completed 72/72 jobs with no failures in about 17.5 minutes.

The launch prints `run_dir=...`.  Progress and ETA can be watched from another
terminal, and the final selection can be plotted on demand:

```bash
uv run mairl-status <run_dir> --watch
uv run python -m experiments.guard.paper.figures.lambda_tuning <run_dir> \
  --output tmp/lambda_tuning.png
```

The decisive machine-readable output is `<run_dir>/lambda_selection.json`.
Any incomplete split, non-finite metric, failed confirmation, or upper-boundary
selection raises an explicit error instead of producing a chosen lambda.

## Reproducing

For a new checkout, `make smoke` checks short simulations and `make plots`
rebuilds all registered figures from tracked numeric data. See
[reproducibility](../../../reproducibility/README.md).

The paper-local Makefile exposes the few workflows a reader needs:

```bash
make -C experiments/guard/paper help

# Run or resume the final experiment (450 independently resumable jobs).
make -C experiments/guard/paper experiment

# Validate, then run or resume the protected lambda-only study.
make -C experiments/guard/paper lambda-tuning-preflight
make -C experiments/guard/paper lambda-tuning

# Regenerate the compact export and figure assets with the two --run-dir
# arguments shown in Current final evaluation above.

# Compile the tracked TikZ figures through the repository-wide preview.
make -C reproducibility preview
```

`preview` does not require the expensive run: the precise TikZ inputs and small
transfer summaries needed by LaTeX are tracked. Passing `RUN_DIR` to `figures`
loads the native results and verifies every source artifact before regenerating
the figures.

For a newly completed full comparison, use its printed directory rather than
copying a historical run ID. Export into new directories first:

```bash
RUN_DIR=artifacts/runs/guard-lobeke-paper/YOUR_RUN_ID
uv run python -m experiments.guard.paper.figures.export_final \
  --run-dir "$RUN_DIR" --output-dir artifacts/exports/lobeke-final
uv run python -m experiments.guard.paper.figures.export_source_convergence \
  --run-dir "$RUN_DIR" --output-dir artifacts/exports/lobeke-source-convergence
uv run python -m experiments.guard.paper.figures.export_behavioral_transfer \
  --run-dir "$RUN_DIR" --output-dir artifacts/exports/lobeke-behavioral-transfer
```

The last export evaluates the full downstream suite and can be expensive; it is
not part of the smoke check. The source-convergence export retains every
40-action checkpoint, while the broader diagnostic export thins checkpoints.

`run.py` prints `run_dir=...`; the two frozen final run IDs are
`349676e5922332fd` and `cf7b0f494e850f2d`. Raw runs are gitignored, so the
small composed arrays and generated figure inputs are committed with complete
per-run provenance.

The tracked Lobéké publication data are:

| Path | Produced from | Purpose |
|---|---|---|
| `reproducibility/data/lobeke/final_evaluation_unit_reward_v2/` | clean runs `349676e5922332fd` and `cf7b0f494e850f2d` | canonical per-seed target, certificate, and all-method taskwise transfer diagnostics |
| `reproducibility/data/lobeke/final_evaluation_native/` | all 450 jobs in earlier run `8c4ab05e9b2fe388` | preserved pre-retuning native export; not a canonical figure input |
| `reproducibility/data/lobeke/final_evaluation_corrected/` | 350 historical jobs plus 100 corrected OLS traces | preserved prior composite export; not a canonical figure input |
| `reproducibility/data/lobeke/paired_ols_correction/` | retained follower trajectories plus sampled paired leader actions | full 40-checkpoint corrected OLS traces and protocol provenance |
| `reproducibility/data/lobeke/final_evaluation/` | original final run `a4b1b2a1cdf886dd` | preserved pre-correction audit export; not a canonical figure input |
| `reproducibility/data/lobeke/tuning.json` | two protected selections and two confirmations | frozen-candidate audit and confirmation effects |
| `reproducibility/data/lobeke/ridge_zero_tuning/` | 15-cell post-freeze ridge-zero audit | canonical 1,600-sample sensitivity surface with per-seed paired intervals |
| `reproducibility/data/lobeke/tuning_grid.{json,npz}` | protected selection prefixes plus completion run `544713bd317e400d` | preserved superseded 50-cell, 1,000-sample surface |
| `reproducibility/data/lobeke/fixed_expert_mle_stackelberg_tuning.json` | 64-seed source-only selection and confirmation | Stackelberg-expert MLE ridge audit |
| `reproducibility/data/lobeke/environment/{data.npz,provenance.json}` | deterministic GUARD environment export | appendix environment schematic |

Regenerate the final and environment exports into temporary directories before
comparing them with the frozen copies:

```bash
uv run python -m experiments.guard.paper.figures.export_final \
  --run-dir artifacts/runs/guard-lobeke-paper-mle-corrected-v2/349676e5922332fd \
  --run-dir artifacts/runs/guard-lobeke-paper-ols-corrected-v2/cf7b0f494e850f2d \
  --output-dir /tmp/lobeke-final-evaluation-native

uv run python -m experiments.guard.paper.figures.export_environment \
  --output-dir /tmp/lobeke-environment
```

The exporters validate source manifests, paired-action counts and arrays,
result checksums, cache identity, shapes, dtypes, and finiteness before writing
derived data. Nonempty output directories are rejected.

The directory separates scientific execution from presentation:

```text
paper/
├── application.py, study.py, run.py   # simulation and resumable study
├── figures/                           # independent Python figure recipes
├── tex/                               # paper layouts and generated TikZ
└── results/                           # selected, versioned data and figures
```

Every figure is exported as both PNG (for this README and quick inspection)
and vector PDF (for LaTeX). No figure-level title, caption, panel letter, or
paper-specific narration is baked into the files; necessary data and category
labels remain. Compose the rest in LaTeX.

For the main-text preview, `figures/tikz.py` loads the GUARD data and calls the
central recipes in `reproducibility/figures/recipes/lobeke.py`. The repository-wide
`reproducibility/tex/preview.tex` inputs the resulting fragments directly and contains
placeholder prose so the figures can be judged at their actual paper width. It begins
directly with the numerical-validation section so the render shows the space occupied
on a normal body page.
Figure 2 is intentionally limited
to Active IRL and fixed-expert MLE, alongside an oracle row. This compares two
estimators under the same general logit-response model while isolating the value
of adaptive data collection; OLS remains in Figure 1 and the appendix because it
uses additional fixed-expert assumptions. Its reference-task column is
separated from near, median, and far behavioral shifts selected from the fixed
ten-task suite, and every
policy map uses the same size and color scale. Generation fails if the selected
baseline is not fixed-expert MLE.

`tex/layout_mockups.tex` composes those untouched assets into three paper-page
studies: mechanism-first, a compact application inset, and transfer-first. Its
rendered `output/pdf/layout_mockups.pdf` is only a design sandbox; the figures
remain independent.

## Results

### Main-text building blocks

The application story is exported as four independent assets so the paper can
arrange them without inheriting a Python layout.

**Environment** — Lobéké target density and ranger base:

![Environment](results/illustrations/environment.png)

**Interaction** — patrol coverage and induced poacher response:

![Interaction](results/illustrations/interaction.png)

**Learned object** — the three behavioral coefficients:

![Behavior model](results/illustrations/behavior-model.png)

**Transfer** — patrols optimized from the same behavioral model for three
leader objectives:

![Transfer patrols](results/illustrations/transfer-patrols.png)

**Selective performance comparison** — the central result only: Active IRL,
Liao's zero-sum estimator, random patrols, and follower-only OLS. The shaded
region and vertical rule mark Active IRL's first 200 samples; describe that in
the LaTeX caption rather than inside the image.

![Main performance](results/lobeke_paper_main_performance.png)

**A week of changing priorities** — the concrete transfer illustration. Each
row uses the true coefficients, a representative Active IRL estimate, or a
representative follower-only OLS estimate. Active IRL tracks the oracle patrol
over all five objectives, whereas follower-only OLS does not because its
coverage-interaction coefficient has the wrong sign (`[0.95, 0.90, 1.51]`
versus the true `[0, 1, -0.67]`).

![Temporal evolution](results/lobeke_paper_temporal_evolution.png)

### Appendix figures

**Main comparison** — all 8 method/warm-up arms, Stackelberg sub-optimality,
Fisher transfer error, and coefficient L2 error vs. follower samples:

![Main comparison](results/lobeke_paper_comparison.png)

**Downstream-task transfer** — the same run's already-collected trajectories
(no resampling), replayed against three behaviorally stratified members of the
fixed synthetic leader-reward suite. The suite uses
`draw_diverse_structured_tasks`: the
real reward's coverage-dependent functional form (reward `+value[b]` if target
`b` is covered, else `-value[b]`) with an independently random per-target
value, standing in for "a different downstream priority" -- not literal i.i.d.
noise per action pair, which almost always collapses to a degenerate pure
optimum and makes sub-optimality a discontinuous function of the noise rather
than of estimation quality (see `figures/data.py`'s docstrings for both task
generators, and the note below). The displayed tasks are the nearest, median,
and farthest source-policy shifts; the rule is independent of method outcomes:

![Downstream-task transfer](results/lobeke_paper_downstream_tasks.png)

**Environment maps** — same true omega, different (structured) leader reward,
shown on the actual park map:

![Environment maps](results/lobeke_paper_environment_maps.png)

*Aside — a real structural finding, not used for the figures above*: with
literally unstructured i.i.d.-noise downstream rewards (`draw_diverse_tasks`),
the optimal patrol collapses to a single pure action in effectively every
draw (verified: 20/20 random draws), while the real anti-poaching reward
yields a genuine mixed patrol (73%/13%/14%). That contrast is itself
interesting -- it's the classic security-game reason to randomize patrols,
and it shows up only because the real reward has structure aligned with the
follower's response, not because mixing is generically optimal.

**Archived warm-up threshold sweep.** The retained image is not scientific
evidence: its intermediate budgets use seeds `100..149` and
`lambda/eta^2=1`, while its endpoints reuse final seeds `30000..30049` with
separately tuned multipliers 3 and 12. It has therefore been removed from the
reproduction target. The valid replacement is a common-seed factorial
comparison of warm-up budget and lambda.

The replacement factorial combines native reference run `8c4ab05e9b2fe388`
with swapped-lambda runs `8513aac9cbe3189a` (the initial 20 paired seeds) and
`2d6a6b2f2d58eebd` (the disjoint 30-seed extension). Both swapped runs completed
without failure. At matched `lambda/eta^2=3`, the 50-seed paired
warm-minus-no-warm Target SubOpt effect is `-1.10e-4`, 95% CI
`[-2.12e-4, -8.19e-6]`; at multiplier 12 it is `9.12e-6`, CI
`[-5.81e-5, 7.63e-5]`. Averaged-transfer effects cross zero at both
multipliers. The result supports a lambda-dependent benefit on the source task,
not a universal transfer benefit. Compact arrays and paired effects are in
`reproducibility/data/lobeke/warmup_factorial_native_v2/`; the initial 20-seed
export remains preserved in the adjacent directory.
