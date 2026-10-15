# Zero-sum estimator comparison

This external application compares reward estimation methods at the same budget of
200 follower responses:

1. Active IRL with a warm-up budget frozen on separate tuning seeds;
2. fixed-leader OLS using only the follower best-response equations;
3. Liao's zero-sum OLS using both players' regularized-QRE equations.

Both OLS methods use the same follower samples for every seed. Active IRL receives the
same number of follower responses but chooses its leader policies adaptively.

## Game

The experiment reuses the nontrivial three-leader-action, four-follower-action,
three-feature toy geometry. The follower reward is

`r_f(a,b) = <phi(a,b), omega>`,

and the leader reward is its negative. Both players' QRE temperature is `eta = 1.25`,
equivalently an entropy coefficient of `1 / eta = 0.8`.

The fixed expert QRE is checked numerically before any samples are drawn. Its follower
probabilities range from 0.154 to 0.385. The follower-only OLS system has rank 3 and
condition number 11.25; adding the zero-sum leader equations preserves rank 3 and
improves the condition number to 2.90.

## Metrics

The primary metric is

`||omega_hat - omega||_{I_mu_expert(omega)}`,

the one-sample Fisher norm induced by Liao's fixed expert leader and evaluated at the
true reward parameter. This is the natural in-distribution estimation error for the
fixed-leader protocol.

Two secondary metrics prevent that local norm from hiding other errors:

- coefficient L2 error;
- L1 error of the leader policy in the two-sided entropy-regularized QRE reconstructed
  from the reward estimate.

## Warm-up tuning

The historical results below tuned only the warm-up budget. The final paper
refresh instead tunes warm-up and the online information multiplier jointly on
the protected selection seeds `40000..40063`, freezes a shortlisted candidate,
confirms it on `50000..50063`, and reserves `60000..60099` for final evaluation.
The theorem-motivated `(K0=0, lambda/eta^2=1)` configuration remains the
predeclared reference. Downstream tasks never participate in selection.

Run the joint selection stage with:

```bash
uv run python -m experiments.zero_sum_comparison.run_joint_selection
```

The predeclared local refinement around the coarse warm-up optimum uses the
same selection split and can be reproduced with:

```bash
uv run python -m experiments.zero_sum_comparison.run_joint_selection \
  --config-name zero_sum_joint_refinement_config
```

The refinement froze `(K0=160, lambda/eta^2=8)` before inspecting confirmation
seeds. Confirm that candidate against the predeclared reference with:

```bash
uv run python -m experiments.zero_sum_comparison.run_joint_selection \
  --config-name zero_sum_joint_confirmation_config
```

The candidate matched the reference's fixed-leader Fisher error on confirmation,
but failed the predeclared source-policy guardrail. We therefore do not reselect
using confirmation outcomes: the final comparison uses the reference
`(K0=0, lambda/eta^2=1)` on the untouched `60000..60099` seeds. This is the
outcome of the tuning protocol, not an assumption that the theorem value is
universally optimal.

The older warm-up-only procedure and results are retained below as historical
context and for reproducing the exploratory figures; they are not the refreshed
paper results.

Warm-up candidates are `0, 24, 40, 80, 120, 160, 200`, always within the same
200-sample total budget. Selection uses 64 seeds, confirmation uses 64 disjoint seeds,
and the final comparison uses 100 untouched seeds.

The selection split minimizes final fixed-leader Fisher error. Among candidates within
one standard error of the raw winner, it chooses the smallest warm-up to preserve more
active rounds. The frozen choice must then be statistically consistent with both the
raw winner and the predeclared no-warm-up baseline on the confirmation split.

Run `de1038d32a1121e8` selected **160 warm-up samples**. Its mean final primary error
was 0.05685 on selection seeds, versus 0.06190 without warm-up. On confirmation seeds,
160 and zero warm-up had means 0.05682 and 0.05538 respectively: the predeclared
statistical check passed, although the mean improvement did not replicate. The budget
remains frozen at 160 rather than using confirmation results to reselect it.

Run tuning:

```bash
uv run python -m experiments.zero_sum_comparison.run_tuning
```

Run the frozen configuration on the untouched comparison seeds:

```bash
uv run python -m experiments.zero_sum_comparison.run
```

Export plots from the two content-addressed run directories:

```bash
uv run python -m experiments.zero_sum_comparison.export_plots \
  --tuning-run artifacts/runs/zero-sum-active-warmup-tuning/<run-id> \
  --comparison-run artifacts/runs/zero-sum-estimator-comparison/<run-id>
```

The result artifacts are resumable and Git-versioned. Generated plots and their small
manifest live in this folder; raw per-seed results remain under `artifacts/runs/`.

## Results

Frozen configuration, 100 comparison seeds, 200-follower-sample budget, mean with 95%
CI (see `results_manifest.json` for the full numbers):

| Metric | Active IRL | Liao zero-sum OLS | Fixed-leader OLS |
| --- | --- | --- | --- |
| Fixed-leader Fisher error (primary) | 0.0732 [0.066, 0.080] | 0.0992 [0.089, 0.110] | 0.1066 [0.098, 0.116] |
| Coefficient L2 error | 0.227 [0.207, 0.247] | 0.301 [0.269, 0.333] | 0.924 [0.815, 1.033] |
| Regularized-QRE leader L1 error | 0.0397 [0.035, 0.044] | 0.0287 [0.024, 0.033] | 0.200 [0.172, 0.228] |

Active IRL has the lowest Fisher error and coefficient L2 error, beating Liao's
zero-sum OLS by about 26% and 25% respectively (both seed-paired win rates above
60%, CIs on the paired difference exclude zero). Liao's OLS instead reconstructs the
leader's policy slightly more accurately (lower L1 error, Active wins only 33/100
seeds there), despite its noisier reward-parameter estimate. Both OLS variants that
use the leader's zero-sum equations dominate the follower-only OLS baseline across
every metric.

![Estimator comparison](estimator_comparison.png)

![Active IRL warm-up tuning](warmup_tuning.png)

## Downstream-task transfer

The zero-sum comparison above evaluates each estimate only against the same fixed
expert's leader policy it was fit on. To test transfer, we take each method's final
(200-sample) `omega_hat` for all 100 seeds and reuse it to plan against 3 held-out
leader-reward matrices over the same game (features, `eta`, `radius` unchanged) that
are unrelated to the zero-sum construction — independent random `(3, 4)` payoff
matrices, entries in `[-1, 1]`, selected from a pool of 200 candidates to be
pairwise near-orthogonal (max cosine similarity 0.014, seed `20260731`). This is the
object the Stackelberg reward-transfer lemma is stated over (`docs/algorithm-spec.md`),
so it directly tests whether Active IRL's estimation-error advantage generalizes to
planning objectives the estimator never saw.

Two metrics per task, mean over 100 seeds with 95% CI (`downstream_tasks_manifest.json`
has the full numbers, including per-seed paired win rates):

- **Fisher-transfer error** — the lemma's estimator term for that task's leader
  reward. **Active IRL wins on all 3 tasks against both baselines**, averaging 0.116
  versus 0.162 for Liao's zero-sum OLS (~29% lower) and 0.428 for the follower-only
  OLS, with 57-70% paired win rates against Liao and CIs on the paired difference
  excluding zero on every task.
- **Stackelberg suboptimality** — the literal deployed leader-value loss versus the
  true optimum. All three methods lose almost nothing in absolute terms on this
  3-action game (means of order `1e-3` to `1e-1`): many seeds land exactly on the
  optimal vertex regardless of estimator, so per-seed win rates are noisy and
  sometimes favor Liao's OLS even where its mean loss is higher (e.g. task 1's mean
  is pulled up by a heavier tail of bad follower-only-OLS estimates). Take this
  metric as a sanity check on the Fisher-transfer numbers rather than a second
  independent verdict.

Net: Active IRL's advantage on Fisher error and coefficient L2 error (see Results
above) is not an artifact of the fixed expert's specific policy — it transfers to
new downstream leader objectives, which is exactly where Liao's OLS advantage on
this experiment's one in-distribution metric (leader-policy L1 error) does not.

![Downstream-task transfer](downstream_tasks.png)

### Leader-policy transfer over the estimation trace

The plot above only looks at the final (200-sample) estimate. This one tracks, at
every evaluation checkpoint, the L1 distance between the leader policy *planned*
from `omega_hat` for a task (via `optimize_leader_policy`) and the true optimal
leader policy for that same task — the direct downstream-task analogue of the
`regularized_qre_leader_l1_error` panel in the Results plot above, which is reused
unchanged for the leftmost (boxed) panel here since it is already the in-distribution
version of the same question.

![Leader-policy transfer across tasks](downstream_tasks_leader_l1_error.png)

This is where the mixed in-distribution result (Results section: Liao's OLS has
*lower* leader-policy L1 error than Active IRL on the original task) gets directly
tested out of distribution, and it does not hold up: on downstream task 1 and task 3,
Active IRL's planned leader policy is closer to optimal than Liao's at every shared
checkpoint; on task 2 both recover the exact optimal leader policy (a dominant action
independent of the reward estimate), while the follower-only OLS baseline does not.
So Liao's one in-distribution edge is specific to the fixed expert's own policy —
it does not transfer, while Active IRL's estimates plan comparably-or-better leader
policies for every downstream task tested.

Reproduce both downstream-task plots and their manifest from the same comparison run
used above:

```bash
uv run python -m experiments.zero_sum_comparison.downstream_tasks \
  --comparison-run artifacts/runs/zero-sum-estimator-comparison/<run-id>
```
