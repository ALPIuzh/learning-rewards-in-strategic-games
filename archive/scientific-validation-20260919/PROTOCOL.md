# Prospective scientific validation — 19 September 2026

This protocol is fixed before generating the confirmation trajectories or inspecting
their outcomes. Existing experiments, figures, and raw data are preserved. The
400-observation timing pilot uses learning seeds 2026091900–2026091901 and task
seed 202609190; it is excluded from all confirmation analyses.

## Questions and design

The primary question is whether each previously retained Active IRL protocol has
lower mean final transfer suboptimality than fixed-expert MLE on a fresh Lobeke
task suite. Retained settings are no warm-up with information multiplier 0.03 and
200 warm-up observations with multiplier 0.003. No hyperparameters will be selected
from the new confirmation results.

All methods receive 1,600 follower observations. Fifty learning seeds,
2026092000–2026092049, are shared across methods using the existing purpose-separated
random streams. MLE ridge is zero. The resolved source configuration is
`reproducibility/data/lobeke/ridge_zero_tuning/source_config.yaml`; all other
application, inference, and optimization settings remain as recorded there.
The code starts from efficiency commit dc54c21, whose outputs passed paired
numerical-equivalence checks against the earlier implementation.

The four Active IRL cells cross warm-up budgets {0, 200} with multipliers
{0.003, 0.03}. Controls are fixed-expert MLE, fixed random and online random
(each with and without 200 warm-up observations), follower-only OLS, and
zero-sum OLS. Random methods use the retained corresponding warm-up/no-warm-up
settings. OLS uses its existing joint-QRE observation model; zero-sum OLS also
observes the leader action. These are comparisons of complete protocols with
different information assumptions, not a controlled estimator-only comparison.

The primary transfer suite contains 100 rewards selected from 1,000 generated
structured candidates using task seed 202609200. Two secondary suites contain
50 rewards each from 1,000 candidates, with seeds 202609201 and 202609202.
Use the existing payoff-diversity selection rule and normalize each reward to
[0,1]. These are diversity-selected synthetic priorities in one fixed game,
not independently sampled real-world environments. Save the exact task matrices.

## Analysis fixed before confirmation

The primary endpoint is transfer suboptimality at 1,600 observations, averaged
over the primary task suite within each learning seed. Report paired mean
differences for the two retained Active protocols minus fixed-expert MLE,
two-sided paired t tests with Holm correction across those two tests, and
Bonferroni simultaneous 95% confidence intervals (97.5% marginal intervals).
An interval strictly below zero supports a lower mean on this fixed suite.
Always report absolute effects, method means, and unadjusted 95% intervals too.
Do not treat seed × task pairs as independent replicates.

Secondary analyses: matched warm-up minus no-warm-up contrasts at each multiplier,
the factorial interaction, retained protocols versus random/OLS controls,
source-task performance, and both secondary task suites. These are descriptive
with explicitly unadjusted paired 95% intervals. A confidence interval including
zero is inconclusive, not evidence of equivalence. Do not choose the best cell
and call it independently confirmed. Do not pool task suites as independent
replications of learning seeds. Inspect skew and seed-level distributions;
paired bootstrap intervals with a fixed analysis seed provide a sensitivity
check, not an additional decision rule.

## Numerical checks

Before the full run, compare the standard batched deployment optimizer with the
full scalar multistart optimizer on pilot estimates and pilot tasks. After the
full run, repeat this check on the first three primary tasks and first two seeds
of every method, regardless of performance. Use stricter tolerance and 64 random
starts. This checks sensitivity, not global optimality. Record maximum oracle
value changes and deployed-value changes. If numerical discrepancies exceed
1e-6 in normalized reward, extend evaluation checks before interpreting effects;
do not silently replace the original evaluation. Report this numerical caveat
even if the aggregate statistical conclusion is stable.

Validate saved shapes, finiteness, budgets, seeds, simplex policies, MLE residuals,
and hashes. Preserve raw trajectories, taskwise endpoints, diagnostic records,
commands, timings, environment versions, and source revision. Verify the original
data hashes again at completion.

## Compute budget and stopping rule

The timing pilot trained four Active cells and fixed-expert MLE in 23.7 seconds
at two seeds and 400 observations. Historical full runs used about 22 minutes
for 350 MLE jobs at 1,600 observations. Allow 1–3 hours for the 550 confirmation
jobs, transfer evaluation, and numerical checks, including conservative overhead.
Train in ten-seed batches and save every completed batch. Evaluate final estimates
only; do not recompute all learning-curve checkpoints on all new transfer tasks.

Run the fixed sample size, with no significance-based early stopping or sample
extension. Stop and diagnose numerical failures, invalid data, or a three-hour
compute overrun; preserve all completed work and report any incomplete protocol.
Any necessary amendment must be recorded before the affected work starts, with
the reason and what results had already been inspected. No additional tuning is
planned: the main uncertainty concerns confirmation, not finding a more favorable
setting. Existing data must never be overwritten.

## Pilot amendment, before confirmation

The 30 pilot deployment checks found a maximum oracle-value change of 6.6e-11
and fitted-objective improvement of 3.1e-9, but a deployed-value change of 2.4e-6.
This exceeds the numerical diagnostic threshold above. Expand the confirmation
check to the first ten primary tasks and first five seeds of all eleven methods
(550 fixed checks). These indices are chosen before any confirmation outcomes
are available. Preserve both evaluations and quantify changes to method contrasts.
Small objective errors can change deployment value; do not claim certified global
solutions or treat the diagnostic threshold as a formal error bound.
