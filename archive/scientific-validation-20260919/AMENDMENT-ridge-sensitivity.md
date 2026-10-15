# Additional fixed-expert regularization sensitivity

This exploratory check was added after confirmation training began. It is not
part of the original prespecified analysis and must not be presented as an
independent confirmatory test. The original primary protocol and its two
hypothesis tests remain unchanged.

The historical audit reports substantial fixed-expert improvement from ridge 1,
an earlier source-task-selected setting. A comparison only with unregularized
MLE would not establish superiority over a well-regularized fixed-expert learner.
Refit the exact fresh fixed-expert observations at final budget 1,600 using ridge
{0.01, 0.1, 1, 10}; retain the existing ridge-zero estimate as the reference.
The grid includes the historically used value and its neighboring orders of
magnitude. Fit from zero, without using the true reward as initialization.

Evaluate all four settings on the already fixed primary 100-task suite. Report
every grid value, paired differences from both retained Active protocols, and
unadjusted 95% intervals. This is a secondary sensitivity analysis, not tuning
followed by independent testing. Do not choose a winner and call its score an
unbiased estimate of selected-protocol performance. The historical ridge-1 value
is highlighted as a previously used control, not selected from the new data.

No trajectories change. New files live under `ridge-sensitivity/` in the new
artifact directory. Expected cost is under ten minutes: 200 final MLE fits and
20,000 task-specific deployments, with per-task checkpoints. Stop and diagnose
numerical errors rather than silently changing an estimator or discarding seeds.
If the additional grid cannot finish within ten minutes, reassess the remaining
cost before extending it. The overall three-hour compute budget still applies.
