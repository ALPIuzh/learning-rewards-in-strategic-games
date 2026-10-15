# Lobeke experiment

The game uses a 5×5 grid with 23 patrol actions and 23 attack targets, built
from GUARD's Lobeke tracking data. The follower reward has three features:
coverage, target value, and their product.

The comparison includes Active IRL, fixed and per-round random policies,
fixed-expert MLE, zero-sum OLS, and follower-only OLS. Exploration methods run
with and without warm-up. MLE uses a fixed Stackelberg-optimal leader; OLS uses
a joint-QRE expert; zero-sum OLS additionally uses sampled leader actions.

The retained experiment uses 50 seeds and 1,600 follower observations per seed.
All MLE methods use `ridge=1e-6`. Active IRL uses a 50-sample warm-up and
`lambda/eta^2=0.03`, or no warm-up and `lambda/eta^2=0.1`. Evaluation leader
rewards are normalized to `[0,1]`.

## Reproducing

From the repository root:

```bash
uv run python -m experiments.run --config-name runs/lobeke/comparison --cfg
make experiment
```

The launcher prints the run directory. Export its results with:

```bash
uv run python -m experiments.guard.paper.figures.export_final \
  --run-dir <run-directory> --output-dir <new-export-directory>
```

Pass `--run-dir` more than once to combine runs containing different methods.
The current figure data are in
`reproducibility/data/lobeke/final_evaluation_unit_reward_v2/`.

The paper heatmap uses 32 seeds per cell and MLE ridge $10^{-6}$; the main
comparisons use its selected settings on fresh seeds and separate tasks. `make plots` renders both from the included data.

For short checks and all paper plots, run `make smoke` and `make plots`.
See [reproduction instructions](../../../reproducibility/README.md) for the
figure registry, exports, and LaTeX commands.

The published losses use a separate scalar deployment evaluation with 64 starts
and tolerance `1e-12`, checked on selected cases with 256 starts. The private
[campaign archive](../../../archive/final-evaluation-20260921/README.md) retains
all raw runs, evaluation scripts, and checks. `make plots` needs only the compact
figure data and does not rerun optimization.
