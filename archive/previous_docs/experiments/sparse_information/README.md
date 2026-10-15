# Sparse-information experiment

This synthetic game is a deliberately favorable case for active exploration. It
tests whether the implementation can discover the only informative leader action
instead of spreading samples over many observationally useless alternatives.

It is an application outside `src/`: the file
[`application.py`](application.py) specifies the game and its metrics, while the
generic `mairl` and `mairl_exp` packages run the algorithms and experiments.

## Construction

The configured instance has:

- 16 leader actions and 2 follower actions;
- one unknown reward coefficient, with `omega = 1`;
- inverse temperature `eta = 2`;
- 200 follower samples over 100 seeds;
- either no warm-up or 40 common warm-up samples.

Only leader action zero changes the follower features:

```text
phi(0, 0) = 0       phi(0, 1) = 1
phi(a, b) = 0       for every a > 0
```

Let `x = mu[0]` be the probability assigned to the informative leader action. Then

```text
p_x = sigmoid(eta * omega * x)
I_mu(omega) = eta^2 * x^2 * p_x * (1 - p_x).
```

The other 15 leader actions are exact informational decoys.

## Why active exploration should win

At the initial estimate `omega_hat = 0`, the active information objective is
proportional to `x^2`, so it chooses `x = 1`. A policy drawn uniformly from the
16-action simplex instead satisfies

```text
E[x^2] = 2 / (16 * 17) = 1 / 136.
```

The active policy therefore has exactly **136 times more initial Fisher information**
than a uniformly random simplex policy. This is an initial information ratio, not a
claim that the final estimation error must improve by a factor of 136.

The construction also makes estimation matter for the leader. Action zero pays one
when the follower selects action one and zero otherwise. Every decoy guarantees a
safe payoff of `0.75`. Under the true reward,

```text
P(follower action 1 | leader action 0) = sigmoid(2) = 0.8808,
```

so the informative action is truly optimal. The estimated policy switches from a
safe decoy to the informative action once

```text
omega_hat > logit(0.75) / eta = 0.5493.
```

A seed that remains on the wrong side of this decision boundary has Stackelberg
suboptimality `sigmoid(2) - 0.75 = 0.1308`.

## Results

All methods receive the same total budget. The random baselines draw policies
uniformly from the simplex: one policy reused forever for fixed random, or a fresh
policy every round. Lines below are means over 100 seeds and bands are 95% confidence
intervals for the mean.

Final Fisher transfer error:

| Warm-up | Active | Fixed random | Random each round |
| --- | ---: | ---: | ---: |
| Disabled | **0.0558** | 0.4550 | 0.4183 |
| 40 samples | **0.0555** | 0.1297 | 0.1265 |

Final Stackelberg suboptimality:

| Warm-up | Active | Fixed random | Random each round |
| --- | ---: | ---: | ---: |
| Disabled | **0.0000** | 0.0471 | 0.0392 |
| 40 samples | **0.0000** | 0.0026 | 0.0013 |

The conclusions are:

1. **The implementation finds the informative action.** Without warm-up, active
   exploration has about 8.2 times lower final Fisher error than fixed random and
   7.5 times lower error than random each round.
2. **The estimation improvement changes the deployed policy.** Active reaches zero
   final Stackelberg suboptimality on all 100 seeds. Without warm-up, fixed random
   deploys the wrong action on 36 seeds and random each round on 30 seeds.
3. **The warm-up behaves exactly as intended.** Its D-optimal design assigns all 40
   samples to action zero. This gives every method informative data and reduces the
   baseline deployment failures to 2 and 1 seeds, respectively.
4. **Active still estimates more accurately after warm-up.** Its final Fisher error
   remains more than twice as small as either baseline, although all methods usually
   cross the leader's decision boundary.
5. **This is an existence check, not a representative benchmark.** The environment
   was intentionally constructed so that uniform exploration wastes almost all its
   mass. It demonstrates that active exploration can be substantially better when
   information is sparse; it does not establish the typical improvement on future
   applications.

## Plots

The dashed line marks the end of the common 40-sample warm-up.

### Estimator error in the Fisher metric

![Sparse-information Fisher transfer error](fisher_transfer_error.png)

### Error of the deployed Stackelberg policy

![Sparse-information Stackelberg suboptimality](stackelberg_suboptimality.png)

## Reproduce

Run or resume the original warm-up comparison:

```bash
uv run python -m experiments.sparse_information.run
```

Run or resume the frozen no-warm-up action-count sweep (paired seeds
`72000..72099`, `m = 2,4,8,16,32`):

```bash
uv run python -m experiments.sparse_information.run_action_count
```

Generate the two metrics from the resulting run:

```bash
uv run python -m experiments.plot \
  run_dir=artifacts/runs/sparse-information-comparison/RUN_ID

uv run python -m experiments.plot \
  run_dir=artifacts/runs/sparse-information-comparison/RUN_ID \
  plot=stackelberg_suboptimality
```

The images committed here come from the clean, completed 100-seed run
`39fda531e133aadd`. [`plots_manifest.json`](plots_manifest.json) records its Git
commit, source plot IDs, and checksums for the copied images.
