# Toy experiment

A synthetic game for comparing exploration rules, warm-up budgets, and follower
temperatures.

## What was compared

Each run uses the same total budget of 1,000 follower samples and the same 100 seeds.
The three methods are:

- **Active information:** the paper's adaptive exploration rule.
- **Fixed random:** draw one leader policy uniformly from the simplex, then reuse it.
- **Random each round:** draw a fresh leader policy uniformly from the simplex.

When the warm-up budget is positive, all three methods receive the same grouped
D-optimal warm-up observations. They only choose different policies after the dashed
line. This makes the sample comparison fair.

The plotted metric is the estimator term in the Stackelberg transfer lemma:

```text
max over μ in {μ*, μ_hat} of ||omega - omega_hat||_{I_μ(omega_hat)}
```

Lower is better. Lines are means over 100 seeds; bands are 95% confidence intervals
for the mean, drawn with low opacity. All threshold plots share the same axes.

## Main conclusions

1. **Active exploration is useful even without warm-up.** At 1,000 samples its mean
   error is `0.0427`, versus `0.1871` for fixed random and `0.0603` for random each
   round.
2. **The initially surprising warm-up plot was a phase-allocation effect.** Before
   the dashed line the methods are exactly the same. A longer warm-up gives every
   method more common informative data and leaves fewer samples in which their online
   policies can differ.
3. **Active is quite insensitive to the tested warm-up size.** Its final mean stays
   between `0.0379` and `0.0431`. The lowest observed mean is at 240 warm-up samples,
   but a seed-paired 95% comparison against every other threshold includes zero.
   Therefore 240 is **not** a statistically justified tuned default.
4. **Warm-up mainly rescues the weak fixed-policy baseline.** Its final error falls
   from `0.1871` without warm-up to `0.0582` with 240 samples and `0.0433` with 800.
5. **The methods must coincide at a warm-up budget of 1,000.** There is no online
   phase, so all three receive exactly the same data. The implementation check found
   identical policies, follower samples, MLEs, and final metric (`0.0426`).
6. **The warm-up design is sensible for this toy game.** It uses all three pure
   leader actions, with weights `[0.4232, 0.4232, 0.1536]`. At budget 120 this becomes
   counts `[51, 51, 18]`. The reduced coverage eigenvalues are
   `[15.98, 25.10, 36.62]`: the design is full-rank and not concentrated on one
   policy.

Final mean Fisher transfer error:

| Warm-up | Active | Fixed random | Random each round |
| ---: | ---: | ---: | ---: |
| 0 | 0.0427 | 0.1871 | 0.0603 |
| 24 | 0.0431 | 0.1053 | 0.0597 |
| 60 | 0.0404 | 0.0781 | 0.0594 |
| 120 | 0.0396 | 0.0741 | 0.0548 |
| 240 | **0.0379** | 0.0582 | 0.0517 |
| 500 | 0.0430 | 0.0459 | 0.0512 |
| 800 | 0.0417 | 0.0433 | 0.0436 |
| 1,000 | 0.0426 | 0.0426 | 0.0426 |

The bold number is only the smallest observed active mean, not a selected
hyperparameter.

## What is the theoretical warm-up threshold?

There are two different thresholds, and they should not be confused.

The implementation's deterministic integer-rounding condition is

```text
ceil((1 + epsilon) * support_size / epsilon)
```

With `epsilon = 0.15` and support size 3, the minimum is **23 samples**. The sweep
starts at 24 simply because 24 was the chosen first grid point.

The paper's probabilistic certificate is much more conservative. The event is

```text
d_*(omega, omega_hat_k) <= 1 / eta for every online step k
```

with probability at least `1 - delta`. For probability 0.9, set `delta = 0.1`.
Using this toy's `eta = 1.25`, `S = 2`, `|B| = 4`, `d = d_diamond = 3`, the leading
expression in the draft,

```text
exp(2 eta S) |B| d_diamond d log(exp(eta S) / delta),
```

is approximately **25,660** before the hidden constants and polylogarithmic factors
in the `tilde-O`. Consequently, this expression alone does not define one precise
numeric threshold. As a second reference point, setting the draft's hidden constant
`C = 1` in its final displayed sufficient condition and conservatively taking
`K = 1,000` gives approximately **2.49 million samples**.

The honest theoretical conclusion is therefore: the 90%-probability worst-case
certificate is on a scale from tens of thousands to millions, depending on how the
hidden proof constants are instantiated. It is far beyond this 1,000-sample
experiment and is evidently very loose for the toy instance. The plot's 95%
confidence intervals concern empirical means and are unrelated to the theorem's
90%-probability event.

## Trajectories

### No warm-up

![No warm-up comparison](warmup_0000_fisher_transfer_error.png)

### 24 warm-up samples

![24-sample warm-up comparison](warmup_0024_fisher_transfer_error.png)

### 60 warm-up samples

![60-sample warm-up comparison](warmup_0060_fisher_transfer_error.png)

### 120 warm-up samples

![120-sample warm-up comparison](warmup_0120_fisher_transfer_error.png)

### 240 warm-up samples

![240-sample warm-up comparison](warmup_0240_fisher_transfer_error.png)

### 500 warm-up samples

![500-sample warm-up comparison](warmup_0500_fisher_transfer_error.png)

### 800 warm-up samples

![800-sample warm-up comparison](warmup_0800_fisher_transfer_error.png)

### 1,000 warm-up samples

![1,000-sample warm-up comparison](warmup_1000_fisher_transfer_error.png)

## Reproduce

Run or resume the source experiments:

```bash
uv run python -m experiments.toy.run
uv run python -m experiments.toy.run_warmup_sweep
```

The paper appendix uses a separate protected refresh on seeds `82000..82099`
and includes every warm-up threshold in one resumable run:

```bash
uv run python -m experiments.toy.run_warmup_final
```

Its inverse-temperature sensitivity study keeps the features, reward, sample
budget, and warm-up fixed, uses new paired seeds `83000..83099`, and enforces
`lambda = eta^2` at every temperature:

```bash
uv run python -m experiments.toy.run_temperature_final
```

Both final exporters validate every raw checksum and retain all per-seed values
used by the registered paper plots. Full traces remain in the local raw run, so
alternative diagnostics can be exported without rerunning either experiment.

Regenerate these eight images from the exact completed runs used above:

```bash
uv run python -m experiments.toy.export_warmup_threshold_plots
```

For different run IDs:

```bash
uv run python -m experiments.toy.export_warmup_threshold_plots \
  --reference-run artifacts/runs/baseline-comparison/REFERENCE_RUN_ID \
  --sweep-run artifacts/runs/toy-warmup-sweep/SWEEP_RUN_ID
```

The exporter refuses incomplete, corrupted, duplicate, or unequal-budget inputs.
[`warmup_threshold_plots_manifest.json`](warmup_threshold_plots_manifest.json) records
the two source manifests, a digest of all 2,400 result files, the numerical summary,
and every output-image checksum.
