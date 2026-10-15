# GUARD security games

This external application connects real-world security-game benchmarks from the
[GUARD](https://github.com/CoffeeAndConvexity/GUARD) library to the paper's
active Stackelberg IRL algorithm. `application.py` is the adapter boundary
(`GuardApplication`, following the same shape as [`experiments/toy`](../toy)); it
depends on `mairl`/`mairl_exp` but neither depends back on it.

GUARD is an independent public project by CoffeeAndConvexity. It is not authored or
maintained as part of MAIRL. This repository records the upstream checkout as a pinned
Git submodule at `experiments/guard/GUARD`; we do not copy, patch, or repackage its
source. MAIRL-specific loading and compatibility behavior belongs in this directory.

## What is compatible, and what is not

GUARD's `SecurityGame.schedule_form(generate_utility_matrix=True)` output is exactly the
shape the paper's setup needs: a finite leader action set (defender patrol schedules), a
finite follower action set (targets), and a payoff matrix. `SoftmaxFollowerOracle`
supplies the entropy-regularized follower response independently of GUARD's own solvers
(GUARD computes SSE/Nash equilibria for a rational attacker; we never call those — the
bounded-rational response is entirely ours, applied on top of GUARD's payoffs).

Two GUARD game families are loaded straight from
`GUARD/predefined_games/*.pkl`:

| family | file pattern | domain | targets | defender patrols |
|---|---|---|---|---|
| Green Security Game (anti-poaching) | `etosha_sfg_gsg_*`, `lobeke_sfg_gsg_*` | Etosha NP (Namibia), Lobeke NP (Cameroon) | 7-12 | 121-24,389 |
| Infrastructure Security Game | `*_chinatown_sfg_isg_*` | NYC/NJ power grid | 8-13 | 23-50 |

The `_nfg_*` pkl variants (plain zero-sum matrices, no `target_utilities`) are **not**
usable — they don't carry the per-target decomposition the feature map needs, and
loading them with `feature_mode="behavioral"` raises `ConfigurationError`.

## Feature maps

Two of the three `feature_mode` options were already near-tabular:

- `"tabular"`: `d = n_leader * n_follower`, one coefficient per cell. Not interesting.
- `"target_coverage"`: `d = 2 * n_targets`, one covered/uncovered coefficient pair per
  target. Better, but still scales with the number of targets, not a genuine
  low-dimensional structural model.

**`"behavioral"` (new, this session) is a real 3-dimensional SUQR-style map:**

```text
phi(a, t) = (coverage_indicator, target_value, coverage_indicator * target_value)
```

This falls out of GUARD's own generator, not an invented simplification: every bundled
game builds `attacker_covered = attacker_uncovered / attacker_penalty_factor` with a
**single scalar** `attacker_penalty_factor` shared across all targets (verified on every
predefined game: 5.0 for the GSG games, 3.0 for the ISG games). Under that structure the
true reward is *exactly* `omega = (0, 1, -1/penalty_factor)` — confirmed by fitting the
3-dim model to GUARD's own payoff matrix via least squares and checking the residual is
at machine precision (`~1e-12`) on every game, not merely approximately linear. Ground
truth is derived this way (not hard-coded) so the mode degrades gracefully — with a
raised, reported fit error — if ever pointed at a future GUARD game whose penalty factor
turns out not to be uniform across targets.

`d=3` and the games have `T=7-13` targets, so `Phi(mu)` is generically full rank
(`identifiable_dimension == 3` on every game checked): exact non-identifiability
(Lemma 3) is a constructed corner case here, not something that occurs naturally, matching
`experiments/ssg/ssg-experiment-spec.md`'s equivalent observation for the synthetic
Nguyen et al. benchmark (§10 there).

## A core-library performance fix, not just a GUARD-local one

`optimize_leader_policy` (`mairl/planning/leader.py`) solves the leader's non-concave
Stackelberg objective with `1 + n_leader_actions + random_starts` independent SLSQP
solves — one full local optimization per pure-action vertex start, sequential, no
batching across checkpoints or seeds. That's fine at `n_leader_actions=23` (Chinatown)
but at `n_leader_actions=169` (Etosha) a single evaluation call took ~15s and a full
checkpoint's worth of metrics (`stackelberg_suboptimality` + `fisher_transfer_error`,
each calling it independently) took ~30s — an 8-seed/400-sample run that should complete
in under a minute did not finish in over an hour.

Added `optimize_leader_policy_batch`, mirroring the existing
`select_information_policy_batch` pattern in `mairl/exploration/policy.py`: a vectorized
projected-gradient prefilter across every start for every problem at once, followed by
exact SLSQP refinement on only the top `batch_refine_starts` candidates per problem.
Wired into `StackelbergFisherError.evaluate_many` and this adapter's
`evaluate_estimates`, so every application that evaluates a checkpoint trace benefits,
not just this one. Verified to match the sequential optimizer's values to ~`1e-12` on
both a synthetic problem and the real Etosha game, with a **161x** speedup on Etosha.
Tests: `tests/test_gradients_and_solvers.py::test_batched_leader_policy_matches_sequential`
and `..._grid_fallback_matches_scalar`.

## Methodology: what "the random baseline" should and shouldn't get

The offline warm-up (`mairl.exploration.design.compute_warmup_design`, Algorithm 4 in the
paper) is a Frank-Wolfe D-optimal experimental design computed purely from the geometry
of the feature map — it is not random, and not specific to active exploration. Giving it
to a random baseline "for free" (`warmup_enabled=True` for every method, as
`BaselineComparisonStudy`'s `warmup_modes` sweep can do symmetrically) mechanically
narrows the visible active-vs-random gap, because a large, well-conditioned fraction of
the total sample budget is then identical between methods. On Chinatown this compressed a
true ~3x gap down to ~1.6x in an early version of this validation.

The comparison in `behavioral/validate_behavioral.py` is therefore asymmetric by design,
matching `experiments/zero_sum_comparison`'s estimator-comparison figure: **Active IRL
keeps its own required warm-up** (it's a load-bearing precondition of Algorithm 2's
finite-sample guarantee, not optional), while **the random baseline gets none** — nobody
would hand-design a D-optimal experiment for a policy about to act uniformly at random
anyway. The warm-up budget (120, out of a 400-sample total budget) is empirically tuned
for Chinatown via `behavioral/tune_warmup.py` (a lightweight sweep over
`{0, 40, 80, 120, 200, 300}`, both with and without the shared-warmup arm); Etosha reuses
it as a reasonable default pending its own sweep.

## Results

32 seeds, 400-sample total budget, `feature_mode="behavioral"`. Active with its own
120-sample warm-up vs. random-each-round with no warm-up.

| game | active final SubOpt | random final SubOpt | ratio |
|---|---:|---:|---:|
| Chinatown (23 actions) | 0.00119 | 0.00370 | 3.1x |
| Etosha (169 actions) | 0.00052 | 0.00270 | 5.2x |

![Chinatown comparison](behavioral/behavioral_comparison_nw_chinatown_sfg_isg_1.png)
![Etosha comparison](behavioral/behavioral_comparison_etosha_sfg_gsg_1.png)

Coefficient recovery (true `omega = (0, 1, -1/penalty)`) separates just as cleanly; see
`behavioral/behavioral_coefficients_*.png`.

The Stackelberg sub-optimality bands use the median with a 10th-90th percentile band
rather than mean +/- 1.96 SE: the metric is bounded at zero and occasionally exactly zero
for individual seeds on these well-conditioned small problems, which breaks the normal
approximation and produces a floor-clipping artifact on a log-scale plot.

## Geographic map (Tier 3)

`geographic/map_lobeke.py` regenerates `GreenSecurityGame` directly from the upstream
submodule's `GUARD/data/lobeke/*.csv` elephant-tracking data (fully offline after
checkout; requires the `guard`
optional dependency group: `uv sync --extra guard`) instead of loading the precomputed
pkl, recovering real lat/lon grid positions that the pkl doesn't carry. It solves for the
optimal single-patrol defender coverage against the exact behavioral `omega` and plots it
against the poacher's resulting attack distribution on the actual park grid:

![Lobeke map](geographic/lobeke_map.png)

The behavioral feature map fits this freshly-generated game to machine precision
(`~3.6e-16`) exactly as it does the precomputed games, and recovers the same
`omega = (0, 1, -2/3)` as Chinatown (same default `attacker_penalty_factor=3`). With a
single defender unit and singleton-target patrol schedules, the optimum concentrates
73% of coverage on the single highest-density cell, with smaller shares (13%, 14%) on
two adjacent high-value cells — the rest of the park is patrolled with near-zero
probability, and poacher attack probability spreads out roughly in proportion to density
elsewhere. The pinned upstream revision's `utils/target_utils.py` refers to `np.floor`
without importing NumPy. The geographic adapter supplies that missing module name at
import time; the GUARD checkout itself remains untouched. GUARD also imports
`scikit-learn` and `h5py` at module load time, so both are declared in MAIRL's optional
`guard` dependency group.

## Reproduce

```bash
make guard
uv run python -m experiments.guard.behavioral.validate_behavioral --seed-count 32
uv run python -m experiments.guard.behavioral.tune_warmup --game nw_chinatown_sfg_isg_1
uv run python -m experiments.guard.geographic.map_lobeke
```

The scripts above run individual analyses. For resumable experiments, use
`experiments.guard.run` with the settings in `conf/application/guard.yaml`.
See [the experiment guide](../../docs/experiments.md).

## Roadmap

- **Tier 1 (done, this directory → `behavioral/`):** the 3-dim behavioral feature map,
  validated on Chinatown and Etosha.
- **Tier 2 (`scale/`, planned):** `lobeke_sfg_gsg_1`, 24,389 defender schedules — a
  stress test of `optimize_leader_policy_batch` and the active policy selector at an
  order of magnitude larger leader action space.
- **Tier 3 (done, `geographic/`):** regenerated `GreenSecurityGame` from upstream source
  data for a real park-map figure — see above. `scoring_method="density"` avoided needing
  a centroid-clustering-specific dependency, but `target_utils.py`/`green_security_game.py`
  import `scikit-learn`/`h5py` unconditionally regardless, so both ended up in the `guard`
  optional group anyway. Regenerating `InfraSecurityGame` similarly is not recommended —
  it additionally needs `geopandas` + `osmnx` + a Census shapefile not included in the
  upstream checkout or this repository, which trades reproducibility for a marginal
  visual gain over the Chinatown pkl
  already in use.
