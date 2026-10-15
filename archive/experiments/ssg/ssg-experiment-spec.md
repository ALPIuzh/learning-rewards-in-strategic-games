# Active Stackelberg IRL on Stackelberg security games — experiment specification

Instantiation of the paper's setup on the "guards and treasures" security game of
Nguyen et al. (AAAI 2013), the de facto benchmark for bounded-rational adversary
modelling in SSGs. This document fixes the game, the feature map, the ground truth,
the protocol, and the figures. It assumes the algorithm (design step, MLE, forward
Stackelberg solve) is already implemented.

---

## 1. What the application is

A defender allocates $K$ security resources across $T$ targets. She commits to a
*randomized* allocation, publishes it, and an attacker — a human subject, in the
original experiments — sees the resulting coverage probabilities together with the
payoff table and picks one target to attack.

The attacker is not an expected-value maximizer. Nguyen et al. model him with a
**subjective utility** that is linear in the quantities displayed on screen, passed
through a softmax. The weights of that linear combination are the object of interest:
they are not derivable from the payoff table, they differ across populations, and
they must be recovered from observed attacks. That is the IRL problem.

The defender's own payoffs are known. So the only unknown is the attacker's weight
vector, and once it is estimated the defender re-solves her own optimization. This is
exactly the paper's pipeline, minus the step where the defender chooses *which*
coverage vectors to play in order to learn faster.

**Why the reward is behavioral, not payoff-based.** The attacker's true expected value
at target $t$ is $U^a_t = x_t P^a_t + \left( 1 - x_t \right) R^a_t$. The subjective
utility is *not* this quantity. State this explicitly in the paper: $r^2$ here is the
behavioral reward the population acts on, not the game's payoff. This is the strongest
available motivation for IRL (nobody can write the subjective utility down), but it
means the follower cannot be described as entropy-regularized with respect to its own
payoff. Under classic QR the attacker's utility is known up to a single scalar
$\lambda$, and "reward learning" collapses to a one-dimensional problem — the
subjective utility is what makes $\omega$ a genuine $d$-dimensional unknown.

---

## 2. The game

Targets $t \in [T]$. Defender resources $K$. Per target, four numbers:

| symbol | meaning | sign | range |
|---|---|---|---|
| $R^a_t$ | attacker reward, uncovered target attacked | $> 0$ | $[0, 10]$ |
| $P^a_t$ | attacker penalty, covered target attacked | $< 0$ | $[-10, 0]$ |
| $R^d_t$ | defender reward, covered target attacked | $> 0$ | $[0, 10]$ |
| $P^d_t$ | defender penalty, uncovered target attacked | $< 0$ | $[-10, 0]$ |

Coverage vector $x \in \mathcal C := \left\{ x \in [0,1]^T : \sum_t x_t \leq K \right\}$.

Attacker's subjective utility and response:

$$
\hat U^a_t = w_1 x_t + w_2 R^a_t + w_3 P^a_t,
\qquad
q_t(x) = \frac{\exp \left( \lambda \hat U^a_t \right)}{\sum_{t'} \exp \left( \lambda \hat U^a_{t'} \right)},
\qquad \lambda := 1 .
$$

Defender's expected value: $U^d_t(x) = x_t R^d_t + \left( 1 - x_t \right) P^d_t$, and her
objective is $\sum_t q_t(x) \, U^d_t(x)$.

Two settings in the original experiments: $\left(T, K\right) = \left(8, 3\right)$ and
$\left(24, 9\right)$.

---

## 3. Mapping onto the paper's notation

| paper | SSG object |
|---|---|
| $\mathcal B$ | targets $[T]$ — the attacker picks one |
| $\mathcal A$ | pure allocations $\left\{ a \subseteq [T] : \lvert a \rvert = K \right\}$ |
| $\mu \in \Delta(\mathcal A)$ | randomized patrol |
| $\psi_\mu(t)$ | $\left( x_t, R^a_t, P^a_t \right)$, with $x_t = \Pr_{a \sim \mu}\left( t \in a \right)$ |
| $\omega$ | $\left( w_1, w_2, w_3 \right)$ |
| $\eta$ | $\lambda = 1$ |
| $\nu^*_\mu$ | the attack distribution $q$ |
| $r^1(a,t)$ | $R^d_t \mathbb{1}\{t \in a\} + P^d_t \mathbb{1}\{t \notin a\}$ |
| $J^{\mathrm{leader}}_r(\mu)$ | $\sum_t q_t(x) \, U^d_t(x)$ |
| $\hat\mu$ | the SU-BRQR solution under $\hat\omega$ |
| $R_{\max}$ | $\max_t \max\left( \lvert R^d_t \rvert, \lvert P^d_t \rvert \right) = 10$ |

**Work in $x$, not in $\mu$.** Without scheduling constraints every $x \in \mathcal C$ is
implementable by some $\mu \in \Delta(\mathcal A)$ (Kiekintveld et al., AAMAS 2009), and
the follower only ever sees $x$. So the design problem is an optimization over the
polytope $\mathcal C \subset \mathbb R^T$, not over $\Delta(\mathcal A)$, whose dimension is
$\binom{T}{K}$. This collapses the grouped design problem to something directly solvable
and is the single most important implementation shortcut here. Cite Kiekintveld et al. for
the equivalence.

---

## 4. The feature map

$$
\phi(a, t) = \left( \mathbb{1}\{ t \in a \}, \; R^a_t, \; P^a_t \right) \in \mathbb R^3
\quad \Longrightarrow \quad
\psi_\mu(t) = \left( x_t, \; R^a_t, \; P^a_t \right).
$$

Three properties worth stating explicitly in the paper:

1. **The features are known because they are displayed.** The interface shows the
   subject $x_t$, $R^a_t$, $P^a_t$, $R^d_t$, $P^d_t$ per target. Only displayed
   quantities can enter the subjective utility, so the interface *defines* the
   admissible feature map. This is a rare case where Assumption 1 is not an
   assumption.
2. **The leader controls one coordinate.** $R^a_t, P^a_t$ are fixed by the payoff
   structure; only $x_t$ moves. This is the partially-controlled design regime the
   warm-up analysis was written for, and it should be flagged as such.
3. **But it controls the sampling distribution.** Since
   $\mathcal I_\mu(\omega) = \eta^2 \, \mathrm{Cov}_{t \sim q(x)}\left[ \psi_\mu(t) \right]$,
   moving $x$ reweights which targets are observed, and hence how much information the
   *fixed* coordinates carry. Because $w_1 < 0$ and large in magnitude, raising coverage
   on the attractive targets flattens $q$ and maximizes that covariance. The design rule
   has a one-line reading: **cover the targets everyone attacks, so the responses spread
   out and you learn about the rest** — in direct conflict with the defender's own
   objective, which wants coverage where $R^d$ is high. That is the exploration/exploitation
   tension, and it is interpretable.

### Optional $d = 5$ variant

$\phi(a,t) = \left( \mathbb{1}\{t \in a\}, R^a_t, P^a_t, R^d_t, P^d_t \right)$. Useful for a
$d$-sweep, but note that Nguyen et al. report the 5-parameter model predicting *worse*
than the 3-parameter one on a held-out set of 100 payoff structures (44% vs 51% of
structures fitted at $\alpha = 0.05$). Present it as a harder problem, not a better model.

### Normalization and the $\eta S$ constant

Assumption 1 wants $\max \lVert \phi \rVert_2 \leq 1$. Raw features have
$\lVert \phi \rVert_2 \leq \sqrt{1 + 100 + 100} \approx 14.2$, so the footnote-2
rescaling gives $\tilde\eta = 14.2$ and $\tilde S \approx 9.9$, hence
$\tilde\eta \tilde S \approx 140$ and a theoretical
$n_{\mathrm{warm}} \sim e^{2\tilde\eta\tilde S}$ that is astronomically larger than the
~45 subjects per game actually run.

Do not let this scare you off, but do not hide it either. The quantity that actually
governs self-concordance is

$$
d_\mu\left( \omega, \hat\omega \right) = \max_{t, t'} \left\lvert \left\langle \psi_\mu(t) - \psi_\mu(t'), \, \omega - \hat\omega \right\rangle \right\rvert ,
$$

which depends on the **estimation error**, not on $\lVert \omega \rVert$. In raw units the
certificate $\max_\mu d_\mu \leq 1/\eta$ asks that the induced subjective-utility
differences be accurate to about 1 nat, i.e. roughly
$\lVert \omega - \hat\omega \rVert_2 \lesssim 10^{-1}$ given
$\lVert \Delta\psi \rVert \sim 14$. That is demanding but attainable. **Tune
$n_{\mathrm{warm}}$ empirically, report the tuned value, and say in one sentence that the
theoretical constant is conservative at these payoff scales.** A reviewer who discovers
this gap unaided will treat it as a defect; stated, it is a normal constant-factor gap.

---

## 5. Ground truth $\omega^\star$

Three published MLE fits, all with $\lambda = 1$ and the same feature map:

| population / setting | $w_1$ | $w_2$ | $w_3$ | $\lVert \omega \rVert_2$ |
|---|---|---|---|---|
| AMT workers, 8 targets | $-9.85$ | $0.37$ | $0.15$ | $9.86$ |
| Israeli intelligence experts, 8 targets | $-11.0$ | $0.54$ | $0.35$ | $11.0$ |
| AMT workers, 24 targets | $-15.29$ | $0.53$ | $0.34$ | $15.3$ |

(5-parameter fit, for the $d = 5$ variant: $\left(-8.23, 0.28, 0.12, 0.07, 0.09\right)$.)

Use the 8-target AMT vector as the default. The three vectors give a free sweep over
$\eta S$ at fixed feature map — run the same protocol under each and show the rate
degrading with $\lVert \omega \rVert$, which tests the $\eta S$ dependence on real
parameter values rather than on a sweep you invented.

---

## 6. The exogenous context

The payoff structures are generated from GAMUT covariance games with covariance
parameter $r \in [-1, 0]$ in increments of $0.1$, 100 structures per value; Nguyen et al.
select 22 for their experiments. $r = -1$ is near zero-sum ($R^d \approx -R^a$),
$r = 0$ is uncorrelated.

Set $\theta = \left( R^a, P^a, R^d, P^d \right)$, drawn from this generator. Then:

- $\omega$ is **invariant** across $\theta$ (same population).
- $\phi_\theta$ and $r^1_\theta$ **both** depend on $\theta$.
- There are no dynamics: each round is a fresh normal-form game, the follower is myopic.

This is the exogenous-context structure the setup requires, and it is not contrived —
it is how the benchmark was already built.

**Train/test split.** Train on contexts $\theta_1, \dots, \theta_K$ drawn from the
generator; evaluate $\mathrm{SubOpt}$ on held-out $\theta'_1, \dots, \theta'_J$. This
is the experiment that justifies IRL over direct online Stackelberg optimization: the
same $\hat\omega$ must serve payoff structures never seen during training. Without it,
a reviewer asks why you didn't just optimize the leader policy directly.

---

## 7. Protocol

Warm-up produces $\hat\omega_0$ from a fixed design. Then for $k = 1, \dots, K$:

```python
theta_k = sample_context()  # payoff structure
x_k = design_step(G[k - 1], omega_hat[k - 1], theta_k)  # argmax_x tr(G^{-1} I_x)
t_k = sample_response(x_k, omega_star, theta_k)  # t ~ q(x_k), one attack
L += loglik_increment(t_k, x_k, theta_k)
omega_hat[k] = mle(L, Omega)
G[k] = lam * I - hessian(L, omega_hat[k])
```

Evaluation, per held-out context $\theta'$:

```python
x_hat = su_brqr(omega_hat[K], theta_prime)  # forward Stackelberg solve
x_star = su_brqr(omega_star, theta_prime)
subopt = J_leader(x_star, omega_star, theta_prime) - J_leader(x_hat, omega_star, theta_prime)
```

Three primitives are needed from the library, all of which exist:

- `su_brqr(omega, theta)` — $\max_{x \in \mathcal C} \sum_t q_t(x; \omega) \, U^d_t(x)$.
  Smooth, non-concave in $x$; projected gradient or Frank–Wolfe over $\mathcal C$, multi-start.
- `design_step` — $\arg\max_{x \in \mathcal C} \mathrm{tr}\left( G^{-1} \mathcal I_x(\hat\omega) \right)$,
  with $\mathcal I_x(\hat\omega) = \eta^2 \mathrm{Cov}_{t \sim q(x)}\left[ \psi_x(t) \right]$.
  Same geometry, same solver.
- `mle` — concave in $\omega$ (negative semidefinite Hessian), so any convex solver;
  Nguyen et al. use `fmincon`.

**One observation per round** = one attacked target. In the original experiments this is
one human subject; in simulation, one draw from $q$. If you batch $n$ subjects per
coverage vector, say so — it changes the effective $K$.

---

## 8. Baselines

| baseline | what it isolates | source |
|---|---|---|
| MLE at SU-BRQR coverage | passive learning; the empirical form of the impossibility result | Nguyen et al. 2013 |
| MLE at MATCH / BRQR coverage | the coverage that actually generated the published data | Pita et al. 2012; Yang et al. 2011 |
| decision-focused MLE | loss reweighting vs. data selection | Perrault et al. AAAI 2020 |
| fixed-budget query selection | non-adaptive design | Haghtalab et al. IJCAI 2016 |
| offline G-optimal design + MLE | adaptivity, holding design fixed | — |
| adaptive D-optimal / E-optimal | the choice of scalarization | — |
| pure exploitation, $x_k = \hat x_{k-1}$ | what a deployed system does | — |

The framing to use: decision-focused learning fixes the *loss* at fixed data; this work
fixes the *data*. The two are complementary, not contradictory — say so explicitly, or a
reviewer who knows that literature will read the result as a conflict.

---

## 9. What to plot

1. **The game board.** A $T$-cell grid, cell shade = $x_t$, circle area = $q_t$, cell
   annotated with $\left( R^a_t, P^a_t \right)$. Two panels: SU-BRQR coverage (responses
   concentrated on two or three targets) and the actively designed coverage (responses
   spread). This is the figure that makes the mechanism legible.
2. **Convergence.** $\mathrm{SubOpt}_\omega(\hat\mu)$ on held-out contexts against $K$,
   log-log, against the $-1/2$ reference slope. The passive baseline plateaus; that
   plateau is the impossibility result.
3. **Conditioning.** $\sigma_{\min}\left( \Phi(\mu) \right)$ under SU-BRQR coverage vs.
   under the active design, as a function of the GAMUT covariance parameter $r$. See
   §10 — this is the honest version of the identifiability claim on this benchmark.
4. **$\eta S$ sweep.** Same protocol under the three published $\omega^\star$ vectors.
5. **Warm-up ablation.** Performance with and without warm-up, across $\eta$. Converts
   the ugliest constant in the paper into evidence that the phase does something.
6. **Context strip.** Four boards with different payoff structures, one $\omega$, four
   different $x^\star$. Communicates the exogenous-context story with no prose.

---

## 10. A correction to expect

With $T = 8 \geq d + 1 = 4$ and generic payoff structures, the difference vectors
$\left( x_t - x_1, R^a_t - R^a_1, P^a_t - P^a_1 \right)$ span $\mathbb R^3$, so
$\Phi(\mu)$ has full rank and $\omega$ is *exactly* identifiable from a single coverage
vector. **Lemma 3's exact degeneracy does not bite on generic instances of this
benchmark.** What bites is conditioning: near $r = -1$ the game is nearly zero-sum, a good
defender's coverage becomes approximately a function of $\left( R^a, P^a \right)$, and
$x_t - x_1$ falls close to the span of the payoff differences. $\Phi(\mu)$ becomes
ill-conditioned rather than singular.

Two consequences. First, present the empirical claim as ill-conditioning, and reserve
exact non-identifiability for a constructed instance. Second, the constructed instance is
easy: take payoff structures with $P^a_t = \alpha R^a_t + \beta$ across targets, and
interpolate between that and a generic structure. That gives a controlled degeneracy
parameter and a clean figure, and it is honest about being synthetic.

There is a real story in this: *defending optimally makes coverage collinear with the
payoffs, which is exactly what destroys identifiability*. That is a better motivation
for active selection than any counterexample.

---

## 11. Open items to verify

- Whether the raw human-subject attack counts are obtainable. If not, the fallback is a
  simulator calibrated to the published $\omega^\star$ and the GAMUT generator, with the
  calibration source cited — still stronger than an invented environment.
- The exact GAMUT covariance-game payoff generator settings, so the context distribution
  is reproducible.
- Whether Perrault et al. (AAAI 2020) release code and the processed benchmark; their
  8/3 and 24/9 splits with 44 games per setting are the version most recently reused.
