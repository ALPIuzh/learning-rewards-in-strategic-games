"""Adapter connecting GUARD security game benchmarks to the MAIRL experiment framework."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mairl.core.oracle import SoftmaxFollowerOracle
from mairl.core.problem import CompiledProblem, FloatArray, compile_problem
from mairl.exceptions import ConfigurationError
from mairl.planning.leader import LeaderOptimizationConfig, leader_value_and_gradient, optimize_leader_policy_batch
from mairl.zero_sum import CompiledZeroSumGame, compile_zero_sum_game
from mairl_exp.application import ApplicationRuntime
from mairl_exp.config import GUARD_ROOT, ExperimentNumerics
from mairl_exp.evaluation.metrics import FisherTransferMetricConfig, StackelbergFisherError


def load_guard_game_data(game_path_or_name: str | Path,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray | None]:  # fmt: skip
    """Load payoffs and optional target utilities from a GUARD pickle file."""

    path = Path(game_path_or_name)
    if not path.exists():
        # Check relative to predefined_games directory in GUARD sub-repository
        repo_predefined = GUARD_ROOT / "predefined_games" / f"{game_path_or_name}.pkl"
        if repo_predefined.exists():
            path = repo_predefined
        else:
            repo_predefined_raw = GUARD_ROOT / "predefined_games" / str(game_path_or_name)
            if repo_predefined_raw.exists():
                path = repo_predefined_raw
            else:
                raise ConfigurationError(f"Could not find GUARD game file at {game_path_or_name!r}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        attacker_mat = np.asarray(data["attacker_utility_matrix"], dtype=np.float64)
        defender_mat = np.asarray(data["defender_utility_matrix"], dtype=np.float64)
        target_utils = np.asarray(data["target_utilities"], dtype=np.float64) if "target_utilities" in data else None
        schedules = data.get("schedules")
        return attacker_mat, defender_mat, target_utils, schedules

    if isinstance(data, tuple) and len(data) == 2:
        attacker_mat = np.asarray(data[0], dtype=np.float64)
        defender_mat = np.asarray(data[1], dtype=np.float64)
        return attacker_mat, defender_mat, None, None

    if isinstance(data, np.ndarray):
        # Zero-sum game matrix where B = -A or A = data
        attacker_mat = np.asarray(data, dtype=np.float64)
        defender_mat = -attacker_mat
        return attacker_mat, defender_mat, None, None

    raise ConfigurationError(f"Unrecognized data format in pickle file {path}")


def coverage_indicator(attacker_mat: FloatArray, target_utils: FloatArray) -> FloatArray:
    """Fraction-covered indicator per (leader, follower) action pair, in [0, 1].

    Shared by the behavioral feature map below and by every caller that needs
    GUARD's coverage geometry directly rather than through the compiled feature
    map: downstream-task generation and the environment map visualizations.
    """

    att_covered = target_utils[2]
    att_uncovered = target_utils[3]
    diff = att_covered - att_uncovered
    diff_safe = np.where(np.abs(diff) < 1e-12, 1e-12, diff)
    return np.clip((attacker_mat - att_uncovered[None, :]) / diff_safe[None, :], 0.0, 1.0)


def _behavioral_features(attacker_mat: FloatArray, target_utils: FloatArray,
        *, fit_tolerance: float) -> tuple[FloatArray, FloatArray, float]:  # fmt: skip
    """Build the 3-dim behavioral (SUQR-style) feature map from target utilities.

    GUARD derives ``attacker_covered = attacker_uncovered / attacker_penalty_factor``
    with a single scalar ``attacker_penalty_factor`` shared across all targets
    (verified across every bundled predefined game). Under that structure the
    attacker's reward is exactly linear in
    ``phi(a, t) = (coverage_indicator, target_value, coverage_indicator * target_value)``,
    a genuine low-dimensional feature map rather than a per-target one-hot. The true
    coefficient is recovered by least squares (rather than hard-coded) so the mode
    degrades gracefully — with a reported fit error — if pointed at a future GUARD
    game whose penalty factor is not uniform across targets.
    """

    att_uncovered = target_utils[3]
    n_leader, n_follower = attacker_mat.shape
    cov_matrix = coverage_indicator(attacker_mat, target_utils)

    value = att_uncovered
    phi = np.empty((n_leader, n_follower, 3), dtype=np.float64)
    phi[:, :, 0] = cov_matrix
    phi[:, :, 1] = value[None, :]
    phi[:, :, 2] = cov_matrix * value[None, :]

    design = phi.reshape(-1, 3)
    target = attacker_mat.reshape(-1)
    true_omega, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fit_error = float(np.max(np.abs(design @ true_omega - target)))
    if fit_error > fit_tolerance:
        raise ConfigurationError(
            "behavioral feature mode assumes attacker_covered/attacker_uncovered follows "
            "a single game-wide penalty factor, but the least-squares fit of "
            f"phi(a,t)={{coverage, value, coverage*value}} to the attacker matrix has "
            f"max residual {fit_error:.3e} (tolerance {fit_tolerance:.3e}). This game's "
            "target utilities are not exactly representable by the 3-dim behavioral "
            "model; use feature_mode='target_coverage' instead."
        )

    return phi, true_omega.astype(np.float64), fit_error


def _scale_behavioral_features(phi_raw: FloatArray, true_omega: FloatArray, *, eta: float,
        radius_multiplier: float) -> tuple[FloatArray, float, float]:  # fmt: skip
    """Normalize raw behavioral features and derive eta/radius scaling.

    Shared by every behavioral-mode GUARD game builder (pickle-loaded games via
    :func:`_build_guard_features`, and live-generated games such as Lobeke's
    ``GreenSecurityGame`` reconstruction in ``geographic/map_lobeke.py``), so
    this normalization exists in exactly one place.
    """

    feature_norms = np.linalg.norm(phi_raw.reshape(-1, phi_raw.shape[-1]), axis=1)
    norm_bound = max(1.0, float(feature_norms.max()))
    phi = phi_raw / norm_bound
    problem_eta = eta * norm_bound
    radius = max(2.0, float(np.linalg.norm(true_omega)) * radius_multiplier)
    return phi, problem_eta, radius


def _build_guard_features(game_path_or_name: str | Path,
        *, feature_mode: Literal["target_coverage", "tabular", "behavioral"], eta: float, radius_multiplier: float,
        behavioral_fit_tolerance: float) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:  # fmt: skip
    """Build the uncompiled feature map, true omega, and problem scaling.

    Shared by :func:`build_guard_problem` (which compiles a plain follower
    problem) and :func:`build_guard_zero_sum_game` (which compiles a zero-sum
    game with leader payoff exactly ``-phi @ omega``), so the feature
    construction and eta/radius scaling logic exists in exactly one place.
    """

    attacker_mat, defender_mat, target_utils, _ = load_guard_game_data(game_path_or_name)
    n_leader, n_follower = attacker_mat.shape
    problem_eta = eta

    if feature_mode == "behavioral":
        if target_utils is None:
            raise ConfigurationError(
                f"feature_mode='behavioral' requires target utilities, which {game_path_or_name!r} "
                "does not provide (only schedule-form ('_sfg_') GUARD games carry them)"
            )
        phi_raw, true_omega, _fit_error = _behavioral_features(
            attacker_mat, target_utils, fit_tolerance=behavioral_fit_tolerance
        )
        phi, problem_eta, radius = _scale_behavioral_features(
            phi_raw, true_omega, eta=eta, radius_multiplier=radius_multiplier
        )
        return phi, true_omega, defender_mat, problem_eta, radius
    elif feature_mode == "target_coverage" and target_utils is not None:
        att_covered = target_utils[2]
        att_uncovered = target_utils[3]
        diff = att_covered - att_uncovered
        diff = np.where(np.abs(diff) < 1e-12, 1e-12, diff)
        cov_matrix = np.clip((attacker_mat - att_uncovered[None, :]) / diff[None, :], 0.0, 1.0)

        d = 2 * n_follower
        phi = np.zeros((n_leader, n_follower, d), dtype=np.float64)
        for a in range(n_leader):
            for b in range(n_follower):
                e_b = np.zeros(n_follower, dtype=np.float64)
                e_b[b] = 1.0
                c = cov_matrix[a, b]
                phi[a, b, :n_follower] = c * e_b
                phi[a, b, n_follower:] = (1.0 - c) * e_b

        true_omega = np.concatenate([att_covered, att_uncovered]).astype(np.float64)
    else:
        # Tabular feature mode: d = n_leader * n_follower
        d = n_leader * n_follower
        phi = np.zeros((n_leader, n_follower, d), dtype=np.float64)
        true_omega = attacker_mat.ravel().copy()
        idx = 0
        for a in range(n_leader):
            for b in range(n_follower):
                phi[a, b, idx] = 1.0
                idx += 1

    norm_omega = float(np.linalg.norm(true_omega))
    radius = max(2.0, norm_omega * radius_multiplier)
    return phi, true_omega, defender_mat, problem_eta, radius


def build_guard_problem(game_path_or_name: str | Path,
        *, feature_mode: Literal["target_coverage", "tabular", "behavioral"] = "target_coverage",
        eta: float = 1.25, radius_multiplier: float = 1.5,
        behavioral_fit_tolerance: float = 1e-8) -> tuple[CompiledProblem, FloatArray, FloatArray]:  # fmt: skip
    """Build a compiled MAIRL problem from a GUARD security game instance."""

    phi, true_omega, defender_mat, problem_eta, radius = _build_guard_features(
        game_path_or_name,
        feature_mode=feature_mode,
        eta=eta,
        radius_multiplier=radius_multiplier,
        behavioral_fit_tolerance=behavioral_fit_tolerance,
    )
    problem = compile_problem(phi, eta=problem_eta, radius=radius)
    return problem, true_omega, defender_mat


def build_guard_zero_sum_game(game_path_or_name: str | Path,
        *, feature_mode: Literal["behavioral"] = "behavioral", eta: float = 1.25, radius_multiplier: float = 1.5,
        behavioral_fit_tolerance: float = 1e-8) -> tuple[CompiledZeroSumGame, FloatArray, FloatArray]:  # fmt: skip
    """Build a zero-sum-compiled GUARD game with leader payoff exactly ``-phi @ omega``.

    GUARD's bundled games already have ``defender_mat == -attacker_mat`` exactly
    (verified for every predefined game); this additionally requires the
    feature map itself to reconstruct ``attacker_mat`` (up to
    ``behavioral_fit_tolerance``), which is what makes the leader's estimation-time
    payoff ``-phi @ omega`` consistent with the *compiled* zero-sum game object
    that Liao's OLS estimator and the joint QRE solve require.
    """

    phi, true_omega, defender_mat, problem_eta, radius = _build_guard_features(
        game_path_or_name,
        feature_mode=feature_mode,
        eta=eta,
        radius_multiplier=radius_multiplier,
        behavioral_fit_tolerance=behavioral_fit_tolerance,
    )
    game = compile_zero_sum_game(-phi, phi, eta=problem_eta, radius=radius)
    return game, true_omega, defender_mat


@dataclass(frozen=True, slots=True)
class GuardApplication:
    """Application adapter connecting GUARD security games to MAIRL experiment runner."""

    game_name: str = "nw_chinatown_sfg_isg_1"
    feature_mode: Literal["target_coverage", "tabular", "behavioral"] = "target_coverage"
    eta: float = 1.25
    radius_multiplier: float = 1.5

    def build(self, numerics: object) -> ApplicationRuntime:
        """Compile problem and construct oracle factory and estimate evaluator."""

        if not isinstance(numerics, ExperimentNumerics):
            raise TypeError("GuardApplication requires ExperimentNumerics")

        problem, true_omega, leader_reward = build_guard_problem(
            self.game_name, feature_mode=self.feature_mode, eta=self.eta, radius_multiplier=self.radius_multiplier
        )

        fisher_metric = StackelbergFisherError(
            problem,
            true_omega,
            leader_reward,
            FisherTransferMetricConfig(leader=LeaderOptimizationConfig(numerics.evaluation_policy)),
        )

        optimal_leader_sol = optimize_leader_policy_batch(
            problem, true_omega[None, :], leader_reward, LeaderOptimizationConfig(numerics.evaluation_policy), (0,)
        )[0]

        def oracle_factory(_: int) -> SoftmaxFollowerOracle:
            return SoftmaxFollowerOracle(problem, true_omega)

        def evaluate_estimates(estimates: FloatArray, seed: int) -> dict[str, FloatArray]:
            n_checkpoints = estimates.shape[0]
            coeff_l2_errors = np.linalg.norm(estimates - true_omega[None, :], axis=1)

            est_solutions = optimize_leader_policy_batch(
                problem,
                estimates,
                leader_reward,
                LeaderOptimizationConfig(numerics.evaluation_policy),
                tuple(seed for _ in range(n_checkpoints)),
            )
            stackelberg_subopt = np.zeros(n_checkpoints, dtype=np.float64)
            for i, est_sol in enumerate(est_solutions):
                val_est_under_true, _ = leader_value_and_gradient(problem, true_omega, leader_reward, est_sol.policy)
                stackelberg_subopt[i] = max(0.0, float(optimal_leader_sol.value - val_est_under_true))

            return {
                "stackelberg_suboptimality": stackelberg_subopt,
                "fisher_transfer_error": fisher_metric.evaluate_many(estimates, seed=seed),
                "coefficient_l2_error": coeff_l2_errors,
            }

        return ApplicationRuntime(
            name=f"guard_{self.game_name}",
            problem=problem,
            oracle_factory=oracle_factory,
            evaluate_estimates=evaluate_estimates,
            metadata={
                "game_name": self.game_name,
                "feature_mode": self.feature_mode,
                "eta": self.eta,
                "n_leader_actions": problem.n_leader_actions,
                "n_follower_actions": problem.n_follower_actions,
                "feature_dimension": problem.feature_dimension,
                "identifiable_dimension": problem.identifiable_dimension,
            },
        )
