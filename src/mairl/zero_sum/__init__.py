"""Entropy-regularized two-player zero-sum inference."""

from mairl.zero_sum.equilibrium import QRESolveConfig, QREValidationConfig, ValidatedQRE, solve_qre, validate_qre
from mairl.zero_sum.liao import fit_liao_zero_sum_ols, fit_liao_zero_sum_paired_actions, solve_liao_zero_sum_ols
from mairl.zero_sum.problem import CompiledZeroSumGame, ZeroSumCompilationConfig, compile_zero_sum_game

__all__ = [
    "CompiledZeroSumGame",
    "QRESolveConfig",
    "QREValidationConfig",
    "ValidatedQRE",
    "ZeroSumCompilationConfig",
    "compile_zero_sum_game",
    "fit_liao_zero_sum_ols",
    "fit_liao_zero_sum_paired_actions",
    "solve_liao_zero_sum_ols",
    "solve_qre",
    "validate_qre",
]
