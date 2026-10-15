"""Application-agnostic active Stackelberg inverse reinforcement learning."""

from mairl.algorithm import ActiveIRL, ActiveIRLConfig, ActiveIRLResult, IterationDiagnostics
from mairl.core.oracle import FollowerOracle, SoftmaxFollowerOracle
from mairl.core.problem import CompilationConfig, CompiledProblem, compile_problem
from mairl.exceptions import CompilationError, ConfigurationError, ConvergenceError, MAIRLError
from mairl.exploration.design import WarmupConfig, WarmupDesign, compute_warmup_design
from mairl.exploration.policy import PolicySelectionResult, select_information_policy
from mairl.exploration.simplex import PolicyOptimizationConfig
from mairl.exploration.warmup import PreparedWarmup
from mairl.inference.best_response import (
    BestResponseValidationConfig,
    ValidatedFollowerBestResponse,
    validate_follower_best_response,
)
from mairl.inference.data import InteractionDataset
from mairl.inference.fixed_leader_ols import (
    EstimateOutsideBallWarning,
    FixedLeaderOLSConfig,
    FixedLeaderOLSResult,
    FrequencySmoothingWarning,
    fit_fixed_leader_ols,
    solve_fixed_leader_ols,
)
from mairl.inference.mle import MLEConfig, MLEResult, solve_mle
from mairl.planning.leader import (
    LeaderOptimizationConfig,
    LeaderOptimizationResult,
    optimize_leader_policy,
    optimize_leader_policy_batch,
)
from mairl.zero_sum import (
    CompiledZeroSumGame,
    QRESolveConfig,
    QREValidationConfig,
    ValidatedQRE,
    ZeroSumCompilationConfig,
    compile_zero_sum_game,
    fit_liao_zero_sum_ols,
    fit_liao_zero_sum_paired_actions,
    solve_liao_zero_sum_ols,
    solve_qre,
    validate_qre,
)

__all__ = [
    "ActiveIRL",
    "ActiveIRLConfig",
    "ActiveIRLResult",
    "BestResponseValidationConfig",
    "CompilationConfig",
    "CompilationError",
    "CompiledProblem",
    "CompiledZeroSumGame",
    "ConfigurationError",
    "ConvergenceError",
    "EstimateOutsideBallWarning",
    "FixedLeaderOLSConfig",
    "FixedLeaderOLSResult",
    "FollowerOracle",
    "FrequencySmoothingWarning",
    "InteractionDataset",
    "IterationDiagnostics",
    "LeaderOptimizationConfig",
    "LeaderOptimizationResult",
    "MAIRLError",
    "MLEConfig",
    "MLEResult",
    "PolicyOptimizationConfig",
    "PolicySelectionResult",
    "PreparedWarmup",
    "QRESolveConfig",
    "QREValidationConfig",
    "SoftmaxFollowerOracle",
    "ValidatedFollowerBestResponse",
    "ValidatedQRE",
    "WarmupConfig",
    "WarmupDesign",
    "ZeroSumCompilationConfig",
    "compile_problem",
    "compile_zero_sum_game",
    "compute_warmup_design",
    "fit_fixed_leader_ols",
    "fit_liao_zero_sum_ols",
    "fit_liao_zero_sum_paired_actions",
    "optimize_leader_policy",
    "optimize_leader_policy_batch",
    "select_information_policy",
    "solve_fixed_leader_ols",
    "solve_liao_zero_sum_ols",
    "solve_mle",
    "solve_qre",
    "validate_follower_best_response",
    "validate_qre",
]
