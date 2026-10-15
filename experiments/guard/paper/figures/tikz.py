"""Load the Lobeke experiment outputs and invoke the central TikZ recipes.

Presentation logic lives in :mod:`reproducibility.figures.recipes.lobeke`; this
module is the experiment-specific adapter retained for command compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.guard.geographic.map_lobeke import home_base_position
from experiments.guard.paper.figures.corrected_traces import DEFAULT_CORRECTION_DIR, load_figure_run_traces
from experiments.guard.paper.figures.data import (
    infer_warmup_budget,
    load_runtime,
    run_source_suboptimality_is_unit_reward,
    source_suboptimality_in_unit_reward,
)
from mairl_exp.config import GENERATED_TIKZ_DIR, RESULTS_DIR
from reproducibility.figures.recipes.lobeke import (
    MAIN_TRANSFER_BASELINE,
    write_compact_transfer_figure,
    write_convergence_figure,
    write_downstream_transfer_figure,
    write_transfer_figure,
)

FIGURE2_DATA_DIR = RESULTS_DIR / "figure2_assets_behavioral_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, action="append")
    parser.add_argument("--correction-dir", type=Path, default=DEFAULT_CORRECTION_DIR)
    parser.add_argument("--figure2-data-dir", type=Path, default=FIGURE2_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_TIKZ_DIR)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--warmup-budget", type=int, default=None)
    args = parser.parse_args()

    run_dirs = tuple(args.run_dir)
    traces = load_figure_run_traces(run_dirs, args.correction_dir)
    runtime = load_runtime(run_dirs[0])
    unit_markers = {run_source_suboptimality_is_unit_reward(path) for path in run_dirs}
    if len(unit_markers) != 1:
        raise ValueError("composed Lobeke runs use different source-reward metric units")
    traces = source_suboptimality_in_unit_reward(
        traces,
        raw_reward_range=float(runtime.metadata["leader_reward_normalization"]["raw_range"]),
        already_unit_reward=unit_markers.pop(),
    )
    warmup_budget = infer_warmup_budget(traces) if args.warmup_budget is None else args.warmup_budget
    write_convergence_figure(
        traces,
        args.output_dir / "figure1_convergence_unit_reward.tex",
        confidence_level=args.confidence_level,
        warmup_budget=warmup_budget,
    )
    write_transfer_figure(
        args.figure2_data_dir,
        args.output_dir / "figure2_transfer_unit_reward.tex",
        base_position=home_base_position(),
        baseline_key=MAIN_TRANSFER_BASELINE,
    )
    write_compact_transfer_figure(
        args.figure2_data_dir,
        args.output_dir / "figure2_transfer_unit_reward_compact.tex",
        base_position=home_base_position(),
        baseline_key=MAIN_TRANSFER_BASELINE,
    )
    write_downstream_transfer_figure(
        args.figure2_data_dir,
        args.output_dir / "figure4_downstream_transfer_unit_reward.tex",
        base_position=home_base_position(),
    )
    print(f"wrote TikZ figures to {args.output_dir}")


if __name__ == "__main__":
    main()
