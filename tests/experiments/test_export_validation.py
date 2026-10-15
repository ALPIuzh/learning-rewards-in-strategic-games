from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.guard.paper.figures.corrected_traces import _validate_native_paired_run
from experiments.guard.paper.figures.data import infer_warmup_budget
from experiments.guard.paper.figures.transfer_assets import _suboptimality_summary, _write_tex_values


def test_lobeke_transfer_value_export_supports_five_candidates(tmp_path: Path) -> None:
    result = {"mean": 1e-3, "ci_half_width": 2e-4}
    summary = {"methods": {"active": {"tasks": [result] * 5}}}
    output = tmp_path / "values.tex"

    _write_tex_values(summary, output)

    rendered = output.read_text()
    assert "\\SuboptActiveOne" in rendered
    assert "\\SuboptActiveFive" in rendered

    summary["methods"]["active"]["tasks"].append(result)
    with pytest.raises(ValueError, match="at most 5 downstream tasks"):
        _write_tex_values(summary, output)


def test_lobeke_warmup_budget_is_inferred_from_all_warm_traces() -> None:
    traces = {
        ("active", True): [
            SimpleNamespace(checkpoints=np.asarray([200, 240, 280])),
            SimpleNamespace(checkpoints=np.asarray([200, 240, 280])),
        ]
    }
    assert infer_warmup_budget(traces) == 200

    traces[("active", True)][1].checkpoints[0] = 100
    with pytest.raises(ValueError, match="inconsistent warm-up budgets"):
        infer_warmup_budget(traces)


def test_transfer_summary_rejects_uninitialized_regrets() -> None:
    with pytest.raises(ValueError, match="finite"):
        _suboptimality_summary(np.asarray([0.1, np.nan]))


def test_native_lobeke_plot_input_requires_paired_ols_samples(tmp_path: Path) -> None:
    (tmp_path / "semantic_config.json").write_text(
        json.dumps({"experiment": {"ols": {"follower_reference_action": 22, "leader_reference_action": 22}}})
    )
    job_dir = tmp_path / "jobs" / "paired"
    job_dir.mkdir(parents=True)
    summary_path = job_dir / "summary.json"
    summary = {
        "sampling_protocol": "paired_empirical_leader_and_follower_actions",
        "observed_action_counts": {"leader": 3, "follower": 3},
    }
    summary_path.write_text(json.dumps(summary))
    np.savez(job_dir / "result.npz", leader_actions=np.asarray([0, 1, 2]), follower_actions=np.asarray([2, 1, 0]))
    manifest = {"expected_jobs": [{"job_id": "paired", "spec": {"method": "zero_sum_ols"}}]}

    _validate_native_paired_run(tmp_path, manifest)

    summary["sampling_protocol"] = "follower_actions_only"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="not paired-action"):
        _validate_native_paired_run(tmp_path, manifest)
