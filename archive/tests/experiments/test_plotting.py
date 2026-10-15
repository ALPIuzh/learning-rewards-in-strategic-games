from __future__ import annotations

import json
from pathlib import Path

from experiments.toy.application import ToyApplication
from mairl_exp.plotting import FisherTransferPlot
from mairl_exp.runner import run_experiment
from mairl_exp.studies import BaselineComparisonStudy
from mairl_exp.warmup_sweep import WarmupSweepStudy
from mairl_exp.warmup_sweep_plot import WarmupSweepPlot


def test_plot_is_content_addressed_and_reuses_stored_results(
    experiment_numerics, baseline_study, repository_root, tmp_path: Path
) -> None:
    application = ToyApplication(problem_seed=5)
    numerics = experiment_numerics
    study = baseline_study
    run_dir = run_experiment(
        application=application,
        study=study,
        numerics=numerics,
        semantic_config={"application": 5, "study": study.name},
        resolved_config_yaml="test: true\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        command=("pytest",),
    )
    plot = FisherTransferPlot()
    config = {"plot": "fisher-transfer", "quantiles": [0.1, 0.9]}
    figure = plot.render(
        run_dir,
        repository=repository_root,
        semantic_plot_config=config,
        resolved_plot_config_yaml="plot: fisher-transfer\n",
        require_clean_git=False,
    )
    repeated = plot.render(
        run_dir,
        repository=repository_root,
        semantic_plot_config=config,
        resolved_plot_config_yaml="plot: fisher-transfer\n",
        require_clean_git=False,
    )
    changed = FisherTransferPlot(lower_quantile=0.2, upper_quantile=0.8).render(
        run_dir,
        repository=repository_root,
        semantic_plot_config={"plot": "fisher-transfer", "quantiles": [0.2, 0.8]},
        resolved_plot_config_yaml="plot: changed\n",
        require_clean_git=False,
    )

    assert figure == repeated
    assert figure.exists()
    assert changed.exists()
    assert changed.parent != figure.parent
    summary = json.loads((figure.parent / "summary.json").read_text())
    assert summary["without_warmup"]["fixed_random"]["seeds"] == 1
    assert summary["without_warmup"]["fixed_random"]["band"] == "mean_ci"
    assert (
        summary["without_warmup"]["fixed_random"]["lower"]
        == summary["without_warmup"]["fixed_random"]["mean"]
        == summary["without_warmup"]["fixed_random"]["upper"]
    )


def test_warmup_sweep_plot_combines_reference_and_new_runs(
    experiment_numerics, repository_root, tmp_path: Path
) -> None:
    application = ToyApplication(problem_seed=5)
    numerics = experiment_numerics
    reference_study = BaselineComparisonStudy(
        name="warmup-reference-test",
        seed_start=4,
        seed_count=1,
        total_samples=24,
        warmup_budget=0,
        evaluation_interval=12,
        methods=("fixed_random",),
        warmup_modes=(False,),
    )
    reference_run = run_experiment(
        application=application,
        study=reference_study,
        numerics=numerics,
        semantic_config={"study": "warmup-reference-test"},
        resolved_config_yaml="test: reference\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        command=("pytest",),
    )
    sweep_study = WarmupSweepStudy(
        name="warmup-sweep-test",
        seed_start=4,
        seed_count=1,
        total_samples=24,
        warmup_budgets=(24,),
        evaluation_interval=12,
        methods=("fixed_random",),
    )
    sweep_run = run_experiment(
        application=application,
        study=sweep_study,
        numerics=numerics,
        semantic_config={"study": "warmup-sweep-test"},
        resolved_config_yaml="test: sweep\n",
        output_root=tmp_path,
        original_cwd=repository_root,
        require_clean_git=False,
        resume=True,
        command=("pytest",),
    )
    figure = WarmupSweepPlot(
        reference_run_dirs=(str(reference_run),),
    ).render(
        sweep_run,
        repository=repository_root,
        semantic_plot_config={"plot": "warmup-sweep-test"},
        resolved_plot_config_yaml="plot: warmup-sweep-test\n",
        require_clean_git=False,
    )
    summary = json.loads((figure.parent / "summary.json").read_text())

    assert figure.exists()
    assert set(summary) == {"0", "24"}
    assert summary["0"]["fixed_random"]["seeds"] == 1
    assert summary["24"]["fixed_random"]["seeds"] == 1
