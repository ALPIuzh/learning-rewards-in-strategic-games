from __future__ import annotations

import json
from pathlib import Path

from experiments.toy.application import ToyApplication
from mairl_exp.evaluation.plotting import FisherTransferPlot
from mairl_exp.execution.runner import run_experiment


def test_plot_is_content_addressed_and_reuses_stored_results(experiment_numerics,
        baseline_study, repository_root, tmp_path: Path) -> None:  # fmt: skip
    application = ToyApplication(problem_seed=5)
    study = baseline_study
    run_dir = run_experiment(
        application=application,
        study=study,
        numerics=experiment_numerics,
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
