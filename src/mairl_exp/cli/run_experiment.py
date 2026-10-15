"""Hydra entry point for resumable MAIRL experiments."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from mairl.exceptions import ConfigurationError
from mairl_exp.application import ExperimentApplication
from mairl_exp.cli.hydra_compose import composed_config
from mairl_exp.config import ExperimentNumerics
from mairl_exp.execution.runner import run_experiment


def main(*, config_directory: Path, default_config_name: str = "runs/toy") -> None:
    cfg = composed_config(
        config_directory=config_directory,
        description=__doc__ or "Run a MAIRL experiment",
        default_config_name=default_config_name,
    )
    if cfg is None:
        return
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise ConfigurationError("root Hydra configuration must be a mapping")

    application = instantiate(cfg.application)
    numerics = instantiate(cfg.numerics)
    # Candidate lists may contain structured dataclass targets. Object conversion keeps
    # those typed instead of leaking OmegaConf containers into study execution.
    study = instantiate(cfg.experiment, _convert_="object")
    if not isinstance(application, ExperimentApplication):
        raise ConfigurationError("application config did not instantiate ExperimentApplication")
    if not isinstance(numerics, ExperimentNumerics):
        raise ConfigurationError("numerics config did not instantiate ExperimentNumerics")

    semantic_config = {
        "schema_version": resolved["schema_version"],
        "application": resolved["application"],
        "experiment": resolved["experiment"],
        "numerics": resolved["numerics"],
    }
    run_experiment(
        application=application,
        study=study,
        numerics=numerics,
        semantic_config=semantic_config,
        resolved_config_yaml=OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True),
        output_root=Path(str(cfg.run.root)),
        original_cwd=Path.cwd(),
        require_clean_git=bool(cfg.run.require_clean_git),
        resume=bool(cfg.run.resume),
        seed_batch_size=int(cfg.run.seed_batch_size),
    )
