"""Generate a versioned plot from stored experiment results."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from mairl.exceptions import ConfigurationError
from mairl_exp.cli.hydra_compose import composed_config
from mairl_exp.execution.artifacts import find_repository


def main(*, config_directory: Path) -> None:
    cfg = composed_config(
        config_directory=config_directory, description=__doc__ or "Plot a MAIRL experiment", default_config_name="plot"
    )
    if cfg is None:
        return
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise ConfigurationError("root plot configuration must be a mapping")
    plot = instantiate(cfg.plot)
    if not callable(getattr(plot, "render", None)):
        raise ConfigurationError("plot config did not instantiate a supported plot")
    repository = find_repository(Path.cwd())
    figure = plot.render(
        Path(str(cfg.run_dir)).resolve(),
        repository=repository,
        semantic_plot_config={"schema_version": resolved["schema_version"], "plot": resolved["plot"]},
        resolved_plot_config_yaml=OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True),
        require_clean_git=bool(cfg.require_clean_git),
    )
    print(f"plot={figure}")
