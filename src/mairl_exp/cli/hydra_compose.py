"""Hydra composition behind a Python-3.14-compatible command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from mairl.exceptions import ConfigurationError


def composed_config(*, config_directory: Path, description: str, default_config_name: str) -> DictConfig | None:
    """Compose Hydra groups from a caller-owned directory, or print the config."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides such as experiment.seed_count=50 or numerics=high_accuracy"
    )
    parser.add_argument("--config-name", default=default_config_name)
    parser.add_argument("--cfg", action="store_true", help="print the resolved config and exit")
    args = parser.parse_args()

    config_directory = config_directory.resolve()
    if not config_directory.is_dir():
        raise ConfigurationError(
            "Hydra configuration directory does not exist: "
            f"{config_directory}. Paper experiment recipes are run from a MAIRL "
            "repository checkout and are not bundled with the installable package."
        )

    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)
    if args.cfg:
        print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))
        return None
    return cfg
