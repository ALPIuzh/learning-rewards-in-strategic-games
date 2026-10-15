"""Render a stored experiment using the repository's tracked plot recipes."""

from mairl_exp.cli.plot_experiment import main
from mairl_exp.config import CONFIG_DIRECTORY

if __name__ == "__main__":
    main(config_directory=CONFIG_DIRECTORY)
