"""Run an experiment selected by its tracked configuration."""

from mairl_exp.cli.run_experiment import main
from mairl_exp.config import CONFIG_DIRECTORY

if __name__ == "__main__":
    main(config_directory=CONFIG_DIRECTORY)
