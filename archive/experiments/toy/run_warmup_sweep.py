"""Run the fixed-total-budget toy warm-up sweep."""

from experiments._config import CONFIG_DIRECTORY
from mairl_exp.cli.run_experiment import main

if __name__ == "__main__":
    main(config_directory=CONFIG_DIRECTORY, default_config_name="toy_warmup_sweep_config")
