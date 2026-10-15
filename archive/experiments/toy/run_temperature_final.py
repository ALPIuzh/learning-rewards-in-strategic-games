"""Run the protected structured-game inverse-temperature sensitivity study."""

from experiments._config import CONFIG_DIRECTORY
from mairl_exp.cli.run_experiment import main

if __name__ == "__main__":
    main(
        config_directory=CONFIG_DIRECTORY,
        default_config_name="toy_temperature_final_config",
    )
