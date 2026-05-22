# Standard library
import sys

from typing import Callable, Optional

# Load project-local .env (if present) into os.environ BEFORE any other
# imports, so vars like WANDB_PATH / XLA_FLAGS / WANDB_API_KEY are visible
# to research_scaffold, jax, wandb etc. at their import time.
from dotenv import load_dotenv
load_dotenv()

# Local
from research_scaffold.argparsing import get_base_argparser, process_base_args

# For testing multi device semantics
# TODO: Make this a proper parsed argument
# environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Pre-import configuration
config_path: Optional[str] = None
meta_config_path: Optional[str] = None
sweep_config_path: Optional[str] = None

# Import structure note:
# Args and kwargs need to be parsed and interpretted before importing any other library,
# since some things might need to happen before any other imports to properly take effect
# E.g. setting log levels, cuda things


if __name__ == "__main__":
    parser = get_base_argparser()
    args = parser.parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)


# Third-party
from tqdm.contrib.logging import logging_redirect_tqdm

# Local
from research_scaffold.config_tools import execute_experiments
from research_scaffold.util import get_logger
from research_scaffold.wandb_parameter_subset_search import parameter_subset_search

from src.launchers import(
    play_cartpole, play_maze,
    train_rl_agent, distill_rl,
    eval_agent, multi_eval_agent,
    analyse_agent_preferences, fit_developmental_model, sweep_latent_dimension,
    plot_pipeline_training_curves,
    validate_preference_models,
)

# Private launchers — gracefully absent on a public release.
try:
    from src.private.launchers.bilinear_rank import sweep_bilinear_rank
except ImportError:
    sweep_bilinear_rank = None
try:
    from src.private.launchers.kl_integration import plot_kl_loss_over_integration_steps
except ImportError:
    plot_kl_loss_over_integration_steps = None


log = get_logger(__name__)


function_map: dict[str, Callable] = {
    f.__name__: f
    for f in [
        parameter_subset_search,
        play_cartpole,
        play_maze,
        train_rl_agent,
        distill_rl,
        eval_agent,
        multi_eval_agent,
        analyse_agent_preferences,
        fit_developmental_model,
        sweep_latent_dimension,
        plot_pipeline_training_curves,
        validate_preference_models,
    ]
    + ([sweep_bilinear_rank] if sweep_bilinear_rank is not None else [])
    + ([plot_kl_loss_over_integration_steps] if plot_kl_loss_over_integration_steps is not None else [])
}


if __name__ == "__main__":
    with logging_redirect_tqdm():
        log.info("##### Program Start #####")
        execute_experiments(
            function_map=function_map,
            config_path=config_path,
            meta_config_path=meta_config_path,
            sweep_config_path=sweep_config_path,
        )
        log.info("##### Program End #####")

    sys.exit()  # Helps close potentially hanging threads that may get produced by some libraries

