import logging

from pprint import pformat
from collections.abc import Callable
from stable_baselines3.common.base_class import BaseAlgorithm

from src.util import Kwargs
from src.sb3_util import rollout_policy, get_agent, process_and_log_rollout_metrics


log = logging.getLogger(__name__)


def eval_agent(
    rollout_kwargs: Kwargs,
    env_kwargs: Kwargs,
    agent: BaseAlgorithm | None = None,
    get_agent_kwargs: Kwargs | None = None,
    log_name: str = "eval",
    rng_seed: int = 0,
):
    """
    Load a saved agent checkpoint and visualize its policy.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        env_kwargs: Environment configuration for visualization
        algorithm: Algorithm type (PPO, SAC, etc.)
        steps_per_attempt: Max steps per episode (None for unlimited)
        attempts: Number of episodes to visualize
        log_total_rewards: Whether to log total rewards
    """
    assert (agent is None) != (get_agent_kwargs is None), "Either provide an agent or get_agent_kwargs to load one"

    if agent is None:
        assert get_agent_kwargs is not None
        agent = get_agent(env_kwargs=env_kwargs, rng_seed=rng_seed, **get_agent_kwargs)

    episode_rewards, _, episode_infos = rollout_policy(
        agent=agent,
        env_kwargs=env_kwargs,
        **rollout_kwargs,
    )
    process_and_log_rollout_metrics(
        env_id=env_kwargs["id"],
        episode_rewards=episode_rewards,
        episode_infos=episode_infos,
        name=log_name,
    )


log_names_fn_map: dict[str, Callable[[Kwargs], str]] = {
    "goals": lambda env_kwargs: f"eval_{'_and_'.join(env_kwargs['goals'])}",
}


def multi_eval_agent(
    env_kwargs_list: list[Kwargs],
    env_kwargs_base: Kwargs | None = None,
    log_names: list[str] | None = None,
    log_name_fn: Callable[[Kwargs], str] | str | None = None,
    **eval_agent_kwargs
):
    if env_kwargs_base is not None:
        env_kwargs_list = [
            {**env_kwargs_base, **env_kwargs} for env_kwargs in env_kwargs_list
        ]

    if log_names is None:
        if log_name_fn is not None:
            if isinstance(log_name_fn, str):
                log_name_fn = log_names_fn_map[log_name_fn]
            log_names = [log_name_fn(env_kwargs) for env_kwargs in env_kwargs_list]
        else:
            log_names = [f"eval_{i}" for i in range(len(env_kwargs_list))]
    else:
        assert log_name_fn is None, "Provide either log_names or log_name_fn, not both"

    assert len(env_kwargs_list) == len(log_names), "env_kwargs_list and log_names must have the same length"

    for env_kwargs, log_name in zip(env_kwargs_list, log_names):
        log.info(f"Evaluating agent in environment:\n{pformat(env_kwargs)}\nwith log name: {log_name}")
        eval_agent(
            env_kwargs=env_kwargs,
            log_name=log_name,
            **eval_agent_kwargs,
        )

