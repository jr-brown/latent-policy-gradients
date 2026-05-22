import wandb
import torch
import logging
import numpy as np
import tempfile
import os

from tqdm import tqdm
from copy import copy
from pprint import pformat
from typing import Type, Any
from datetime import datetime
from itertools import count
from dataclasses import dataclass, field

import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.util import Kwargs, wait_for_user


log = logging.getLogger(__name__)

DEFAULT_ASSUMED_AGENT_N_STEPS_PPO: int=2048  # 2048 is SB3 PPO default

agent_aliases: dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "SAC": SAC,
}


@dataclass
class RolloutBuffer:
    """Circular buffer to store teacher rollout data for distillation."""
    max_size: int
    use_memmap: bool = False
    memmap_dir: str | None = None
    warn_on_overwrite: bool = True
    observations: np.ndarray | None = None
    actions: np.ndarray | None = None
    logits: np.ndarray | None = None
    rewards: np.ndarray | None = None
    dones: np.ndarray | None = None
    _current_idx: int = 0
    _is_full: bool = False
    _memmap_files: list[str] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        """Return the number of samples currently in the buffer."""
        return self.max_size if self._is_full else self._current_idx
    
    def _initialize_arrays(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        logit: np.ndarray,
    ) -> None:
        """Initialize arrays with correct shapes based on first experience."""
        if self.use_memmap:
            # Create memmap-backed arrays
            # Observations
            obs_file = tempfile.NamedTemporaryFile(
                mode='w+b', delete=False, dir=self.memmap_dir, suffix='_obs.npy'
            )
            self._memmap_files.append(obs_file.name)
            obs_file.close()
            self.observations = np.memmap(
                obs_file.name, dtype=observation.dtype, mode='w+',
                shape=(self.max_size, *observation.shape)
            )
            
            # Actions
            action_file = tempfile.NamedTemporaryFile(
                mode='w+b', delete=False, dir=self.memmap_dir, suffix='_act.npy'
            )
            self._memmap_files.append(action_file.name)
            action_file.close()
            self.actions = np.memmap(
                action_file.name, dtype=action.dtype, mode='w+',
                shape=(self.max_size, *action.shape)
            )
            
            # Logits
            logit_file = tempfile.NamedTemporaryFile(
                mode='w+b', delete=False, dir=self.memmap_dir, suffix='_logits.npy'
            )
            self._memmap_files.append(logit_file.name)
            logit_file.close()
            self.logits = np.memmap(
                logit_file.name, dtype=logit.dtype, mode='w+',
                shape=(self.max_size, *logit.shape)
            )
            
            # Rewards (always float32)
            reward_file = tempfile.NamedTemporaryFile(
                mode='w+b', delete=False, dir=self.memmap_dir, suffix='_rewards.npy'
            )
            self._memmap_files.append(reward_file.name)
            reward_file.close()
            self.rewards = np.memmap(
                reward_file.name, dtype=np.float32, mode='w+',
                shape=(self.max_size,)
            )
            
            # Dones (always bool)
            done_file = tempfile.NamedTemporaryFile(
                mode='w+b', delete=False, dir=self.memmap_dir, suffix='_dones.npy'
            )
            self._memmap_files.append(done_file.name)
            done_file.close()
            self.dones = np.memmap(
                done_file.name, dtype=bool, mode='w+',
                shape=(self.max_size,)
            )
            
            log.info(f"Created memmap-backed RolloutBuffer with {len(self._memmap_files)} files in {self.memmap_dir or 'temp dir'}")
        else:
            # Create regular numpy arrays
            self.observations = np.empty((self.max_size, *observation.shape), dtype=observation.dtype)
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.logits = np.empty((self.max_size, *logit.shape), dtype=logit.dtype)
            self.rewards = np.empty((self.max_size,), dtype=np.float32)
            self.dones = np.empty((self.max_size,), dtype=bool)
    
    def __del__(self):
        """Cleanup memmap files on deletion."""
        if self.use_memmap:
            for filepath in self._memmap_files:
                try:
                    if os.path.exists(filepath):
                        os.unlink(filepath)
                except Exception as e:
                    log.warning(f"Failed to cleanup memmap file {filepath}: {e}")
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        logit: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Add a single experience to the buffer."""
        # Initialize arrays on first add
        if self.observations is None:
            self._initialize_arrays(observation, action, logit)
        
        assert self.observations is not None
        assert self.actions is not None
        assert self.logits is not None
        assert self.rewards is not None
        assert self.dones is not None
        
        # Warn if overwriting
        if self._is_full and self.warn_on_overwrite and self._current_idx == 0:
            log.warning(f"RolloutBuffer full (size={self.max_size}), now overwriting old experiences")
        # Add experience at current index
        self.observations[self._current_idx] = observation
        self.actions[self._current_idx] = action
        self.logits[self._current_idx] = logit
        self.rewards[self._current_idx] = reward
        self.dones[self._current_idx] = done
        
        # Update index and full flag
        self._current_idx = (self._current_idx + 1) % self.max_size
        if self._current_idx == 0:
            self._is_full = True
    
    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        logits: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of experiences to the buffer."""
        batch_size = len(observations)
        
        # Initialize arrays on first add
        if self.observations is None:
            self._initialize_arrays(observations[0], actions[0], logits[0])
        
        assert self.observations is not None
        assert self.actions is not None
        assert self.logits is not None
        assert self.rewards is not None
        assert self.dones is not None
        
        for i in range(batch_size):
            # Warn if overwriting
            if self._is_full and self.warn_on_overwrite and self._current_idx == 0:
                log.warning(f"RolloutBuffer full (size={self.max_size}), now overwriting old experiences")
            # Add experience at current index
            self.observations[self._current_idx] = observations[i]
            self.actions[self._current_idx] = actions[i] if actions.ndim > 1 else actions[i].reshape(1)
            self.logits[self._current_idx] = logits[i]
            self.rewards[self._current_idx] = rewards[i]
            self.dones[self._current_idx] = dones[i]
            
            # Update index and full flag
            self._current_idx = (self._current_idx + 1) % self.max_size
            if self._current_idx == 0:
                self._is_full = True
    
    def get_epoch_batches(self, batch_size: int, rng: np.random.Generator):
        """
        Generate batches that cover the entire buffer exactly once per epoch.
        
        Yields batches of indices that partition the buffer. The buffer is shuffled
        at the start of each epoch. If buffer size is not divisible by batch_size,
        the last batch will contain the remaining samples.
        
        Args:
            batch_size: Size of each batch
            rng: Random number generator for shuffling
            
        Yields:
            Dictionary containing batch of experiences with keys:
            observations, actions, logits, rewards, dones
        """
        assert self.observations is not None
        assert self.actions is not None
        assert self.logits is not None
        assert self.rewards is not None
        assert self.dones is not None
        
        # Create shuffled indices for the current buffer size
        indices = rng.permutation(self.size)
        
        # Yield batches
        for start_idx in range(0, self.size, batch_size):
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            yield dict(
                observations=self.observations[batch_indices],
                actions=self.actions[batch_indices],
                logits=self.logits[batch_indices],
                rewards=self.rewards[batch_indices],
                dones=self.dones[batch_indices],
            )
    
    @classmethod
    def merge(cls, buffers: list['RolloutBuffer']) -> 'RolloutBuffer':
        """
        Merge multiple rollout buffers into a single buffer.
        
        Args:
            buffers: List of RolloutBuffer instances to merge
            
        Returns:
            A new RolloutBuffer containing all data from input buffers
        """
        if not buffers:
            raise ValueError("Cannot merge empty list of buffers")
        
        if len(buffers) == 1:
            return buffers[0]
        
        # Calculate total size
        total_size = sum(buf.size for buf in buffers)
        
        # Create new buffer with combined size
        merged = cls(max_size=total_size, warn_on_overwrite=False)
        
        # Add all experiences from each buffer
        for buf in buffers:
            if buf.observations is not None:
                merged.add_batch(
                    observations=buf.observations[:buf.size],
                    actions=buf.actions[:buf.size],
                    logits=buf.logits[:buf.size],
                    rewards=buf.rewards[:buf.size],
                    dones=buf.dones[:buf.size],
                )
        
        return merged


def get_action_distribution_logits(agent: BaseAlgorithm, obs: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor | np.ndarray:
    """
    Extract logits from the agent's policy for given observations.
    
    Args:
        agent: The RL agent
        obs: Observations (batch of observations)
        device: PyTorch device
        requires_grad: Whether to keep gradients (for training)
        
    Returns:
        Logits as torch.Tensor (if requires_grad=True) or numpy array (if requires_grad=False)
    """
    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
    
    with torch.enable_grad() if requires_grad else torch.no_grad():
        # Get the policy distribution
        if hasattr(agent.policy, 'get_distribution'):
            distribution = agent.policy.get_distribution(obs_tensor)
            
            # For categorical (discrete) actions
            if hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'logits'):
                logits = distribution.distribution.logits
            # For continuous actions (Gaussian)
            elif hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'loc'):
                # For Gaussian, we'll use mean and log_std
                mean = distribution.distribution.loc
                log_std = torch.log(distribution.distribution.scale)
                logits = torch.cat([mean, log_std], dim=-1)
            else:
                raise ValueError(f"Unsupported distribution type: {type(distribution)}")
        else:
            raise ValueError("Policy does not have get_distribution method")
    
    return logits if requires_grad else logits.cpu().numpy()


def _format_action(act, dtype):
    act_arr = np.array(act, dtype=dtype)
    return dtype(act_arr.item()) if act_arr.size == 1 else act_arr


def rollout_policy(
    agent: BaseAlgorithm,
    env: gym.Env | None = None,
    env_kwargs: Kwargs | None = None,
    n_steps: int | None = None,
    n_episodes: int | None = None,
    max_steps_per_episode: int | None = None,
    visualise_episodes: int = 0,
    should_wait_for_user: bool = False,
    log_rewards: bool = False,
    collect_buffer: bool = False,
    buffer: RolloutBuffer | None = None,
    device: torch.device | None = None,
    rng_seed: int = 0,
    use_tqdm: bool = False,
) -> tuple[list[float], RolloutBuffer | None, list[list[dict]]]:
    """
    Generic function to rollout a policy in an environment.
    
    Args:
        agent: The RL agent to rollout
        env: Pre-existing environment (if provided, env_kwargs ignored)
        env_kwargs: Environment configuration kwargs
        n_steps: Total number of steps to collect (mutually exclusive with n_episodes)
        n_episodes: Number of episodes to collect (mutually exclusive with n_steps)
        max_steps_per_episode: Maximum steps per episode (None for unlimited)
        visualise_episodes: Number of episodes to visualize (requires human render mode)
        should_wait_for_user: Pause before starting rollouts
        log_rewards: Log episode rewards as they complete
        collect_buffer: Whether to collect rollout buffer data
        buffer: Existing RolloutBuffer to add experiences to (requires collect_buffer=True)
        device: PyTorch device (required if collect_buffer=True)
        rng_seed: Random seed for environment
        use_tqdm: Whether to show progress bar
        
    Returns:
        Tuple of (episode_rewards, rollout_buffer, episode_infos)
        - episode_rewards: List of episode rewards
        - rollout_buffer: RolloutBuffer if collect_buffer=True, else None
        - episode_infos: List of lists of info dicts, one list per episode
    """
    # Validation
    if (n_steps is None) == (n_episodes is None):
        if visualise_episodes > 0:
            n_episodes = visualise_episodes
        else:
            raise ValueError("Exactly one of n_steps or n_episodes must be provided")
    
    if collect_buffer and device is None:
        raise ValueError("device must be provided when collect_buffer=True")
    
    if buffer is not None and not collect_buffer:
        raise ValueError("buffer can only be provided when collect_buffer=True")

    if should_wait_for_user:
        wait_for_user("Waiting before starting rollouts...")

    # Create or use environment
    no_visualise_env = None
    
    if env is None:
        if env_kwargs is None:
            raise ValueError("Either env or env_kwargs must be provided")
        
        # Create non-visualizing environment
        no_visualise_env = gym.make(**env_kwargs)
        
        # If visualization requested, create separate visualizing environment
        if visualise_episodes > 0:
            visualise_env_kwargs = {**env_kwargs, "render_mode": "human"}
            env = gym.make(**visualise_env_kwargs)
        else:
            env = no_visualise_env

    elif visualise_episodes > 0:
        log.warning("visualise_episodes > 0 with pre-existing env: visualization may not work as expected")
    
    # Determine action dtype
    dtype = np.int32 if isinstance(env.action_space, gym.spaces.Discrete) else np.float32
    
    # Initialize tracking variables
    episode_rewards = []
    episode_infos = []
    current_episode_reward = 0.0
    current_episode_infos = []
    step_count = 0
    episode_count = 0
    
    # Initialize or use existing buffer
    if collect_buffer:
        if buffer is None:
            # Create new buffer with exact size needed
            buffer_size = n_steps if n_steps is not None else (n_episodes * (max_steps_per_episode or 1000))
            buffer = RolloutBuffer(max_size=buffer_size, warn_on_overwrite=True)
            log.info(f"Created new RolloutBuffer with max_size={buffer_size}")
        else:
            log.info(f"Using existing RolloutBuffer (current size: {buffer.size}/{buffer.max_size})")
    
    # Reset environment
    reset_result = env.reset(seed=rng_seed)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    # Main rollout loop
    for _ in tqdm(count(), desc="Getting rollouts...", disable=not use_tqdm):
        # Check termination conditions
        if ((n_steps is not None and step_count >= n_steps) or
            (n_episodes is not None and episode_count >= n_episodes)):
            break
        
        # Get action from agent
        if buffer is not None:
            # Get logits for buffer
            obs_batch = obs[np.newaxis, ...]
            logits, = get_action_distribution_logits(agent, obs_batch, device, requires_grad=False)
        
        action, _ = agent.predict(obs, deterministic=False)
        
        # Step environment
        result = env.step(_format_action(action, dtype))
        
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = result
        
        reward_scalar = float(reward)
        done_scalar = bool(done)
        
        # Store data in buffer if collecting
        if buffer is not None:
            buffer.add(
                observation=obs,
                action=action,
                logit=logits,
                reward=reward_scalar,
                done=done_scalar,
            )
        
        # Store info for this step
        current_episode_infos.append(info)
        
        # Update tracking
        current_episode_reward += reward_scalar
        step_count += 1
        obs = next_obs
        
        # Handle episode completion
        if done_scalar or (max_steps_per_episode is not None and 
                          step_count >= (max_steps_per_episode * (episode_count + 1))):
            episode_count += 1

            if log_rewards:
                log.info(f"Episode {episode_count} reward: {current_episode_reward:.2f}")

            # Switch from visualizing to non-visualizing environment if needed
            if (
                visualise_episodes > 0 and 
                episode_count >= visualise_episodes and 
                no_visualise_env is not None
            ):
                visualise_episodes = 0
                env.close()
                env = no_visualise_env

            episode_rewards.append(current_episode_reward)
            episode_infos.append(current_episode_infos)
            current_episode_reward = 0.0
            current_episode_infos = []
            
            # Reset environment for next episode
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    env.close()
    
    if buffer is not None:
        log.info(f"Buffer now contains {buffer.size} steps")
    
    return episode_rewards, buffer, episode_infos


def process_and_log_rollout_metrics(
    env_id: str,
    episode_rewards: list[float],
    episode_infos: list[list[dict]],
    name: str = "eval",
):
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    log.info(f"{name} mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")

    if wandb.run is not None:
        wandb.log({
            f"{name}/mean_episode_reward": mean_reward,
            f"{name}/std_episode_reward": std_reward,
        })

    if env_id[:5] == "Maze-":
        # Process episode infos to calculate object touch statistics
        touch_counts_per_episode = []
        
        for episode_info_list in episode_infos:
            # Count touches for this episode
            episode_touch_count = {}
            for step_info in episode_info_list:
                touched_objects = step_info.get("touched_objects", [])
                for obj_name in touched_objects:
                    episode_touch_count[obj_name] = episode_touch_count.get(obj_name, 0) + 1
            touch_counts_per_episode.append(episode_touch_count)
        
        # Calculate average touches per object across all episodes
        if touch_counts_per_episode:
            # Get all unique object names
            all_objects = set()
            for counts in touch_counts_per_episode:
                all_objects.update(counts.keys())
            
            # Calculate statistics for each object
            object_stats = {}
            for obj_name in all_objects:
                touches = [counts.get(obj_name, 0) for counts in touch_counts_per_episode]
                object_stats[obj_name] = {
                    "mean": float(np.mean(touches)),
                    "std": float(np.std(touches)),
                    "min": int(np.min(touches)),
                    "max": int(np.max(touches)),
                    "touched_in_episodes": sum(1 for t in touches if t > 0),
                }
            
            # Log statistics
            log.info("Object touch statistics across episodes:")
            for obj_name, stats in sorted(object_stats.items()):
                log.info(
                    f"  {obj_name}: "
                    f"mean={stats['mean']:.2f} ± {stats['std']:.2f}, "
                    f"min={stats['min']}, max={stats['max']}, "
                    f"touched in {stats['touched_in_episodes']}/{len(episode_infos)} episodes"
                )
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb_stats = {}
                for obj_name, stats in object_stats.items():
                    wandb_stats[f"{name}/touches/{obj_name}/mean"] = stats["mean"]
                    wandb_stats[f"{name}/touches/{obj_name}/std"] = stats["std"]
                    wandb_stats[f"{name}/touches/{obj_name}/episodes"] = stats["touched_in_episodes"]
                wandb.log(wandb_stats)


def create_agent(
    env: gym.Env | VecEnv,
    agent_cls: Type[BaseAlgorithm],
    name: str = "",
    verbose: int = 1,
    **kwargs,
) -> BaseAlgorithm:
    """
    Create a new agent.
    
    Args:
        env: The environment for the agent
        algorithm: Algorithm type (PPO, SAC, etc.)
        policy_kwargs: Policy network configuration
        name: Name for tensorboard logging
        **kwargs: Additional arguments for agent initialization
    
    Returns:
        Newly created agent
    """
    if name == "":
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    policy = kwargs.pop("policy", "MlpPolicy")

    if isinstance(kwargs.get("policy_kwargs", {}), str):
        kwargs["policy_kwargs"] = eval(
            kwargs["policy_kwargs"],
            globals={},
        )

    return agent_cls(
        policy,
        env,
        tensorboard_log=f".tensorboard_logs/{name}",
        verbose=verbose,
        **kwargs
    )


# To-do: Incoporate returning of something that allows calculation of log_interval
def get_agent(
    env: gym.Env | VecEnv | None = None,
    env_kwargs: Kwargs | None = None,
    venv_kwargs: Kwargs | None = None,
    algorithm: str = "PPO",
    checkpoint_path: str | None = None,
    create_agent_kwargs: Kwargs | None = None,
    rng_seed: int = 0,
) -> BaseAlgorithm:
    """
    Create a new agent or load from checkpoint.
    
    Args:
        env: Optional pre-existing environment
        env_kwargs: Environment configuration kwargs (for single env)
        venv_kwargs: Vectorized environment configuration kwargs
        algorithm: Algorithm type (PPO, SAC, etc.)
        policy_kwargs: Policy network configuration
        checkpoint_path: Path to saved model checkpoint (if loading)
        name: Name for tensorboard logging (ignored if loading)
        rng_seed: Random seed for environment creation
        **kwargs: Additional arguments for agent initialization (ignored if loading)
    
    Returns:
        Agent instance (newly created or loaded)
    """
    assert (env is None) != (env_kwargs is None and venv_kwargs is None), "Either env or env_kwargs/venv_kwargs must be provided, but not both"
    assert (checkpoint_path is None) != (create_agent_kwargs is None), "Either checkpoint_path or create_agent_kwargs must be provided, but not both"

    agent_cls = agent_aliases[algorithm]
    env = env or get_env(env_kwargs=env_kwargs, venv_kwargs=venv_kwargs, rng_seed=rng_seed)

    if checkpoint_path is not None:
        agent = agent_cls.load(checkpoint_path, env=env)
        log.info(f"Loaded agent from {checkpoint_path}")
        return agent

    elif create_agent_kwargs is not None:
        agent = create_agent(env, agent_cls, **create_agent_kwargs)
        log.info(f"Created new agent of type {algorithm}")
        return agent

    else:
        raise ValueError("Either checkpoint_path or create_agent_kwargs must be provided")


def replace_venv_str_with_cls(vec_env_kwargs):
    vec_env_cls_table = {
        "subprocvecenv": SubprocVecEnv,
        "dummyvecenv": DummyVecEnv,
    }
    _vec_env_kwargs = copy(vec_env_kwargs)

    if "vec_env_cls" in vec_env_kwargs.keys():
        _vec_env_kwargs["vec_env_cls"] = vec_env_cls_table[vec_env_kwargs["vec_env_cls"].lower()]

    return _vec_env_kwargs


def get_env(
    env_kwargs: Kwargs | None = None,
    venv_kwargs: Kwargs | None = None,
    rng_seed: int =0,
    ) -> gym.Env | VecEnv:

    assert (env_kwargs is not None) != (venv_kwargs is not None), "Exactly one of env_cfg or venv must be provided"

    if env_kwargs is not None:
        log.info(f"env_kwargs\n{pformat(env_kwargs)}")
        return gym.make(**env_kwargs)

    elif venv_kwargs is not None:
        venv_kwargs = replace_venv_str_with_cls(venv_kwargs)
        assert venv_kwargs is not None
        return make_vec_env(**venv_kwargs, seed=rng_seed)

    else:
        raise ValueError("Either env_kwargs or venv_kwargs must be provided")


"""
def transform_get_agent_and_agent_train_kwargs(
    get_agent_kwargs: Kwargs | None = None,
    agent_train_kwargs: Kwargs | None = None,
    venv_kwargs: Kwargs | None = None,
    log_steps: int | None = None,
) -> tuple[Kwargs, Kwargs]:

    get_agent_kwargs = get_agent_kwargs or {}

    if get_agent_kwargs.get("algorithm", "PPO") == "PPO":
        default_assumed_agent_n_steps = DEFAULT_ASSUMED_AGENT_N_STEPS_PPO
    else:
        default_assumed_agent_n_steps =  1

    if agent_train_kwargs is None:
        agent_train_kwargs = {"total_timesteps": 10_000}

    if log_steps is not None:
        assert agent_train_kwargs.get("log_interval", None) is None
        modified_agent_train_kwargs = copy(agent_train_kwargs)
        n_steps = agent_kwargs.get("n_steps", default_assumed_agent_n_steps)

        if venv_kwargs is not None:
            n_envs = venv_kwargs.get("n_envs", 1)
            log.info(f"Using {n_envs=} from venv_kwargs")
        else:
            n_envs = 1

        log_interval = max(log_steps // (n_steps * n_envs), 1)
        log.info(f"{log_interval=}")
        modified_agent_train_kwargs["log_interval"] = log_interval
        agent_train_kwargs = modified_agent_train_kwargs

    log.info(f"Final agent_kwargs:\n{pformat(agent_kwargs)}")
    log.info(f"Final agent_train_kwargs:\n{pformat(agent_train_kwargs)}")

    return agent_kwargs, agent_train_kwargs
"""

