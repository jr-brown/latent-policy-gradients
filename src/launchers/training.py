import wandb
import torch
import logging
import gymnasium as gym

from pathlib import Path
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.util import Kwargs
from src.sb3_util import(
    RolloutBuffer, rollout_policy,
    get_env, get_agent,
    process_and_log_rollout_metrics,
)
from src.distillation import distill_student

from src.launchers.evaluation import eval_agent, multi_eval_agent


log = logging.getLogger(__name__)


def train_rl_agent(
    get_agent_kwargs: Kwargs,
    agent_train_kwargs: Kwargs,
    env_kwargs: Kwargs | None = None,
    venv_kwargs: Kwargs | None = None,
    eval_agent_kwargs: Kwargs | None = None,
    multi_eval_agent_kwargs: Kwargs | None = None,
    rng_seed: int=0,
    checkpoint_freq: int | None = None,
    checkpoint_dir: str="./checkpoints",
    checkpoint_kwargs: Kwargs | None = None,
):
    assert (eval_agent_kwargs is None) != (multi_eval_agent_kwargs is None), "Provide either eval_agent_kwargs or multi_eval_agent_kwargs, not both"

    callbacks = []
    
    if wandb.run is not None:
        callbacks.append(WandbCallback(verbose=2))
        if not wandb.patched["tensorboard"]:
            wandb.tensorboard.patch()  # As not syncing tensorboard by default in research_scaffold
    
    if checkpoint_freq is not None:
        checkpoint_kwargs = checkpoint_kwargs or {}
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Adjust save_freq for vectorized envs
        if venv_kwargs is not None:
            n_envs = venv_kwargs.get("n_envs", 1)
            checkpoint_freq = max(1, checkpoint_freq // n_envs)

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_path),
            **checkpoint_kwargs,
        )
        callbacks.append(checkpoint_callback)
        log.info(f"Checkpoints will be saved to {checkpoint_path} every {checkpoint_freq} steps")
    
    callback_list = CallbackList(callbacks) if callbacks else None

    agent = get_agent(env_kwargs=env_kwargs, venv_kwargs=venv_kwargs, rng_seed=rng_seed, **get_agent_kwargs)

    try:
        log.info(f"Agent  pi:\n{agent.policy.actor}")
        log.info(f"Agent  vf:\n{agent.policy.vf}")
    except AttributeError:
        log.info(f"Agent policy (pi or vf not available):\n{agent.policy}")

    agent.learn(callback=callback_list, **agent_train_kwargs)

    # If we just have venv_kwargs, convert to env_kwargs for evaluation
    if env_kwargs is None and venv_kwargs is not None:
        env_kwargs = {"id": venv_kwargs["env_id"], **venv_kwargs.get("env_kwargs", {})}

    assert env_kwargs is not None

    if eval_agent_kwargs is not None:
        log.info("Evaluating trained agent...")
        eval_agent(
            agent=agent,
            env_kwargs=env_kwargs,
            **eval_agent_kwargs,
        )
    elif multi_eval_agent_kwargs is not None:
        log.info("Evaluating trained agent in multiple environments...")
        multi_eval_agent(
            agent=agent,
            env_kwargs_base=env_kwargs,
            **multi_eval_agent_kwargs,
        )


# TODO: De-duplicate all the env stuff where possible
def distill_rl(
    teacher_get_agent_kwargs: Kwargs | list[Kwargs],
    student_get_agent_kwargs: Kwargs,
    distill_student_kwargs: Kwargs,
    total_teacher_rollout_steps: int,
    env_kwargs: Kwargs | None = None,
    env_kwargs_for_multi_distill: list[Kwargs] | None = None,
    total_teacher_validation_rollout_steps: int = 0,
    eval_student_kwargs: Kwargs | None = None,
    rng_seed: int = 0,
):
    """
    Perform student-teacher distillation of an RL model with support for multiple teachers.
    
    Args:
        teacher_get_agent_kwargs: Configuration for loading teacher agent(s). Can be a single
            dict for one teacher or a list of dicts for multiple teachers.
        student_get_agent_kwargs: Configuration for creating student agent
        distill_student_kwargs: Kwargs for distillation (num_epochs, batch_size, etc.)
        eval_student_policy_kwargs: Configuration for evaluating student performance
        total_teacher_rollout_steps: Total number of steps to collect from all teachers for training.
            Will be divided equally among teachers.
        total_teacher_validation_rollout_steps: Total number of steps to collect from all teachers
            for validation (0 to disable). Will be divided equally among teachers.
        env_kwargs: Environment configuration
        rng_seed: Random seed
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Normalize teacher_get_agent_kwargs to list
    if isinstance(teacher_get_agent_kwargs, dict):
        teacher_get_agent_kwargs_list = [teacher_get_agent_kwargs]
    else:
        teacher_get_agent_kwargs_list = teacher_get_agent_kwargs
    assert len(teacher_get_agent_kwargs_list) > 0, "At least one teacher must be provided"

    env_kwargs = env_kwargs or {}
    env_kwargs_for_multi_distill = env_kwargs_for_multi_distill or [
        {}
        for _ in teacher_get_agent_kwargs_list
    ]
    all_env_kwargs = [
        {**env_kwargs, **specific_env_kwargs}
        for specific_env_kwargs in env_kwargs_for_multi_distill
    ]
    # del's to avoid accidental use later
    del env_kwargs
    del env_kwargs_for_multi_distill
    assert len(all_env_kwargs) == len(teacher_get_agent_kwargs_list), \
        "Length of env_kwargs_for_multi_distill must match number of teachers"
    
    n_teachers = len(teacher_get_agent_kwargs_list)
    log.info(f"Distilling from {n_teachers} teacher(s)")
    
    # Calculate steps per teacher
    rollout_steps_per_teacher = total_teacher_rollout_steps // n_teachers
    validation_steps_per_teacher = total_teacher_validation_rollout_steps // n_teachers if total_teacher_validation_rollout_steps > 0 else 0
    
    log.info(f"Each teacher will generate {rollout_steps_per_teacher} training steps")
    if validation_steps_per_teacher > 0:
        log.info(f"Each teacher will generate {validation_steps_per_teacher} validation steps")

    first_env_kwargs = all_env_kwargs[0]
    
    # Create environment for rollouts
    first_env = get_env(env_kwargs=first_env_kwargs, rng_seed=rng_seed)
    
    is_discrete = isinstance(first_env.action_space, gym.spaces.Discrete)
    log.info(f"Action space type: {'discrete' if is_discrete else 'continuous'}")
    
    # Create buffers for all teachers
    training_buffer = RolloutBuffer(
        max_size=total_teacher_rollout_steps,
        use_memmap=True,
        memmap_dir="./local/cache/rollout_buffer_memmaps",
    )
    validation_buffer = RolloutBuffer(
        max_size=total_teacher_validation_rollout_steps,
        use_memmap=True,
        memmap_dir="./local/cache/rollout_buffer_memmaps",
    ) if total_teacher_validation_rollout_steps > 0 else None
    
    log.info(f"Created training buffer with max_size={total_teacher_rollout_steps} (memmap, float16)")
    if validation_buffer is not None:
        log.info(f"Created validation buffer with max_size={total_teacher_validation_rollout_steps} (memmap, float16)")

    # Collect rollouts from all teachers
    for teacher_idx, (teacher_kwargs, teacher_env_kwargs) in enumerate(
        zip(teacher_get_agent_kwargs_list, all_env_kwargs),
        1
    ):
        log.info(f"Loading teacher {teacher_idx}/{n_teachers}")
        teacher_env = get_env(env_kwargs=teacher_env_kwargs, rng_seed=rng_seed)
        assert type(teacher_env.action_space) == type(first_env.action_space), \
            f"Teacher {teacher_idx} action space type does not match first teacher"
        assert type(teacher_env.observation_space) == type(first_env.observation_space), \
            f"Teacher {teacher_idx} observation space type does not match first teacher"
        teacher = get_agent(env_kwargs=teacher_env_kwargs, rng_seed=rng_seed, **teacher_kwargs)
        
        log.info(f"Generating {rollout_steps_per_teacher} training rollout steps from teacher {teacher_idx}/{n_teachers}...")
        teacher_episode_rewards, training_buffer, episode_infos = rollout_policy(
            agent=teacher,
            env=teacher_env,
            n_steps=rollout_steps_per_teacher,
            collect_buffer=True,
            buffer=training_buffer,
            device=device,
            use_tqdm=True,
        )
        process_and_log_rollout_metrics(
            env_id=teacher_env_kwargs["id"],
            episode_rewards=teacher_episode_rewards,
            episode_infos=episode_infos,
            name=f"teacher_{teacher_idx}_rollout",
        )
        
        # Generate validation rollouts if requested
        if validation_buffer is not None:
            log.info(f"Generating {validation_steps_per_teacher} validation rollout steps from teacher {teacher_idx}/{n_teachers}...")
            _, validation_buffer, _ = rollout_policy(
                agent=teacher,
                env=teacher_env,
                n_steps=validation_steps_per_teacher,
                collect_buffer=True,
                buffer=validation_buffer,
                device=device,
                use_tqdm=True,
            )

    assert training_buffer is not None

    log.info(f"Final training buffer size: {training_buffer.size}")
    if validation_buffer is not None:
        log.info(f"Final validation buffer size: {validation_buffer.size}")
    
    log.info("Initializing student agent (using first environment if multiple specified)")
    student = get_agent(first_env, **student_get_agent_kwargs)
    
    try:
        log.info(f"Student policy:\n{student.policy}")
    except AttributeError:
        pass
    
    # Distill student from merged teacher data
    student = distill_student(
        student=student,
        rollout_buffer=training_buffer,
        validation_buffer=validation_buffer,
        is_discrete=is_discrete,
        device=device,
        rng_seed=rng_seed,
        **distill_student_kwargs,
    )
    first_env.close()

    if eval_student_kwargs is not None:
        log.info("Evaluating trained agent")

        if "env_kwargs" not in eval_student_kwargs:
            log.info("Adding first env_kwargs to eval_student_kwargs as none were provided")
            eval_student_kwargs["env_kwargs"] = first_env_kwargs

        eval_agent(
            agent=student,
            log_name="student/eval",
            **eval_student_kwargs,
        )

