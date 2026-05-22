import jax.numpy as jnp
import logging

from src.preference_modelling.data_structures import (
    TrainingEnvironment,
    TrainingStage,
    TrainingPipeline,
    PaddedPipeline,
)


log = logging.getLogger(__name__)


def parse_object_string(
    obj_str: str,
    feature_to_idx: dict[str, int],
    n_features: int,
) -> jnp.ndarray | None:
    """
    Parse a single object string (e.g., "red_cross") into a feature vector.
    
    Args:
        obj_str: Object string like "red_cross" or "black_diamond"
        feature_to_idx: Mapping from feature names to indices
        n_features: Total number of features
    
    Returns:
        Binary feature vector, or None if parsing fails
    """
    # Strip _distractor suffix if present
    obj_str = obj_str.replace("_distractor", "")
    
    parts = obj_str.split("_")
    if len(parts) != 2:
        return None
    
    colour, shape = parts
    if colour not in feature_to_idx or shape not in feature_to_idx:
        return None
    
    feature_vec = jnp.zeros(n_features)
    feature_vec = feature_vec.at[feature_to_idx[colour]].set(1.0)
    feature_vec = feature_vec.at[feature_to_idx[shape]].set(1.0)
    
    return feature_vec


def parse_environment_string(
    env_str: str,
    feature_to_idx: dict[str, int],
    n_features: int,
) -> TrainingEnvironment | None:
    """
    Parse an environment string into a TrainingEnvironment.
    
    Handles _or_, _with_, and _distractor modifiers.
    
    Args:
        env_str: Environment string like "red_cross_or_blue_ring_with_black_plus_distractor"
        feature_to_idx: Mapping from feature names to indices
        n_features: Total number of features
    
    Returns:
        TrainingEnvironment, or None if parsing fails
    """
    goals = []
    distractors = []
    
    # Split by _or_ and _with_ to separate objects
    objects_in_env = env_str.split("_with_")
    objects_in_env = [obj for part in objects_in_env for obj in part.split("_or_")]

    # Separate goals and distractors
    goal_parts = [obj for obj in objects_in_env if "_distractor" not in obj]
    distractor_strs = [obj for obj in objects_in_env if "_distractor" in obj]
    
    # Parse goals
    for goal_str in goal_parts:
        goal_vec = parse_object_string(goal_str.strip(), feature_to_idx, n_features)
        if goal_vec is None:
            log.warning(f"Failed to parse goal: {goal_str}")
            return None
        goals.append(goal_vec)
    
    # Parse distractors
    for dist_str in distractor_strs:
        dist_vec = parse_object_string(dist_str.strip(), feature_to_idx, n_features)
        if dist_vec is None:
            log.warning(f"Failed to parse distractor: {dist_str}")
            return None
        distractors.append(dist_vec)
    
    return TrainingEnvironment(goals=goals, distractors=distractors)


def parse_stage_string(
    stage_str: str,
    feature_to_idx: dict[str, int],
    n_features: int,
) -> TrainingStage | None:
    """
    Parse a stage string into a TrainingStage.
    
    Handles _and_ to create multiple environments trained simultaneously but separately.
    
    Args:
        stage_str: Stage string like "red_cross_and_blue_ring"
        feature_to_idx: Mapping from feature names to indices
        n_features: Total number of features
    
    Returns:
        TrainingStage, or None if parsing fails
    """
    environments = []
    
    # Split by _and_ to get separate environments
    env_parts = stage_str.split("_and_")
    
    for env_str in env_parts:
        env = parse_environment_string(env_str.strip(), feature_to_idx, n_features)
        if env is None:
            return None
        environments.append(env)
    
    return TrainingStage(environments=environments)


def parse_pipeline_string(
    run_name: str,
    feature_to_idx: dict[str, int],
    n_features: int,
) -> TrainingPipeline | None:
    """
    Parse a run name into a complete TrainingPipeline.
    
    Operator precedence (highest to lowest): _then_, _and_, _or_/_with_, _distractor
    
    Args:
        run_name: Full run name like "red_cross_and_blue_ring_then_black_diamond"
        feature_to_idx: Mapping from feature names to indices
        n_features: Total number of features
    
    Returns:
        TrainingPipeline, or None if parsing fails
    
    Raises:
        ValueError: If the pipeline string is invalid
    """
    # Clean up run name
    cleaned_name = run_name.replace("maze_eval_", "").replace("_distill", "")
    
    stages = []
    
    # Split by _then_ to get sequential stages
    stage_parts = cleaned_name.split("_then_")
    
    for stage_str in stage_parts:
        stage = parse_stage_string(stage_str.strip(), feature_to_idx, n_features)
        if stage is None:
            log.warning(f"Failed to parse stage: {stage_str} in run {run_name}")
            return None
        stages.append(stage)
    
    if not stages:
        log.warning(f"No valid stages parsed from run: {run_name}")
        return None
    
    return TrainingPipeline(stages=stages)


def pipeline_to_debug_string(
    pipeline: TrainingPipeline,
    all_features: list[str],
) -> str:
    """
    Convert a pipeline to a human-readable debug string showing feature names.
    
    Args:
        pipeline: The parsed TrainingPipeline
        all_features: Ordered list of feature names
    
    Returns:
        Human-readable string representation
    """
    def vec_to_features(vec: jnp.ndarray) -> list[str]:
        return [all_features[i] for i in range(len(all_features)) if vec[i] > 0]
    
    lines = []
    for stage_idx, stage in enumerate(pipeline.stages):
        stage_parts = []
        for env in stage.environments:
            goal_strs = ["+".join(vec_to_features(g)) for g in env.goals]
            goals_repr = " | ".join(goal_strs) if len(goal_strs) > 1 else goal_strs[0] if goal_strs else "none"
            
            if env.distractors:
                dist_strs = ["+".join(vec_to_features(d)) for d in env.distractors]
                dists_repr = ", ".join(dist_strs)
                stage_parts.append(f"[{goals_repr} (distractors: {dists_repr})]")
            else:
                stage_parts.append(f"[{goals_repr}]")
        
        lines.append(f"Stage {stage_idx + 1}: {' & '.join(stage_parts)}")
    
    return " → ".join(lines)


def pad_pipeline(
    pipeline: TrainingPipeline,
    max_stages: int,
    max_envs_per_stage: int,
    max_goals_per_env: int,
    max_distractors_per_env: int,
    n_features: int,
) -> PaddedPipeline:
    """
    Convert a TrainingPipeline to padded arrays for batched JAX processing.
    
    Args:
        pipeline: The TrainingPipeline to pad
        max_stages: Maximum number of stages
        max_envs_per_stage: Maximum environments per stage
        max_goals_per_env: Maximum goals per environment
        max_distractors_per_env: Maximum distractors per environment
        n_features: Number of features
    
    Returns:
        PaddedPipeline with all arrays padded and masked
    """
    # Initialize arrays
    goals = jnp.zeros((max_stages, max_envs_per_stage, max_goals_per_env, n_features))
    distractors = jnp.zeros((max_stages, max_envs_per_stage, max_distractors_per_env, n_features))
    stage_mask = jnp.zeros(max_stages)
    env_mask = jnp.zeros((max_stages, max_envs_per_stage))
    goal_mask = jnp.zeros((max_stages, max_envs_per_stage, max_goals_per_env))
    distractor_mask = jnp.zeros((max_stages, max_envs_per_stage, max_distractors_per_env))
    
    for stage_idx, stage in enumerate(pipeline.stages):
        if stage_idx >= max_stages:
            log.warning(f"Pipeline has more than {max_stages} stages, truncating")
            break
        
        stage_mask = stage_mask.at[stage_idx].set(1.0)
        
        for env_idx, env in enumerate(stage.environments):
            if env_idx >= max_envs_per_stage:
                log.warning(f"Stage has more than {max_envs_per_stage} environments, truncating")
                break
            
            env_mask = env_mask.at[stage_idx, env_idx].set(1.0)
            
            for goal_idx, goal in enumerate(env.goals):
                if goal_idx >= max_goals_per_env:
                    log.warning(f"Environment has more than {max_goals_per_env} goals, truncating")
                    break
                goals = goals.at[stage_idx, env_idx, goal_idx].set(goal)
                goal_mask = goal_mask.at[stage_idx, env_idx, goal_idx].set(1.0)
            
            for dist_idx, dist in enumerate(env.distractors):
                if dist_idx >= max_distractors_per_env:
                    log.warning(f"Environment has more than {max_distractors_per_env} distractors, truncating")
                    break
                distractors = distractors.at[stage_idx, env_idx, dist_idx].set(dist)
                distractor_mask = distractor_mask.at[stage_idx, env_idx, dist_idx].set(1.0)
    
    return PaddedPipeline(
        goals=goals,
        distractors=distractors,
        stage_mask=stage_mask,
        env_mask=env_mask,
        goal_mask=goal_mask,
        distractor_mask=distractor_mask,
    )

