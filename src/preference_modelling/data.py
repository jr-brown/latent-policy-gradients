import jax.numpy as jnp
import json
import hashlib
import logging
import pickle

from pathlib import Path
from tqdm import tqdm
from typing import Literal

from src.preference_modelling.data_structures import TrainingPipeline, PaddedPipeline
from src.preference_modelling.parsing import (
    parse_pipeline_string,
    pad_pipeline,
    parse_object_string,
    pipeline_to_debug_string,
)


log = logging.getLogger(__name__)

# Module constant for numerical stability
EPSILON = 1e-8


def compute_pipeline_max_sizes(
    pipelines: list[TrainingPipeline],
) -> tuple[int, int, int, int]:
    """
    Compute maximum sizes across all pipelines for padding.
    
    Args:
        pipelines: List of TrainingPipeline objects
    
    Returns:
        Tuple of (max_stages, max_envs_per_stage, max_goals_per_env, max_distractors_per_env)
    """
    max_stages = 0
    max_envs = 0
    max_goals = 0
    max_distractors = 0
    
    for pipeline in pipelines:
        max_stages = max(max_stages, pipeline.n_stages)
        for stage in pipeline.stages:
            max_envs = max(max_envs, stage.n_environments)
            for env in stage.environments:
                max_goals = max(max_goals, env.n_goals)
                max_distractors = max(max_distractors, env.n_distractors)
    
    return max_stages, max_envs, max_goals, max_distractors


def extract_features_from_goals(
    possible_goals: list[str],
    include_no_goal_feature: bool = False,
) -> tuple[list[str], list[str], list[str], dict[str, int], int]:
    """
    Extract all unique features from possible goals.

    Args:
        possible_goals: List of all possible goal strings
        include_no_goal_feature: If True (default), adds a dedicated 'no_goal' feature
            as the last dimension, allowing the model to learn a weight for "doing nothing"
            that evolves through the training pipeline. If False, no_goal values are
            computed via forward pass with all-zeros input (original behavior).

    Returns:
        Tuple of:
            - all_colours: Sorted list of colour names
            - all_shapes: Sorted list of shape names
            - all_features: Ordered list of all feature names (colours + shapes [+ no_goal])
            - feature_to_idx: Mapping from feature names to indices
            - n_features: Total number of features
    """
    all_colours = set()
    all_shapes = set()

    for goal in possible_goals:
        colour, shape = goal.split('_')
        all_colours.add(colour)
        all_shapes.add(shape)

    all_colours = sorted(all_colours)
    all_shapes = sorted(all_shapes)

    if include_no_goal_feature:
        # Add 'no_goal' as the final feature dimension
        all_features = all_colours + all_shapes + ['no_goal']
    else:
        # Original behavior: no dedicated no_goal feature
        all_features = all_colours + all_shapes

    n_features = len(all_features)

    feature_to_idx = {feature: i for i, feature in enumerate(all_features)}

    return all_colours, all_shapes, all_features, feature_to_idx, n_features


def prepare_training_data(
    run_env_metrics: dict[str, dict[str, tuple[float, float] | None]],
    possible_goals: list[str],
    include_no_goal_feature: bool = False,
) -> tuple[dict[str, tuple[TrainingPipeline, PaddedPipeline]], dict[str, int], list[str], int]:
    """
    Prepare training data by parsing pipelines and extracting features.

    Args:
        run_env_metrics: Mapping from run name to environment metrics
        possible_goals: List of all possible goal strings
        include_no_goal_feature: If True (default), includes a dedicated 'no_goal' feature
            dimension. If False, no_goal values are computed via forward pass with
            all-zeros input (original behavior).

    Returns:
        Tuple of:
            - agent_training_pipelines: Dict mapping agent name to (TrainingPipeline, PaddedPipeline)
            - feature_to_idx: Mapping from feature names to indices
            - all_features: Ordered list of feature names
            - n_features: Total number of features
    """
    all_colours, all_shapes, all_features, feature_to_idx, n_features = extract_features_from_goals(
        possible_goals, include_no_goal_feature=include_no_goal_feature
    )

    log.info(f"Features - Colours: {all_colours}, Shapes: {all_shapes}")

    # First pass: parse all pipelines to TrainingPipeline objects
    parsed_pipelines: dict[str, TrainingPipeline] = {}
    parse_failures = []

    for run_name in run_env_metrics.keys():
        pipeline = parse_pipeline_string(run_name, feature_to_idx, n_features)

        if pipeline is None or pipeline.n_stages == 0:
            parse_failures.append(run_name)
            continue

        parsed_pipelines[run_name] = pipeline

    # Compute max sizes from all parsed pipelines
    max_stages, max_envs_per_stage, max_goals_per_env, max_distractors_per_env = compute_pipeline_max_sizes(
        list(parsed_pipelines.values())
    )

    log.info(f"Total features: {n_features}, Max stages: {max_stages}, "
             f"Max envs/stage: {max_envs_per_stage}, Max goals/env: {max_goals_per_env}, "
             f"Max distractors/env: {max_distractors_per_env}")

    # Second pass: create padded versions
    agent_training_pipelines = {}

    for run_name, pipeline in parsed_pipelines.items():
        padded = pad_pipeline(
            pipeline=pipeline,
            max_stages=max_stages,
            max_envs_per_stage=max_envs_per_stage,
            max_goals_per_env=max_goals_per_env,
            max_distractors_per_env=max_distractors_per_env,
            n_features=n_features,
        )

        agent_training_pipelines[run_name] = (pipeline, padded)

        # Debug log the parsed pipeline
        debug_str = pipeline_to_debug_string(pipeline, all_features)
        log.debug(f"Parsed pipeline for {run_name}: {debug_str}")

    log.info(f"Parsed training pipelines for {len(agent_training_pipelines)} agents")
    if parse_failures:
        log.warning(f"Failed to parse {len(parse_failures)} pipelines: {parse_failures[:5]}...")

    return agent_training_pipelines, feature_to_idx, all_features, n_features


def parse_goal_to_binary(goal_str: str, feature_to_idx: dict[str, int], n_features: int) -> jnp.ndarray:
    """
    Parse a goal string into a binary feature vector.

    Real goals have 0 in the no_goal position (last dimension).
    The no_goal position is only set to 1 via get_no_goal_features().

    Args:
        goal_str: Goal string like "black_cross"
        feature_to_idx: Mapping from feature names to indices
        n_features: Total number of features (including no_goal dimension)

    Returns:
        Binary array of shape (n_features,) where goal[feature_idx] = 1 if feature is in goal,
        and goal[-1] = 0 (no_goal feature is 0 for real goals)
    """
    result = parse_object_string(goal_str, feature_to_idx, n_features)
    if result is None:
        return jnp.zeros(n_features)
    # Result already has n_features dimensions with last dimension = 0 for real goals
    return result


def _get_padding_dims(padded_pipeline: PaddedPipeline) -> dict:
    """Extract padding dimensions from a padded pipeline."""
    return {
        "max_stages": int(padded_pipeline.goals.shape[0]),
        "max_envs_per_stage": int(padded_pipeline.goals.shape[1]),
        "max_goals_per_env": int(padded_pipeline.goals.shape[2]),
        "max_distractors_per_env": int(padded_pipeline.distractors.shape[2]),
    }


def _compute_raw_cache_hash(raw_cache_runs: dict) -> str:
    """Compute hash of raw cache for processed cache invalidation."""
    content = json.dumps(sorted(raw_cache_runs.keys()))
    return hashlib.md5(content.encode()).hexdigest()


def prepare_examples(
    run_metrics_dict: dict[str, dict[str, tuple[float, float] | None]],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    feature_to_idx: dict[str, int],
    n_features: int,
    cache_dir: str = "local/cache",
    include_no_goal_feature: bool = False,
) -> list[dict]:
    """
    Convert run metrics to training examples with two-tier caching.

    Uses a raw cache (per-run data, incrementally updated) and a processed cache
    (fully padded examples, invalidated when raw cache changes).

    Separate cache files are maintained for include_no_goal_feature=True/False
    to avoid frequent cache invalidation when switching between modes.

    Args:
        run_metrics_dict: Dict mapping run_name to env_metrics
        agent_training_pipelines: Dict mapping agent name to (TrainingPipeline, PaddedPipeline)
        feature_to_idx: Mapping from feature names to indices
        n_features: Total number of features
        cache_dir: Directory for cache files
        include_no_goal_feature: If True (default), uses cache files with "_with_no_goal" suffix.
            If False, uses cache files with "_without_no_goal" suffix.

    Returns:
        List of example dicts with keys: pipeline, padded_pipeline, goal_0, goal_1,
        observed_prob_0, observed_prob_1, weight, run_name, env_name
    """
    if not agent_training_pipelines:
        return []

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Use different cache files based on include_no_goal_feature to avoid invalidation
    cache_suffix = "_with_no_goal" if include_no_goal_feature else "_without_no_goal"
    raw_cache_file = cache_path / f"examples_raw_cache{cache_suffix}.json"
    processed_cache_file = cache_path / f"examples_processed_cache{cache_suffix}.pkl"

    # Get current padding dimensions
    first_padded = next(iter(agent_training_pipelines.values()))[1]
    current_padding_dims = _get_padding_dims(first_padded)

    # === STEP 1: Load and update raw cache ===
    raw_cache_runs: dict[str, dict] = {}
    raw_cache_modified = False

    if raw_cache_file.exists():
        with open(raw_cache_file, 'r') as f:
            raw_cache_data = json.load(f)
        if raw_cache_data.get("n_features") == n_features:
            raw_cache_runs = raw_cache_data.get("runs", {})
            log.info(f"Loaded raw cache with {len(raw_cache_runs)} runs")
        else:
            log.info("Raw cache invalidated: n_features changed")
            raw_cache_modified = True

    # Check for new runs
    runs_to_compute = []
    for run_name in run_metrics_dict.keys():
        if run_name in agent_training_pipelines and run_name not in raw_cache_runs:
            runs_to_compute.append(run_name)

    # Compute raw data for new runs only
    if runs_to_compute:
        log.info(f"Computing raw data for {len(runs_to_compute)} new runs")
        for run_name in tqdm(runs_to_compute, desc="Computing new runs", leave=False):
            pipeline, _ = agent_training_pipelines[run_name]
            env_metrics = run_metrics_dict[run_name]

            run_examples = []
            for env_name, rates in env_metrics.items():
                if rates is None:
                    continue
                goal_0_rate, goal_1_rate = rates
                total_rate = goal_0_rate + goal_1_rate
                if total_rate <= 0:
                    continue

                goal_0_str, goal_1_str = env_name.split('_and_')
                goal_0 = parse_goal_to_binary(goal_0_str, feature_to_idx, n_features)
                goal_1 = parse_goal_to_binary(goal_1_str, feature_to_idx, n_features)

                # Store raw rates (mode-independent) - derived values computed in examples_to_batch
                run_examples.append({
                    'env_name': env_name,
                    'goal_0': goal_0.tolist(),
                    'goal_1': goal_1.tolist(),
                    'goal_0_rate': goal_0_rate,
                    'goal_1_rate': goal_1_rate,
                })

            raw_cache_runs[run_name] = {
                'pipeline': pipeline.to_dict(),
                'examples': run_examples,
            }
        raw_cache_modified = True

    # Save raw cache if modified
    if raw_cache_modified:
        log.info(f"Saving raw cache with {len(raw_cache_runs)} runs")
        with open(raw_cache_file, 'w') as f:
            json.dump({"n_features": n_features, "runs": raw_cache_runs}, f)

    # === STEP 2: Check processed cache ===
    raw_cache_hash = _compute_raw_cache_hash(raw_cache_runs)

    if processed_cache_file.exists() and not raw_cache_modified:
        with open(processed_cache_file, 'rb') as f:
            processed_data = pickle.load(f)

        cached_hash = processed_data.get("raw_cache_hash")
        cached_dims = processed_data.get("padding_dims", {})
        cached_n_features = processed_data.get("n_features")

        if (cached_hash == raw_cache_hash and
            cached_dims == current_padding_dims and
            cached_n_features == n_features):
            # Fast path: load processed examples directly (already deserialized)
            log.info(f"Loading {len(processed_data['examples'])} examples from processed cache")
            return processed_data['examples']

    # === STEP 3: Reprocess all examples ===
    log.info("Reprocessing all examples from raw cache")
    examples = []

    for run_name, run_data in tqdm(raw_cache_runs.items(), desc="Processing examples", leave=False):
        if run_name not in agent_training_pipelines:
            continue

        pipeline, padded_pipeline = agent_training_pipelines[run_name]

        for ex_raw in run_data['examples']:
            # Store raw rates (mode-independent) - derived values computed in examples_to_batch
            example = {
                'pipeline': pipeline,
                'padded_pipeline': padded_pipeline,
                'goal_0': jnp.array(ex_raw['goal_0']),
                'goal_1': jnp.array(ex_raw['goal_1']),
                'goal_0_rate': ex_raw['goal_0_rate'],
                'goal_1_rate': ex_raw['goal_1_rate'],
                'run_name': run_name,
                'env_name': ex_raw['env_name'],
            }
            examples.append(example)

    # Save processed cache (pickle preserves JAX arrays directly)
    log.info(f"Saving processed cache with {len(examples)} examples")
    with open(processed_cache_file, 'wb') as f:
        pickle.dump({
            "n_features": n_features,
            "padding_dims": current_padding_dims,
            "raw_cache_hash": raw_cache_hash,
            "examples": examples,
        }, f)

    return examples


def get_no_goal_features(n_features: int, include_no_goal_feature: bool = False) -> jnp.ndarray:
    """
    Create the no-goal feature vector.

    Args:
        n_features: Total number of features
        include_no_goal_feature: If True, the no-goal option has all features set to 0
            except the dedicated no_goal feature (last dimension) which is set to 1.
            If False, returns all zeros (no_goal value computed via forward pass).

    Returns:
        Feature vector of shape (n_features,)
    """
    if include_no_goal_feature:
        return jnp.zeros(n_features).at[-1].set(1.0)
    else:
        return jnp.zeros(n_features)


def get_no_goal_features_batched(
    n_features: int,
    batch_shape: tuple[int, ...] = (1,),
    include_no_goal_feature: bool = False,
) -> jnp.ndarray:
    """
    Create batched no-goal feature vectors.

    Args:
        n_features: Total number of features
        batch_shape: Shape to prepend to the feature dimension
        include_no_goal_feature: If True, sets the last dimension (no_goal) to 1.
            If False, returns all zeros.

    Returns:
        Feature vectors of shape (*batch_shape, n_features)
    """
    result = jnp.zeros((*batch_shape, n_features))
    if include_no_goal_feature:
        return result.at[..., -1].set(1.0)
    else:
        return result


NoGoalMode = Literal["unweighted_ignore_no_goal", "weighted_ignore_no_goal", "full_distribution"]


def examples_to_batch(
    examples: list[dict],
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> dict[str, jnp.ndarray | list]:
    """
    Convert examples list to structure suitable for batched processing.

    Computes mode-dependent derived values from raw rates:
    - "unweighted_ignore_no_goal": Normalize probabilities, weight = 1.0 (current behavior)
    - "weighted_ignore_no_goal": Normalize probabilities, weight = total_rate
    - "full_distribution": Use raw rates as probabilities, include no-goal probability

    Args:
        examples: List of example dictionaries with goal_0_rate and goal_1_rate
        no_goal_mode: How to handle no-goal (neither) choices

    Returns:
        Dictionary with batched arrays and pipeline lists

    Raises:
        ValueError: If examples list is empty
    """
    if not examples:
        raise ValueError("Cannot create batch from empty examples list")

    # Compute mode-dependent values
    observed_probs_0 = []
    observed_probs_1 = []
    observed_probs_no_goal = []
    weights = []

    for ex in examples:
        goal_0_rate = ex['goal_0_rate']
        goal_1_rate = ex['goal_1_rate']
        total_rate = goal_0_rate + goal_1_rate

        if no_goal_mode == "full_distribution":
            # Use raw rates directly, compute no-goal probability
            observed_probs_0.append(goal_0_rate)
            observed_probs_1.append(goal_1_rate)
            observed_probs_no_goal.append(1.0 - goal_0_rate - goal_1_rate)
            weights.append(1.0)
        else:
            # Normalize to binary distribution (ignore no-goal)
            if total_rate > 0:
                observed_probs_0.append(goal_0_rate / total_rate)
                observed_probs_1.append(goal_1_rate / total_rate)
            else:
                observed_probs_0.append(0.5)
                observed_probs_1.append(0.5)
            observed_probs_no_goal.append(-1.0)  # Sentinel: no-goal not modelled
            # Weight depends on mode
            weights.append(total_rate if no_goal_mode == "weighted_ignore_no_goal" else 1.0)

    # Stack padded pipeline components
    padded_pipelines = [ex['padded_pipeline'] for ex in examples]

    return {
        # Padded pipeline arrays (for models that can use batched processing)
        'goals': jnp.stack([p.goals for p in padded_pipelines]),
        'distractors': jnp.stack([p.distractors for p in padded_pipelines]),
        'stage_masks': jnp.stack([p.stage_mask for p in padded_pipelines]),
        'env_masks': jnp.stack([p.env_mask for p in padded_pipelines]),
        'goal_masks': jnp.stack([p.goal_mask for p in padded_pipelines]),
        'distractor_masks': jnp.stack([p.distractor_mask for p in padded_pipelines]),
        # Evaluation goals
        'goals_0': jnp.stack([ex['goal_0'] for ex in examples]),
        'goals_1': jnp.stack([ex['goal_1'] for ex in examples]),
        'observed_probs_0': jnp.array(observed_probs_0),
        'observed_probs_1': jnp.array(observed_probs_1),
        'observed_probs_no_goal': jnp.array(observed_probs_no_goal),
        'weights': jnp.array(weights),
        # Keep pipeline objects for models that need full structure
        'pipelines': [ex['pipeline'] for ex in examples],
    }

