import jax
import jax.numpy as jnp
import logging

from typing import Any, Literal

from tqdm import tqdm
from tabulate import tabulate
from collections.abc import Callable

from src.preference_modelling.data import examples_to_batch, NoGoalMode
from src.preference_modelling.models import ModelType, get_model
from src.preference_modelling.logging import log_fitted_parameters
from src.preference_modelling.training import train_and_evaluate_model, train_preference_model
from src.preference_modelling.loss import (
    compute_data_loss_batched,
    compute_per_example_losses_batched,
    DEFAULT_BATCH_SIZE,
    EPSILON,
)
from src.preference_modelling.data_structures import TrainingPipeline, PaddedPipeline
from src.preference_modelling.metrics import (
    compute_per_agent_weights,
    compute_example_predictions,
    compute_metrics_from_predictions,
)


log = logging.getLogger(__name__)


def create_env_holdout_splits(
    filtered_run_metrics: dict[str, dict[str, tuple[float, float] | None]],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    env_holdout_fraction: float,
    env_holdout_seed: int,
) -> dict[str, set[str]]:
    """
    Create environment holdout splits for validation.
    
    Args:
        filtered_run_metrics: Dict mapping run names to environment metrics
        agent_training_pipelines: Dict mapping agent names to (pipeline, padded_pipeline)
        env_holdout_fraction: Fraction of environments to hold out per agent
        env_holdout_seed: Random seed for holdout
    
    Returns:
        Dict mapping run_name to set of held-out environment names
    """
    import numpy as np
    rng = np.random.RandomState(env_holdout_seed)
    
    env_holdout_sets = {}
    total_train_envs = 0
    total_val_envs = 0
    
    for run_name, env_metrics in filtered_run_metrics.items():
        if run_name not in agent_training_pipelines:
            continue
            
        # Get all valid environments for this run
        valid_envs = [
            env_name for env_name, rates in env_metrics.items()
            if rates is not None and (rates[0] + rates[1]) > 0
        ]
        
        if len(valid_envs) < 2:
            log.warning(f"Run {run_name} has <2 valid environments, skipping from env holdout")
            continue
        
        # Hold out a fraction
        n_holdout = max(1, int(len(valid_envs) * env_holdout_fraction))
        shuffled = rng.permutation(valid_envs)
        env_holdout_sets[run_name] = set(shuffled[:n_holdout])
        
        total_train_envs += len(valid_envs) - n_holdout
        total_val_envs += n_holdout
        
        log.debug(f"  {run_name}: {len(valid_envs)} envs, holding out {n_holdout}")
    
    log.info(f"Environment split: {total_train_envs} training, {total_val_envs} validation")
    
    # Sanity check
    if total_val_envs < 20:
        log.warning(f"Only {total_val_envs} validation environments total - consider increasing env_holdout_fraction")
    
    return env_holdout_sets


def create_agent_kfold_splits(
    agent_list: list[str],
    agent_kfold_splits: int,
    agent_kfold_seed: int,
) -> list[set[str]]:
    """
    Create K-fold splits for agent cross-validation.
    
    Args:
        agent_list: List of agent names
        agent_kfold_splits: Number of folds
        agent_kfold_seed: Random seed for splits
    
    Returns:
        List of sets, each containing agent names for that fold's validation set
    """
    import numpy as np
    rng = np.random.RandomState(agent_kfold_seed)
    
    # Shuffle agent list
    shuffled_agents = agent_list.copy()
    rng.shuffle(shuffled_agents)
    
    # Create folds
    fold_size = len(shuffled_agents) // agent_kfold_splits
    folds = []
    for i in range(agent_kfold_splits):
        start_idx = i * fold_size
        if i == agent_kfold_splits - 1:
            # Last fold gets any remaining agents
            end_idx = len(shuffled_agents)
        else:
            end_idx = (i + 1) * fold_size
        folds.append(set(shuffled_agents[start_idx:end_idx]))
    
    return folds


def perform_env_holdout_validation(
    training_examples: list[dict],
    env_holdout_sets: dict[str, set[str]],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> dict[str, float | int] | None:
    """
    Perform environment holdout validation.

    Args:
        training_examples: All prepared training examples
        env_holdout_sets: Dict mapping run_name to set of held-out environment names
        n_features: Total number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs

    Returns:
        Dictionary with validation metrics, or None if validation failed
    """
    log.info("="*30)
    log.info("Performing environment holdout validation")
    log.info("="*30)
    
    if not env_holdout_sets:
        log.warning("No valid environment holdout sets created")
        return None
    
    # Split examples by held-out status
    train_examples = []
    val_examples = []
    
    for ex in training_examples:
        run_name = ex['run_name']
        env_name = ex.get('env_name', '')
        
        if run_name not in env_holdout_sets:
            # Run has no holdout set, include in training
            train_examples.append(ex)
        elif env_name in env_holdout_sets[run_name]:
            # This env is held out for validation
            val_examples.append(ex)
        else:
            # This env is for training
            train_examples.append(ex)
    
    if not train_examples or not val_examples:
        log.warning("Insufficient data for environment holdout validation")
        return None

    _, result = train_and_evaluate_model(
        train_examples=train_examples,
        val_examples=val_examples,
        n_features=n_features,
        loss_fn=loss_fn,
        data_loss_fn=data_loss_fn,
        model_type=model_type,
        model_kwargs=model_kwargs,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        no_goal_mode=no_goal_mode,
    )

    table_data = [
        ["Validation loss", f"{result['val_loss']:.6f}"],
        ["Training examples", result['n_train_examples']],
        ["Validation examples", result['n_val_examples']],
    ]
    log.info(f"Environment holdout validation results:\n{tabulate(table_data, tablefmt='simple_outline')}")

    return result


def perform_kfold_agent_validation(
    training_examples: list[dict],
    agent_list: list[str],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    agent_kfold_splits: int = 4,
    agent_kfold_seed: int = 42,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> list[dict[str, float | int]]:
    """
    Perform K-fold cross-validation across agents.

    Args:
        training_examples: All prepared training examples
        agent_list: List of agent names
        n_features: Total number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        agent_kfold_splits: Number of folds
        agent_kfold_seed: Random seed for splits

    Returns:
        List of validation result dictionaries, one per fold
    """
    log.info("="*30)
    log.info(f"Performing {agent_kfold_splits}-fold agent cross-validation")
    log.info("="*30)
    
    folds = create_agent_kfold_splits(
        agent_list=agent_list,
        agent_kfold_splits=agent_kfold_splits,
        agent_kfold_seed=agent_kfold_seed,
    )
    
    log.info(f"Created {agent_kfold_splits} folds with sizes: {[len(f) for f in folds]}")
    
    fold_results = []
    for fold_idx, val_agents in enumerate(tqdm(folds, desc="K-fold CV"), 1):
        # Split examples by agent membership in validation fold
        train_examples = [ex for ex in training_examples if ex['run_name'] not in val_agents]
        val_examples = [ex for ex in training_examples if ex['run_name'] in val_agents]
        
        if not train_examples or not val_examples:
            log.warning(f"Skipping fold {fold_idx} - insufficient data")
            continue

        _, result = train_and_evaluate_model(
            train_examples=train_examples,
            val_examples=val_examples,
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            model_type=model_type,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            no_goal_mode=no_goal_mode,
        )
        result['fold'] = fold_idx
        fold_results.append(result)
    
    # Report K-fold results
    if fold_results:
        val_losses = jnp.array([r['val_loss'] for r in fold_results])
        mean_val_loss = float(jnp.mean(val_losses))
        std_val_loss = float(jnp.std(val_losses))
        
        log.info(f"{agent_kfold_splits}-fold Agent Cross-Validation:")
        log.info(f"  Mean validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")

        table_data = [[f"Fold {r['fold']}", f"{r['val_loss']:.6f}"] for r in fold_results]
        log.info(f"  Per-fold losses:\n{tabulate(table_data, headers=['Fold', 'Val Loss'], tablefmt='simple_outline')}")
    
    return fold_results


def _get_category_index(run_name: str) -> int:
    """
    Compute category index from run name based on 3 binary dimensions.

    The 8 categories are defined by:
    - Bit 0: multi_stage (has "_then_")
    - Bit 1: multi_goal (has "_and_")
    - Bit 2: distractor_present (has "_distractor")

    Returns:
        Integer index from 0-7
    """
    idx = 0
    if "_then_" in run_name:
        idx |= 1
    if "_and_" in run_name:
        idx |= 2
    if "_distractor" in run_name:
        idx |= 4
    return idx


def _get_category_name(category_idx: int) -> str:
    """
    Convert category index back to readable name.

    Args:
        category_idx: Integer index from 0-7

    Returns:
        Human-readable category name
    """
    parts = []

    # Bit 0: multi_stage
    parts.append("multi-stage" if category_idx & 1 else "single-stage")
    # Bit 1: multi_goal
    parts.append("multi-goal" if category_idx & 2 else "single-goal")
    # Bit 2: distractor_present
    parts.append("with distractors" if category_idx & 4 else "without distractors")

    return ", ".join(parts)


def compute_all_training_losses(
    params: dict[str, jnp.ndarray],
    training_examples: list[dict],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    per_example_loss_fn: Callable,
    batch_size: int = DEFAULT_BATCH_SIZE,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> tuple[float, dict[str, float], dict[str, tuple[float, int]]]:
    """
    Compute overall, per-agent, and stratified losses in a single pass.

    This consolidates the previously separate compute_per_agent_losses(),
    compute_stratified_losses(), and compute_data_loss_batched() calls
    into a single pass over the dataset, providing ~10x speedup.

    Args:
        params: Trained model parameters
        training_examples: List of training examples
        agent_training_pipelines: Dict mapping agent names to (pipeline, padded_pipeline)
        per_example_loss_fn: Per-example loss function from create_loss_functions()
        batch_size: Batch size for loss computation

    Returns:
        Tuple of (overall_loss, per_agent_losses, stratified_losses):
        - overall_loss: Float, average loss across all examples
        - per_agent_losses: Dict mapping agent name to loss value
        - stratified_losses: Dict mapping category name to (loss, num_examples) tuple
    """
    # Build agent name to index mapping
    agent_names = list(agent_training_pipelines.keys())
    agent_name_to_idx = {name: idx for idx, name in enumerate(agent_names)}
    n_agents = len(agent_names)

    # Number of categories is 2^3 = 8 (3 binary dimensions)
    n_categories = 8

    # Create index arrays for each example
    agent_indices = jnp.array([
        agent_name_to_idx.get(ex['run_name'], 0)
        for ex in training_examples
    ])

    category_indices = jnp.array([
        _get_category_index(ex['run_name'])
        for ex in training_examples
    ])

    # Create batch once for all examples
    full_batch = examples_to_batch(training_examples, no_goal_mode=no_goal_mode)

    # Compute all per-example losses in ONE pass (batched)
    all_weighted_losses, all_weights = compute_per_example_losses_batched(
        per_example_loss_fn, params, full_batch, batch_size=batch_size
    )

    # ===== Overall loss =====
    total_weighted_loss = jnp.sum(all_weighted_losses)
    total_weight = jnp.sum(all_weights)
    overall_loss = float(total_weighted_loss / jnp.maximum(total_weight, EPSILON))

    # ===== Per-agent losses (using segment_sum) =====
    agent_total_weighted_loss = jax.ops.segment_sum(
        all_weighted_losses, agent_indices, num_segments=n_agents
    )
    agent_total_weight = jax.ops.segment_sum(
        all_weights, agent_indices, num_segments=n_agents
    )
    agent_avg_losses = agent_total_weighted_loss / jnp.maximum(agent_total_weight, EPSILON)

    per_agent_losses = {
        agent_name: float(agent_avg_losses[idx])
        for agent_name, idx in agent_name_to_idx.items()
        if agent_total_weight[idx] > 0
    }

    # ===== Stratified losses (using segment_sum) =====
    category_total_weighted_loss = jax.ops.segment_sum(
        all_weighted_losses, category_indices, num_segments=n_categories
    )
    category_total_weight = jax.ops.segment_sum(
        all_weights, category_indices, num_segments=n_categories
    )
    category_avg_losses = category_total_weighted_loss / jnp.maximum(category_total_weight, EPSILON)

    # Count examples per category
    ones = jnp.ones_like(all_weights)
    category_counts = jax.ops.segment_sum(ones, category_indices, num_segments=n_categories)

    stratified_losses = {
        _get_category_name(cat_idx): (float(category_avg_losses[cat_idx]), int(category_counts[cat_idx]))
        for cat_idx in range(n_categories)
        if category_counts[cat_idx] > 0
    }

    return overall_loss, per_agent_losses, stratified_losses


def perform_no_distractor_to_distractor_validation(
    all_examples: list[dict],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    all_features: list[str],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    per_example_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_name: str | None = None,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> dict[str, float | int] | None:
    """
    Train on runs without distractors, validate on runs with distractors.

    Tests whether a model trained on distractor-free environments generalizes
    to environments with distractor objects present.

    Args:
        all_examples: All prepared training examples
        n_features: Total number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        all_features: List of all feature names
        agent_training_pipelines: Dict mapping agent names to pipelines
        per_example_loss_fn: Per-example loss function from create_loss_functions()
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for loss computation
        save_name: Optional name for saving fitted parameters

    Returns:
        Dictionary with validation metrics, or None if validation failed
    """
    log.info("="*30)
    log.info("Performing no-distractor to distractor validation")
    log.info("="*30)

    train_examples = []
    val_examples = []

    for ex in all_examples:
        run_name = ex['run_name']
        has_distractor = "_distractor" in run_name

        if not has_distractor:
            train_examples.append(ex)
        else:
            val_examples.append(ex)

    if not train_examples:
        log.warning("No distractor-free examples found for training")
        return None

    if not val_examples:
        log.warning("No distractor examples found for validation")
        return None

    log.info(f"Training on {len(train_examples)} distractor-free examples, validating on {len(val_examples)} distractor examples")

    params, result = train_and_evaluate_model(
        train_examples=train_examples,
        val_examples=val_examples,
        n_features=n_features,
        loss_fn=loss_fn,
        data_loss_fn=data_loss_fn,
        model_type=model_type,
        model_kwargs=model_kwargs,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        no_goal_mode=no_goal_mode,
    )

    log_fitted_parameters(
        params=params,
        all_features=all_features,
        agent_training_pipelines=agent_training_pipelines,
        model_type=model_type,
        model_kwargs=model_kwargs,
        save_name=save_name,
    )

    # Compute all losses in a single pass
    overall_loss, agent_losses, stratified_losses = compute_all_training_losses(
        params=params,
        training_examples=all_examples,
        agent_training_pipelines=agent_training_pipelines,
        per_example_loss_fn=per_example_loss_fn,
        batch_size=batch_size,
        no_goal_mode=no_goal_mode,
    )

    # Compute train loss on training subset specifically for comparison
    train_batch = examples_to_batch(train_examples, no_goal_mode=no_goal_mode)
    train_loss = compute_data_loss_batched(data_loss_fn, params, train_batch, batch_size=batch_size)

    result['train_loss'] = train_loss
    result['stratified_val_losses'] = stratified_losses
    result['per_agent_losses'] = agent_losses

    table_data = [
        ["Training loss (no distractors)", f"{train_loss:.6f}"],
        ["Validation loss (with distractors)", f"{result['val_loss']:.6f}"],
        ["Overall loss (all examples)", f"{overall_loss:.6f}"],
        ["Training examples", result['n_train_examples']],
        ["Validation examples", result['n_val_examples']],
    ]
    log.info(f"No-distractor to distractor validation results:\n{tabulate(table_data, tablefmt='simple_outline')}")

    stratified_table = [
        [category, f"{loss:.6f}", f"{n}"]
        for category, (loss, n) in stratified_losses.items()
    ]
    log.info(f"Stratified validation losses:\n{tabulate(stratified_table, headers=['Category', 'Loss', 'Examples'], tablefmt='simple_outline')}")

    # Log best/worst agents
    if agent_losses:
        sorted_losses = sorted(agent_losses.items(), key=lambda x: x[1])
        best_table = [[name, f"{loss:.6f}"] for name, loss in sorted_losses[:5]]
        log.info(f"Best 5 agents:\n{tabulate(best_table, headers=['Agent', 'Loss'], tablefmt='simple_outline')}")
        worst_table = [[name, f"{loss:.6f}"] for name, loss in sorted_losses[-5:]]
        log.info(f"Worst 5 agents:\n{tabulate(worst_table, headers=['Agent', 'Loss'], tablefmt='simple_outline')}")

    return result


def perform_single_stage_to_multi_stage_validation(
    all_examples: list[dict],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    all_features: list[str],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    per_example_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_name: str | None = None,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> dict[str, float | int] | None:
    """
    Train on single-goal single-stage runs, validate on all runs.

    Tests whether a model trained on the single-stage pipelines generalizes to multi-stage ones
    to multi-stage and multi-goal scenarios.

    Args:
        all_examples: All prepared training examples
        n_features: Total number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        all_features: List of all feature names
        agent_training_pipelines: Dict mapping agent names to pipelines
        per_example_loss_fn: Per-example loss function from create_loss_functions()
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for loss computation
        save_name: Optional name for saving fitted parameters

    Returns:
        Dictionary with validation metrics, or None if validation failed
    """
    log.info("="*30)
    log.info("Performing single-stage to multi-stage validation")
    log.info("="*30)

    # Split examples: single-goal single-stage for training, rest for validation
    train_examples = []
    val_examples = []

    for ex in all_examples:
        run_name = ex['run_name']
        is_multi_stage = "_then_" in run_name
        is_multi_goal = "_and_" in run_name

        if not is_multi_stage and not is_multi_goal:
            train_examples.append(ex)
        else:
            val_examples.append(ex)

    if not train_examples:
        log.warning("No single-goal single-stage examples found for training")
        return None

    if not val_examples:
        log.warning("No multi-goal or multi-stage examples found for validation")
        return None

    log.info(f"Training on {len(train_examples)} single-stage examples, validating on {len(val_examples)} multi-stage examples")

    params, result = train_and_evaluate_model(
        train_examples=train_examples,
        val_examples=val_examples,
        n_features=n_features,
        loss_fn=loss_fn,
        data_loss_fn=data_loss_fn,
        model_type=model_type,
        model_kwargs=model_kwargs,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        no_goal_mode=no_goal_mode,
    )

    log_fitted_parameters(
        params=params,
        all_features=all_features,
        agent_training_pipelines=agent_training_pipelines,
        model_type=model_type,
        model_kwargs=model_kwargs,
        save_name=save_name,
    )

    # Compute all losses in a single pass
    overall_loss, agent_losses, stratified_losses = compute_all_training_losses(
        params=params,
        training_examples=all_examples,
        agent_training_pipelines=agent_training_pipelines,
        per_example_loss_fn=per_example_loss_fn,
        batch_size=batch_size,
        no_goal_mode=no_goal_mode,
    )

    # Compute train loss on training subset specifically for comparison
    train_batch = examples_to_batch(train_examples, no_goal_mode=no_goal_mode)
    train_loss = compute_data_loss_batched(data_loss_fn, params, train_batch, batch_size=batch_size)

    result['train_loss'] = train_loss
    result['stratified_val_losses'] = stratified_losses
    result['per_agent_losses'] = agent_losses

    table_data = [
        ["Training loss (single-stage)", f"{train_loss:.6f}"],
        ["Validation loss (multi-stage)", f"{result['val_loss']:.6f}"],
        ["Overall loss (all examples)", f"{overall_loss:.6f}"],
        ["Training examples", result['n_train_examples']],
        ["Validation examples", result['n_val_examples']],
    ]
    log.info(f"Simple-to-complex validation results:\n{tabulate(table_data, tablefmt='simple_outline')}")

    stratified_table = [
        [category, f"{loss:.6f}", f"{n}"]
        for category, (loss, n) in stratified_losses.items()
    ]
    log.info(f"Stratified validation losses:\n{tabulate(stratified_table, headers=['Category', 'Loss', 'Examples'], tablefmt='simple_outline')}")

    # Log best/worst agents
    if agent_losses:
        sorted_losses = sorted(agent_losses.items(), key=lambda x: x[1])
        best_table = [[name, f"{loss:.6f}"] for name, loss in sorted_losses[:5]]
        log.info(f"Best 5 agents:\n{tabulate(best_table, headers=['Agent', 'Loss'], tablefmt='simple_outline')}")
        worst_table = [[name, f"{loss:.6f}"] for name, loss in sorted_losses[-5:]]
        log.info(f"Worst 5 agents:\n{tabulate(worst_table, headers=['Agent', 'Loss'], tablefmt='simple_outline')}")

    return result


def perform_few_shot_validation(
    training_examples: list[dict],
    agent_list: list[str],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    n_train_agents: int = 3,
    n_trials: int = 10,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]] | None = None,
    all_features: list[str] | None = None,
    include_no_goal_feature: bool = False,
) -> dict[str, Any] | None:
    """
    Train on a small random subset of agents, evaluate on the full dataset.

    Repeats the process multiple times with different random samples to
    estimate generalization performance and variance.

    Args:
        training_examples: All prepared training examples
        agent_list: List of agent names
        n_features: Total number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        model_type: Which model formulation to use
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        n_train_agents: Number of agents to sample for training (default: 3)
        n_trials: Number of random trials to run (default: 10)
        seed: Random seed for reproducibility
        batch_size: Batch size for loss computation

    Returns:
        Dictionary with:
            - 'mean_full_loss': Mean loss on full dataset across trials
            - 'std_full_loss': Std of loss on full dataset
            - 'std_error_full_loss': Standard error of the mean
            - 'mean_train_loss': Mean loss on training subset
            - 'std_train_loss': Std of training loss
            - 'mean_holdout_loss': Mean loss on held-out agents only
            - 'std_holdout_loss': Std of holdout loss
            - 'n_train_agents': Number of agents used for training
            - 'n_total_agents': Total number of agents
            - 'n_trials': Number of trials run
            - 'per_trial_results': List of per-trial result dicts
        Or None if validation could not be performed.
    """
    import numpy as np

    log.info("="*30)
    log.info(f"Performing few-shot validation ({n_train_agents} agents, {n_trials} trials)")
    log.info("="*30)

    if n_train_agents >= len(agent_list):
        log.warning(f"n_train_agents ({n_train_agents}) >= total agents ({len(agent_list)}), "
                    "skipping few-shot validation")
        return None

    rng = np.random.RandomState(seed)

    # Prepare full dataset batch for evaluation
    full_batch = examples_to_batch(training_examples, no_goal_mode=no_goal_mode)

    per_trial_results = []

    for trial_idx in tqdm(range(n_trials), desc="Few-shot trials"):
        # Sample agents for training
        train_agents = set(rng.choice(agent_list, size=n_train_agents, replace=False))

        # Split examples
        train_examples_trial = [ex for ex in training_examples if ex['run_name'] in train_agents]
        holdout_examples = [ex for ex in training_examples if ex['run_name'] not in train_agents]

        if not train_examples_trial:
            log.warning(f"Trial {trial_idx}: No training examples, skipping")
            continue

        # Train model on subset
        train_batch = examples_to_batch(train_examples_trial, no_goal_mode=no_goal_mode)

        params = train_preference_model(
            train_dataset=train_batch,
            n_features=n_features,
            loss_fn=loss_fn,
            model_type=model_type,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            show_progress=False,
        )

        # Evaluate on full dataset, training subset, and holdout
        full_loss = compute_data_loss_batched(data_loss_fn, params, full_batch, batch_size=batch_size)
        train_loss = compute_data_loss_batched(data_loss_fn, params, train_batch, batch_size=batch_size)

        holdout_batch = examples_to_batch(holdout_examples, no_goal_mode=no_goal_mode)
        holdout_loss = compute_data_loss_batched(data_loss_fn, params, holdout_batch, batch_size=batch_size)

        trial_result = {
            'trial': trial_idx,
            'train_agents': list(train_agents),
            'full_loss': float(full_loss),
            'train_loss': float(train_loss),
            'holdout_loss': float(holdout_loss),
            'n_train_examples': len(train_examples_trial),
            'n_holdout_examples': len(holdout_examples),
        }

        # Post-hoc dual-metric evaluation of the fitted params: report the same
        # KL both as the full 3-way distribution and 2-way renormalised (the
        # elo report's Experiment 2 methodology, applied per sweep point).
        # Per-agent weights + predictions are computed once over the full set,
        # then sliced for the train / holdout subsets.
        if agent_training_pipelines is not None and all_features is not None:
            agent_weights = compute_per_agent_weights(
                params, model_type, model_kwargs or {}, all_features, agent_training_pipelines,
            )
            pred_probs, obs_probs = compute_example_predictions(
                params, model_type, model_kwargs or {}, agent_weights,
                training_examples, include_no_goal_feature, n_features,
            )
            is_train = np.array([ex['run_name'] in train_agents for ex in training_examples])
            for subset_name, sel in (
                ('full', np.ones(len(training_examples), dtype=bool)),
                ('train', is_train),
                ('holdout', ~is_train),
            ):
                m = compute_metrics_from_predictions(pred_probs[sel], obs_probs[sel])
                trial_result[f'{subset_name}_3way_kl'] = m['3way_kl_mean']
                trial_result[f'{subset_name}_2way_kl'] = m['2way_kl_mean']
                trial_result[f'{subset_name}_3way_tv'] = m['3way_tv_mean']
                trial_result[f'{subset_name}_2way_tv'] = m['2way_tv_mean']
                trial_result[f'{subset_name}_3way_brier'] = m['3way_brier_mean']
                trial_result[f'{subset_name}_2way_brier'] = m['2way_brier_mean']
                trial_result[f'{subset_name}_3way_dir_acc'] = m['3way_directional_accuracy']
                trial_result[f'{subset_name}_2way_dir_acc'] = m['2way_directional_accuracy']

        per_trial_results.append(trial_result)

    if not per_trial_results:
        log.warning("No successful trials in few-shot validation")
        return None

    # Aggregate results
    full_losses = jnp.array([r['full_loss'] for r in per_trial_results])
    train_losses = jnp.array([r['train_loss'] for r in per_trial_results])
    holdout_losses = jnp.array([r['holdout_loss'] for r in per_trial_results])

    n_successful_trials = len(per_trial_results)

    results = {
        'mean_full_loss': float(jnp.mean(full_losses)),
        'std_full_loss': float(jnp.std(full_losses)),
        'std_error_full_loss': float(jnp.std(full_losses) / jnp.sqrt(n_successful_trials)),
        'mean_train_loss': float(jnp.mean(train_losses)),
        'std_train_loss': float(jnp.std(train_losses)),
        'mean_holdout_loss': float(jnp.mean(holdout_losses)),
        'std_holdout_loss': float(jnp.std(holdout_losses)),
        'std_error_holdout_loss': float(jnp.std(holdout_losses) / jnp.sqrt(n_successful_trials)),
        'n_train_agents': n_train_agents,
        'n_total_agents': len(agent_list),
        'n_trials': n_successful_trials,
        'per_trial_results': per_trial_results,
    }

    # Aggregate post-hoc dual metrics across trials, if they were computed
    dual_keys = [
        k for k in per_trial_results[0]
        if any(k.startswith(s + '_') for s in ('full', 'train', 'holdout'))
        and any(t in k for t in ('_kl', '_tv', '_brier', '_dir_acc'))
    ]
    for k in dual_keys:
        vals = jnp.array([r[k] for r in per_trial_results])
        results[f'mean_{k}'] = float(jnp.nanmean(vals))
        results[f'std_{k}'] = float(jnp.nanstd(vals))
        results[f'std_error_{k}'] = float(jnp.nanstd(vals) / jnp.sqrt(n_successful_trials))

    # Log summary
    table_data = [
        ["Train agents", f"{n_train_agents} / {len(agent_list)}"],
        ["Trials", n_successful_trials],
        ["Full dataset loss", f"{results['mean_full_loss']:.6f} +/- {results['std_error_full_loss']:.6f}"],
        ["Training subset loss", f"{results['mean_train_loss']:.6f} +/- {results['std_train_loss']:.6f}"],
        ["Holdout agents loss", f"{results['mean_holdout_loss']:.6f} +/- {results['std_error_holdout_loss']:.6f}"],
    ]
    if 'mean_holdout_3way_kl' in results:
        table_data += [
            ["Holdout 3-way KL", f"{results['mean_holdout_3way_kl']:.6f} +/- {results['std_error_holdout_3way_kl']:.6f}"],
            ["Holdout 2-way KL", f"{results['mean_holdout_2way_kl']:.6f} +/- {results['std_error_holdout_2way_kl']:.6f}"],
            ["Full 3-way KL", f"{results['mean_full_3way_kl']:.6f} +/- {results['std_error_full_3way_kl']:.6f}"],
            ["Full 2-way KL", f"{results['mean_full_2way_kl']:.6f} +/- {results['std_error_full_2way_kl']:.6f}"],
        ]
    log.info(f"Few-shot validation results:\n{tabulate(table_data, tablefmt='simple_outline')}")

    return results


def evaluate_untrained_initial_params(
    training_examples: list[dict],
    n_features: int,
    data_loss_fn: Callable,
    model_type: ModelType,
    model_kwargs: dict | None,
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    all_features: list[str],
    include_no_goal_feature: bool,
    no_goal_mode: NoGoalMode,
    n_total_agents: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """
    The "n=0" point: how well the *untrained initial* hyperparameters fit the
    data (no agent sampling, no SGD). Deterministic -> zero variance, one
    evaluation. With 0 training agents the entire dataset is held out, so
    holdout_* == full_*. Returns the same scalar schema as
    `perform_few_shot_validation` so it slots into the sweep/serialisation/plots
    unchanged.
    """
    log.info("="*30)
    log.info("Evaluating untrained initial params (n=0)")
    log.info("="*30)

    model = get_model(model_type, **(model_kwargs or {}))
    params = model.init_params_from_spec(n_features)

    full_batch = examples_to_batch(training_examples, no_goal_mode=no_goal_mode)
    full_loss = float(compute_data_loss_batched(
        data_loss_fn, params, full_batch, batch_size=batch_size))

    agent_weights = compute_per_agent_weights(
        params, model_type, model_kwargs or {}, all_features, agent_training_pipelines)
    pred_probs, obs_probs = compute_example_predictions(
        params, model_type, model_kwargs or {}, agent_weights,
        training_examples, include_no_goal_feature, n_features)
    m = compute_metrics_from_predictions(pred_probs, obs_probs)

    results: dict[str, Any] = {
        'n_train_agents': 0,
        'n_total_agents': n_total_agents,
        'n_trials': 1,
        'mean_full_loss': full_loss, 'std_full_loss': 0.0, 'std_error_full_loss': 0.0,
        'mean_holdout_loss': full_loss, 'std_holdout_loss': 0.0, 'std_error_holdout_loss': 0.0,
        'mean_train_loss': full_loss, 'std_train_loss': 0.0,
    }
    # holdout == full (no agent was used to fit). Mirror the per-trial schema.
    for subset in ('full', 'holdout'):
        for var in ('3way', '2way'):
            for short, mkey in (
                ('kl', f'{var}_kl_mean'),
                ('tv', f'{var}_tv_mean'),
                ('brier', f'{var}_brier_mean'),
                ('dir_acc', f'{var}_directional_accuracy'),
            ):
                base = f'{subset}_{var}_{short}'
                results[f'mean_{base}'] = float(m[mkey])
                results[f'std_{base}'] = 0.0
                results[f'std_error_{base}'] = 0.0

    log.info(f"Untrained init params: holdout 3-way KL {m['3way_kl_mean']:.6f}, "
             f"2-way KL {m['2way_kl_mean']:.6f}, full-dist loss {full_loss:.6f}")
    return results


def perform_few_shot_sweep_validation(
    training_examples: list[dict],
    agent_list: list[str],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    train_agent_counts: list[int] | None = None,
    n_trials_per_size: int = 10,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]] | None = None,
    all_features: list[str] | None = None,
    include_no_goal_feature: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    Run few-shot validation across multiple training set sizes.

    Useful for plotting learning curves to understand sample complexity.

    Args:
        training_examples: All prepared training examples
        agent_list: List of agent names
        n_features: Total number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        model_type: Which model formulation to use
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        train_agent_counts: List of training set sizes to try (e.g., [3, 10, 50])
            If None, uses powers of 2 up to half the agents
        n_trials_per_size: Number of trials per training set size
        seed: Random seed for reproducibility
        batch_size: Batch size for loss computation

    Returns:
        Dict mapping n_train_agents -> results dict from perform_few_shot_validation
    """
    import numpy as np

    log.info("="*30)
    log.info(f"Performing few-shot sweep validation")
    log.info("="*30)

    if train_agent_counts is None:
        # Default: powers of 2 up to half the agents
        max_agents = len(agent_list) // 2
        train_agent_counts = [2**i for i in range(int(np.log2(max_agents)) + 1)]
        train_agent_counts = [n for n in train_agent_counts if n < len(agent_list)]

    log.info(f"Training set sizes to evaluate: {train_agent_counts}")
    log.info(f"Trials per size: {n_trials_per_size}")

    all_results = {}

    # n=0: untrained initial-params fit (deterministic; no training).
    if 0 in train_agent_counts:
        train_agent_counts = [n for n in train_agent_counts if n != 0]
        if agent_training_pipelines is not None and all_features is not None:
            all_results[0] = evaluate_untrained_initial_params(
                training_examples=training_examples,
                n_features=n_features,
                data_loss_fn=data_loss_fn,
                model_type=model_type,
                model_kwargs=model_kwargs,
                agent_training_pipelines=agent_training_pipelines,
                all_features=all_features,
                include_no_goal_feature=include_no_goal_feature,
                no_goal_mode=no_goal_mode,
                n_total_agents=len(agent_list),
                batch_size=batch_size,
            )
        else:
            log.warning("n=0 (untrained init params) requested but "
                        "agent_training_pipelines/all_features not provided; skipping")

    for n_agents in train_agent_counts:
        results = perform_few_shot_validation(
            training_examples=training_examples,
            agent_list=agent_list,
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            model_type=model_type,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            n_train_agents=n_agents,
            n_trials=n_trials_per_size,
            seed=seed,
            batch_size=batch_size,
            no_goal_mode=no_goal_mode,
            agent_training_pipelines=agent_training_pipelines,
            all_features=all_features,
            include_no_goal_feature=include_no_goal_feature,
        )
        if results is not None:
            all_results[n_agents] = results

    # Log summary table
    if all_results:
        has_dual = any('mean_holdout_3way_kl' in res for res in all_results.values())
        headers = ['N Train Agents', 'Holdout Loss', 'Train Loss']
        if has_dual:
            headers += ['Holdout 3-way KL', 'Holdout 2-way KL']
        summary_table = []
        for n_agents, res in sorted(all_results.items()):
            row = [
                n_agents,
                f"{res['mean_holdout_loss']:.6f} +/- {res['std_error_holdout_loss']:.6f}",
                f"{res['mean_train_loss']:.6f}",
            ]
            if has_dual:
                row += [
                    f"{res['mean_holdout_3way_kl']:.6f} +/- {res['std_error_holdout_3way_kl']:.6f}",
                    f"{res['mean_holdout_2way_kl']:.6f} +/- {res['std_error_holdout_2way_kl']:.6f}",
                ]
            summary_table.append(row)
        log.info(f"Few-shot sweep summary:\n{tabulate(summary_table, headers=headers, tablefmt='simple_outline')}")

    return all_results

