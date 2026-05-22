import jax
import jax.numpy as jnp

from tqdm import tqdm
from typing import Literal
from collections.abc import Callable

from src.preference_modelling.models import ModelType, get_model
from src.preference_modelling.data_structures import PaddedPipeline


EPSILON = 1e-8
DEFAULT_BATCH_SIZE = 64

NoGoalMode = Literal["unweighted_ignore_no_goal", "weighted_ignore_no_goal", "full_distribution"]


def create_loss_functions(
    model_type: ModelType,
    model_kwargs: dict | None = None,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> tuple[Callable, Callable, Callable]:
    """
    Create loss functions for training the preference model.

    Args:
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        no_goal_mode: How to handle no-goal (neither) choices in loss computation:
            - "unweighted_ignore_no_goal": Ignore no-goal, all examples have weight 1.0 (default)
            - "weighted_ignore_no_goal": Ignore no-goal, weight examples by goal_0_rate + goal_1_rate
            - "full_distribution": Include no-goal probability mass in the loss computation

    Returns:
        Tuple of (loss_fn, data_loss_fn, per_example_loss_fn):
        - loss_fn: Full loss with regularisation (for training)
        - data_loss_fn: Data-only loss without regularisation (for evaluation)
        - per_example_loss_fn: Returns (weighted_losses, weights) arrays for aggregation
    """
    model_kwargs = model_kwargs or {}
    model = get_model(model_type, **model_kwargs)

    # Create vmapped version that operates on indices
    def example_loss_by_idx(
        params: dict[str, jnp.ndarray],
        batch: dict[str, jnp.ndarray],
        idx: int | jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss for example at index idx."""
        padded = PaddedPipeline(
            goals=batch['goals'][idx],
            distractors=batch['distractors'][idx],
            stage_mask=batch['stage_masks'][idx],
            env_mask=batch['env_masks'][idx],
            goal_mask=batch['goal_masks'][idx],
            distractor_mask=batch['distractor_masks'][idx],
        )
        return model.single_example_loss(
            params,
            padded,
            batch['goals_0'][idx],
            batch['goals_1'][idx],
            batch['observed_probs_0'][idx],
            batch['observed_probs_1'][idx],
            batch['observed_probs_no_goal'][idx],
            batch['weights'][idx],
        )

    # Vmap over indices
    batched_example_loss = jax.vmap(
        example_loss_by_idx,
        in_axes=(None, None, 0),
    )

    @jax.jit
    def per_example_loss_fn(
        params: dict[str, jnp.ndarray],
        batch: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-example weighted losses (no averaging).

        Returns:
            Tuple of (weighted_losses, weights) arrays of shape (batch_size,)
        """
        # Apply weight override for "unweighted_ignore_no_goal" mode only
        # Note: weights are already computed in examples_to_batch based on mode
        if no_goal_mode == "unweighted_ignore_no_goal":
            batch = {**batch, "weights": jnp.ones_like(batch['weights'])}

        batch_size = batch['weights'].shape[0]
        indices = jnp.arange(batch_size)
        weighted_losses = batched_example_loss(params, batch, indices)
        return weighted_losses, batch['weights']

    @jax.jit
    def data_loss_fn(params: dict[str, jnp.ndarray], batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute average weighted data loss over batch (no regularisation)."""
        weighted_losses, weights = per_example_loss_fn(params, batch)
        total_weight = jnp.sum(weights)
        return jnp.sum(weighted_losses) / jnp.maximum(total_weight, EPSILON)

    def loss_fn(params: dict[str, jnp.ndarray], batch: dict[str, jnp.ndarray | list]) -> jnp.ndarray:
        """Compute average weighted loss with regularisation (for training)."""
        data_loss = data_loss_fn(params, batch)
        regularisation_loss = model.compute_regularisation_loss(params)
        return data_loss + regularisation_loss

    return loss_fn, data_loss_fn, per_example_loss_fn


def compute_per_example_losses_batched(
    per_example_loss_fn: Callable,
    params: dict[str, jnp.ndarray],
    batch: dict[str, jnp.ndarray | list],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute per-example losses over large dataset in mini-batches.

    Args:
        per_example_loss_fn: Per-example loss function from create_loss_functions()
        params: Model parameters dict
        batch: Full dataset batch dict from examples_to_batch()
        batch_size: Examples per mini-batch

    Returns:
        Tuple of (all_weighted_losses, all_weights) arrays of shape (dataset_size,)
    """
    dataset_size = batch['weights'].shape[0]
    array_batch = {k: v for k, v in batch.items() if isinstance(v, jnp.ndarray)}

    # Small datasets: use direct computation
    if dataset_size <= batch_size:
        return per_example_loss_fn(params, array_batch)

    # Collect results from mini-batches
    all_weighted_losses = []
    all_weights = []

    for start_idx in tqdm(range(0, dataset_size, batch_size), desc="Computing per-example losses", leave=False):
        end_idx = min(start_idx + batch_size, dataset_size)
        mini_batch = {k: v[start_idx:end_idx] for k, v in array_batch.items()}

        weighted_losses, weights = per_example_loss_fn(params, mini_batch)
        all_weighted_losses.append(weighted_losses)
        all_weights.append(weights)

    return jnp.concatenate(all_weighted_losses), jnp.concatenate(all_weights)


def compute_data_loss_batched(
    data_loss_fn: Callable,
    params: dict[str, jnp.ndarray],
    batch: dict[str, jnp.ndarray | list],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    """
    Compute data-only loss over large dataset in mini-batches.

    Properly aggregates weighted losses across batches by accumulating
    weighted sums rather than averaging batch losses.

    Args:
        data_loss_fn: Data-only loss function from create_loss_functions()
        params: Model parameters dict
        batch: Full dataset batch dict from examples_to_batch()
        batch_size: Examples per mini-batch (default 256)

    Returns:
        Scalar data loss value as Python float
    """
    dataset_size = batch['weights'].shape[0]

    # Small datasets: use direct computation
    if dataset_size <= batch_size:
        return float(data_loss_fn(params, batch))

    # Extract array-only items for slicing
    array_batch = {k: v for k, v in batch.items() if isinstance(v, jnp.ndarray)}

    total_weighted_loss = 0.0
    total_weight = 0.0

    for start_idx in tqdm(range(0, dataset_size, batch_size), desc="Computing data loss", leave=False):
        end_idx = min(start_idx + batch_size, dataset_size)
        mini_batch = {k: v[start_idx:end_idx] for k, v in array_batch.items()}

        batch_loss = float(data_loss_fn(params, mini_batch))
        batch_weight = float(jnp.sum(mini_batch['weights']))

        total_weighted_loss += batch_loss * batch_weight
        total_weight += batch_weight

    return total_weighted_loss / max(total_weight, EPSILON)
