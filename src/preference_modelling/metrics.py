"""
Post-hoc evaluation metrics for fitted preference models.

Computes total variation distance, Brier score, and directional accuracy
alongside KL divergence, using saved model parameters without retraining.
"""
import jax
import jax.numpy as jnp
import numpy as np
import json
import logging

from typing import Any
from pathlib import Path
from collections.abc import Callable

from src.preference_modelling.models import get_model, ModelType
from src.preference_modelling.models.base import (
    _compute_agent_params_from_pipeline_batched,
    get_no_goal_features,
)
from src.preference_modelling.data_structures import TrainingPipeline, PaddedPipeline
from src.preference_modelling.data import prepare_examples, examples_to_batch
from src.preference_modelling.loss import create_loss_functions, compute_per_example_losses_batched, EPSILON


log = logging.getLogger(__name__)


def load_saved_params(
    summaries_path: str,
    model_key: str,
) -> tuple[dict[str, jnp.ndarray], str, dict]:
    """
    Load saved model parameters from the summaries JSON file.

    Args:
        summaries_path: Path to developmental_model_fit_summaries.json or
            validation_sweep_results/merged_summaries.json
        model_key: Key in the summaries dict (e.g. 'without_feature_full_dist')

    Returns:
        Tuple of (params dict with jnp arrays, model_type string, model_kwargs dict)
    """
    with open(summaries_path) as f:
        summaries = json.load(f)

    if model_key not in summaries:
        raise KeyError(f"Model key '{model_key}' not found. Available: {list(summaries.keys())}")

    entry = summaries[model_key]
    raw_params = entry['params']
    model_type = entry['model_type']
    model_kwargs = entry.get('model_kwargs', {})

    # Convert lists back to jnp arrays
    params = {}
    for k, v in raw_params.items():
        if isinstance(v, list):
            params[k] = jnp.array(v)
        else:
            params[k] = jnp.array(v)

    return params, model_type, model_kwargs


def compute_per_agent_weights(
    params: dict[str, jnp.ndarray],
    model_type: ModelType,
    model_kwargs: dict,
    all_features: list[str],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
) -> dict[str, jnp.ndarray]:
    """
    Compute per-agent weight vectors by running the model forward through
    each agent's training pipeline. Reuses model's learn_agent_parameters_to_be_saved.

    Returns:
        Dict mapping agent names to weight vectors (jnp arrays).
    """
    model = get_model(model_type, **model_kwargs)
    raw_agent_params = model.learn_agent_parameters_to_be_saved(
        params, all_features, agent_training_pipelines,
    )

    # Convert from {agent: {feature: value}} to {agent: jnp.array}
    n_features = len(all_features)

    # Determine label space based on model type
    is_quadratic = 'quadratic' in model_type
    if is_quadratic:
        # Quadratic model: labels are base features + pairwise products
        labels = list(all_features)
        for f_i in all_features:
            for f_j in all_features:
                labels.append(f"{f_i}*{f_j}")
    else:
        latent_dim = getattr(model, '_cached_latent_dimension', None) or n_features
        if latent_dim == n_features:
            labels = all_features
        else:
            labels = [f"z{i}" for i in range(latent_dim)]

    agent_weights = {}
    for agent_name, weight_dict in raw_agent_params.items():
        w = jnp.array([weight_dict.get(label, 0.0) for label in labels])
        agent_weights[agent_name] = w

    return agent_weights


def compute_per_agent_per_goal_values(
    params: dict[str, jnp.ndarray],
    model_type: ModelType,
    model_kwargs: dict,
    agent_weights: dict[str, jnp.ndarray],
    possible_goals: list[str],
    feature_to_idx: dict[str, int],
    n_features: int,
) -> dict[str, dict[str, float]]:
    """
    Compute model value for each (agent, goal) pair.

    Returns:
        Dict mapping agent_name -> {goal_name: value}.
    """
    model = get_model(model_type, **model_kwargs)

    # Build goal feature matrix
    goal_features = jnp.zeros((len(possible_goals), n_features))
    for i, goal in enumerate(possible_goals):
        colour, shape = goal.split('_', 1)
        if colour in feature_to_idx:
            goal_features = goal_features.at[i, feature_to_idx[colour]].set(1.0)
        if shape in feature_to_idx:
            goal_features = goal_features.at[i, feature_to_idx[shape]].set(1.0)

    agent_goal_values: dict[str, dict[str, float]] = {}

    for agent_name, w in agent_weights.items():
        # Use model's value function: forward_batched(ws, goals, params)
        values = model.value_function.forward_batched(
            w[None, :], goal_features, params,
        )[0]  # (n_goals,)

        agent_goal_values[agent_name] = {
            goal: float(values[i]) for i, goal in enumerate(possible_goals)
        }

    return agent_goal_values


def compute_example_predictions(
    params: dict[str, jnp.ndarray],
    model_type: ModelType,
    model_kwargs: dict,
    agent_weights: dict[str, jnp.ndarray],
    training_examples: list[dict],
    include_no_goal_feature: bool,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted and observed probability distributions for each training example.

    Uses the model's value function applied to pre-computed per-agent weights.

    Args:
        params: Saved model parameters
        model_type: Model type string
        model_kwargs: Model construction kwargs
        agent_weights: Pre-computed per-agent weight vectors
        training_examples: List of training example dicts
        include_no_goal_feature: Whether no-goal has a dedicated feature
        n_features: Number of features

    Returns:
        Tuple of (predicted_probs, observed_probs), each shape (n_examples, 3)
        where columns are [goal_0, goal_1, no_goal].
    """
    model = get_model(model_type, **model_kwargs)
    no_goal_features = get_no_goal_features(n_features, include_no_goal_feature)

    n_examples = len(training_examples)

    # Build observed probs array
    observed_probs = np.zeros((n_examples, 3))
    for i, ex in enumerate(training_examples):
        rate_0 = ex['goal_0_rate']
        rate_1 = ex['goal_1_rate']
        observed_probs[i] = [rate_0, rate_1, 1.0 - rate_0 - rate_1]

    # Build index mapping: agent_name -> list of example indices
    agent_name_list = list(agent_weights.keys())
    agent_name_to_idx = {name: idx for idx, name in enumerate(agent_name_list)}

    # Stack all agent weights into a matrix: (n_agents, latent_dim)
    agent_w_matrix = np.stack([np.array(agent_weights[name]) for name in agent_name_list])

    # Build per-example arrays: agent index, goal_0 features, goal_1 features
    example_agent_idx = np.zeros(n_examples, dtype=np.int32)
    goal_0_features = np.zeros((n_examples, n_features))
    goal_1_features = np.zeros((n_examples, n_features))

    for i, ex in enumerate(training_examples):
        agent_name = ex['run_name']
        if agent_name in agent_name_to_idx:
            example_agent_idx[i] = agent_name_to_idx[agent_name]
        goal_0_features[i] = ex['goal_0'][:n_features]
        goal_1_features[i] = ex['goal_1'][:n_features]

    is_quadratic = 'quadratic' in model_type

    if is_quadratic:
        # Quadratic model: expand features to [phi, flatten(phi outer phi)]
        # then logit = (S_diag * w) . expanded_phi (all element-wise then sum)
        def expand(features: np.ndarray) -> np.ndarray:
            """Vectorised quadratic expansion: (..., n) -> (..., n + n^2)"""
            outer = features[..., :, None] * features[..., None, :]  # (..., n, n)
            outer_flat = outer.reshape(*features.shape[:-1], n_features * n_features)
            return np.concatenate([features, outer_flat], axis=-1)

        S_diag = np.array(params['S_diag'])  # (n + n^2,)

        # Per-agent effective weights in expanded space: (n_agents, n + n^2)
        effective_ws = S_diag[None, :] * agent_w_matrix
        example_ews = effective_ws[example_agent_idx]  # (n_examples, n + n^2)

        # Expand goal features
        goal_0_expanded = expand(goal_0_features)  # (n_examples, n + n^2)
        goal_1_expanded = expand(goal_1_features)
        no_goal_expanded = expand(np.array(no_goal_features))  # (n + n^2,)

        logit_0 = np.sum(example_ews * goal_0_expanded, axis=1)
        logit_1 = np.sum(example_ews * goal_1_expanded, axis=1)
        logit_ng = np.sum(example_ews * no_goal_expanded[None, :], axis=1)

    elif hasattr(model, 'value_function'):
        # Linear / standard path: value(w, phi) = (S @ w) . phi
        # Compute effective_ws = agent_w_matrix @ S.T
        all_effective_ws = np.array(
            model.value_function.forward_batched(
                jnp.array(agent_w_matrix),
                jnp.eye(n_features),  # identity to extract effective weights
                params,
            )
        )  # (n_agents, n_features)

        example_ews = all_effective_ws[example_agent_idx]  # (n_examples, n_features)

        logit_0 = np.sum(example_ews * goal_0_features, axis=1)
        logit_1 = np.sum(example_ews * goal_1_features, axis=1)
        no_goal_np = np.array(no_goal_features)
        logit_ng = np.sum(example_ews * no_goal_np[None, :], axis=1)
    else:
        raise NotImplementedError(f"Vectorized predictions not supported for {model_type}")

    # Softmax over [logit_0, logit_1, logit_ng]
    logits_all = np.stack([logit_0, logit_1, logit_ng], axis=1)  # (n_examples, 3)
    # Numerically stable softmax
    logits_all -= logits_all.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_all)
    predicted_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return predicted_probs, observed_probs


def compute_uniform_baseline_metrics(
    training_examples: list[dict],
    directional_threshold: float = 0.10,
) -> dict[str, float]:
    """
    Compute metrics for a uniform baseline that predicts (1/3, 1/3, 1/3) for
    every example.

    The uniform predictor is the constant distribution at the centre of the
    3-simplex --- it has no per-example variance, so it is "smooth" but also
    "opinionless". Its directional accuracy is always 0% under strict sign
    comparison because pred_0 == pred_1 always (i.e. np.sign(0) does not match
    np.sign(±1)). This is reported as-is rather than overridden.

    Args:
        training_examples: List of training examples
        directional_threshold: Threshold for directional accuracy

    Returns:
        Dict with same metric keys as compute_metrics_from_predictions.
    """
    n_examples = len(training_examples)

    observed_probs = np.zeros((n_examples, 3))
    for i, ex in enumerate(training_examples):
        rate_0 = ex['goal_0_rate']
        rate_1 = ex['goal_1_rate']
        observed_probs[i] = [rate_0, rate_1, 1.0 - rate_0 - rate_1]

    predicted_probs = np.full((n_examples, 3), 1.0 / 3.0)

    return compute_metrics_from_predictions(
        predicted_probs, observed_probs, directional_threshold,
    )


def compute_random_baseline_metrics(
    training_examples: list[dict],
    directional_threshold: float = 0.10,
    seed: int = 42,
    n_repeats: int = 10,
) -> dict[str, float]:
    """
    Compute metrics for a random baseline that samples a fresh random
    probability distribution for each example, drawn uniformly from the
    3-simplex via Dirichlet(1, 1, 1).

    Repeated `n_repeats` times with different random seeds; metrics are
    averaged across repeats for variance reduction.

    Args:
        training_examples: List of training examples
        directional_threshold: Threshold for directional accuracy
        seed: Base random seed
        n_repeats: Number of random samples to average over

    Returns:
        Dict with same metric keys as compute_metrics_from_predictions.
    """
    n_examples = len(training_examples)

    observed_probs = np.zeros((n_examples, 3))
    for i, ex in enumerate(training_examples):
        rate_0 = ex['goal_0_rate']
        rate_1 = ex['goal_1_rate']
        observed_probs[i] = [rate_0, rate_1, 1.0 - rate_0 - rate_1]

    # Average metrics across multiple random samples
    all_metric_dicts: list[dict[str, float]] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        # Sample from Dirichlet(1,1,1) → uniform over the 3-simplex
        predicted_probs = rng.dirichlet([1.0, 1.0, 1.0], size=n_examples)
        m = compute_metrics_from_predictions(
            predicted_probs, observed_probs, directional_threshold,
        )
        all_metric_dicts.append(m)

    # Average across repeats (for keys that are scalar floats)
    averaged: dict[str, float] = {}
    metric_keys = [k for k, v in all_metric_dicts[0].items() if isinstance(v, (int, float))]
    for key in metric_keys:
        values = [d[key] for d in all_metric_dicts]
        averaged[key] = float(np.mean(values))

    averaged['n_repeats'] = n_repeats
    return averaged


def _compute_metrics_for_distribution(
    predicted: np.ndarray,
    observed: np.ndarray,
    directional_threshold: float,
    prefix: str,
) -> dict[str, float]:
    """
    Compute KL, TV, Brier, and directional accuracy for a set of
    predicted/observed probability distributions.

    Args:
        predicted: (n_examples, k) predicted probabilities
        observed: (n_examples, k) observed probabilities
        directional_threshold: Minimum |obs_0 - obs_1| for directional accuracy
        prefix: String prefix for metric keys (e.g. '3way_' or '2way_')

    Returns:
        Dict with prefixed metric names and values.
    """
    eps = 1e-10

    kl_per_example = np.sum(
        observed * (np.log(observed + eps) - np.log(predicted + eps)),
        axis=1,
    )
    tv_per_example = 0.5 * np.sum(np.abs(predicted - observed), axis=1)
    brier_per_example = np.sum((predicted - observed) ** 2, axis=1)

    # Directional accuracy (always based on first two columns: goal_0 vs goal_1)
    obs_diff = observed[:, 0] - observed[:, 1]
    pred_diff = predicted[:, 0] - predicted[:, 1]
    directional_mask = np.abs(obs_diff) >= directional_threshold
    n_directional = int(np.sum(directional_mask))

    if n_directional > 0:
        directional_correct = np.sign(obs_diff[directional_mask]) == np.sign(pred_diff[directional_mask])
        directional_accuracy = float(np.mean(directional_correct))
    else:
        directional_accuracy = float('nan')

    return {
        f'{prefix}kl_mean': float(np.mean(kl_per_example)),
        f'{prefix}kl_std': float(np.std(kl_per_example)),
        f'{prefix}tv_mean': float(np.mean(tv_per_example)),
        f'{prefix}tv_std': float(np.std(tv_per_example)),
        f'{prefix}brier_mean': float(np.mean(brier_per_example)),
        f'{prefix}brier_std': float(np.std(brier_per_example)),
        f'{prefix}directional_accuracy': directional_accuracy,
        f'{prefix}n_directional_pairs': n_directional,
    }


def compute_metrics_from_predictions(
    predicted_probs: np.ndarray,
    observed_probs: np.ndarray,
    directional_threshold: float = 0.10,
) -> dict[str, float]:
    """
    Compute KL, TV, Brier, and directional accuracy from predicted and observed
    probability distributions, in both three-way (goal_0, goal_1, no_goal)
    and two-way normalised (goal_0, goal_1) variants.

    Args:
        predicted_probs: (n_examples, 3) predicted [goal_0, goal_1, no_goal]
        observed_probs: (n_examples, 3) observed [goal_0, goal_1, no_goal]
        directional_threshold: Minimum |obs_0 - obs_1| to count for directional accuracy.
            Applied to raw rates for 3-way, normalised rates for 2-way.

    Returns:
        Dict with metric names and values, prefixed '3way_' and '2way_'.
    """
    n = predicted_probs.shape[0]

    # Three-way metrics (full distribution including no_goal)
    metrics = _compute_metrics_for_distribution(
        predicted_probs, observed_probs, directional_threshold, prefix='3way_',
    )

    # Two-way normalised metrics (renormalise to goal_0 + goal_1 = 1)
    eps = 1e-10
    pred_2way_total = predicted_probs[:, 0] + predicted_probs[:, 1]
    obs_2way_total = observed_probs[:, 0] + observed_probs[:, 1]

    pred_2way = np.column_stack([
        predicted_probs[:, 0] / np.maximum(pred_2way_total, eps),
        predicted_probs[:, 1] / np.maximum(pred_2way_total, eps),
    ])
    obs_2way = np.column_stack([
        observed_probs[:, 0] / np.maximum(obs_2way_total, eps),
        observed_probs[:, 1] / np.maximum(obs_2way_total, eps),
    ])

    metrics.update(_compute_metrics_for_distribution(
        pred_2way, obs_2way, directional_threshold, prefix='2way_',
    ))

    metrics['n_examples'] = n
    metrics['directional_threshold'] = directional_threshold

    return metrics


def compute_post_hoc_metrics(
    params: dict[str, jnp.ndarray],
    model_type: ModelType,
    model_kwargs: dict,
    training_examples: list[dict],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    all_features: list[str],
    feature_to_idx: dict[str, int],
    n_features: int,
    include_no_goal_feature: bool = False,
    directional_threshold: float = 0.10,
) -> dict[str, Any]:
    """
    Run post-hoc evaluation of a fitted model: compute per-agent weights,
    predictions, and metrics (KL, TV, Brier, directional accuracy).

    Args:
        params: Saved model parameters (jnp arrays)
        model_type: Model type string
        model_kwargs: Model construction kwargs
        training_examples: List of training examples from prepare_examples()
        agent_training_pipelines: Agent pipeline data from prepare_training_data()
        all_features: Ordered feature name list
        feature_to_idx: Feature name to index mapping
        n_features: Number of features
        include_no_goal_feature: Whether no-goal has a dedicated feature
        directional_threshold: Threshold for directional accuracy

    Returns:
        Dict with 'metrics' (overall), 'per_agent_metrics', and 'agent_weights'.
    """
    log.info(f"Computing per-agent weights for {model_type}...")
    agent_weights = compute_per_agent_weights(
        params, model_type, model_kwargs, all_features, agent_training_pipelines,
    )
    log.info(f"Computed weights for {len(agent_weights)} agents")

    log.info("Computing predictions...")
    predicted_probs, observed_probs = compute_example_predictions(
        params, model_type, model_kwargs, agent_weights,
        training_examples, include_no_goal_feature, n_features,
    )

    log.info("Computing overall metrics...")
    overall_metrics = compute_metrics_from_predictions(
        predicted_probs, observed_probs, directional_threshold,
    )

    # Per-agent metrics
    agent_names = list(set(ex['run_name'] for ex in training_examples))
    per_agent_metrics: dict[str, dict[str, float]] = {}

    for agent_name in agent_names:
        mask = np.array([ex['run_name'] == agent_name for ex in training_examples])
        if np.sum(mask) == 0:
            continue
        agent_metrics = compute_metrics_from_predictions(
            predicted_probs[mask], observed_probs[mask], directional_threshold,
        )
        per_agent_metrics[agent_name] = agent_metrics

    log.info(f"Post-hoc metrics for {model_type}:")
    for variant in ('3way', '2way'):
        log.info(f"  [{variant}]")
        log.info(f"    KL:    {overall_metrics[f'{variant}_kl_mean']:.4f} +/- {overall_metrics[f'{variant}_kl_std']:.4f}")
        log.info(f"    TV:    {overall_metrics[f'{variant}_tv_mean']:.4f} +/- {overall_metrics[f'{variant}_tv_std']:.4f}")
        log.info(f"    Brier: {overall_metrics[f'{variant}_brier_mean']:.4f} +/- {overall_metrics[f'{variant}_brier_std']:.4f}")
        log.info(f"    Dir%:  {overall_metrics[f'{variant}_directional_accuracy']:.4f} "
                 f"({overall_metrics[f'{variant}_n_directional_pairs']} pairs)")

    return {
        'metrics': overall_metrics,
        'per_agent_metrics': per_agent_metrics,
        'agent_weights': agent_weights,
    }
