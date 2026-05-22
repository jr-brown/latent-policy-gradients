import jax
import jax.numpy as jnp
import optax
import logging

from tqdm import tqdm
from typing import Literal
from collections.abc import Callable

from src.preference_modelling.data import examples_to_batch, NoGoalMode
from src.preference_modelling.models import ModelType, get_model
from src.preference_modelling.loss import compute_data_loss_batched, DEFAULT_BATCH_SIZE


log = logging.getLogger(__name__)

EPSILON = 1e-8


def fit_per_agent_per_goal_baseline(
    training_examples: list[dict],
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
    include_no_goal_feature: bool = False,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    batch_size: int = 64,
) -> tuple[dict[str, dict[tuple[float, ...], float]], float]:
    """
    Fit optimal per-agent per-goal values as a baseline on model performance.

    This gives the best possible fit when each agent assigns independent values
    to each unique goal (ignoring feature decomposition), providing a ceiling
    on how well any feature-based model can perform.

    Args:
        training_examples: List of training examples with goal_0_rate and goal_1_rate
        no_goal_mode: How to handle no-goal (neither) choices:
            - "unweighted_ignore_no_goal": Normalize to 2-way distribution, weight=1
            - "weighted_ignore_no_goal": Normalize to 2-way distribution, weight=total_rate
            - "full_distribution": Use raw rates, include no-goal as 3rd option
        include_no_goal_feature: If True (default) and no_goal_mode is "full_distribution",
            include a learned per-agent no-goal value. If False, no-goal value is fixed to 0.
        learning_rate: Learning rate for optimizer
        num_epochs: Number of passes through the dataset
        batch_size: Number of examples per mini-batch

    Returns:
        Tuple of:
            - per_agent_goal_values: Dict mapping agent name to dict of goal_tuple -> value
            - total_loss: Total weighted loss across all agents
    """
    import numpy as np

    is_three_way = (no_goal_mode == "full_distribution")

    # Collect all unique goals and build indices
    goal_to_idx: dict[tuple[float, ...], int] = {}
    for ex in training_examples:
        goal_0_tuple = tuple(ex['goal_0'].tolist())
        goal_1_tuple = tuple(ex['goal_1'].tolist())
        if goal_0_tuple not in goal_to_idx:
            goal_to_idx[goal_0_tuple] = len(goal_to_idx)
        if goal_1_tuple not in goal_to_idx:
            goal_to_idx[goal_1_tuple] = len(goal_to_idx)

    n_goals = len(goal_to_idx)

    # Group examples by agent
    agent_examples: dict[str, list[dict]] = {}
    for ex in training_examples:
        run_name = ex['run_name']
        if run_name not in agent_examples:
            agent_examples[run_name] = []
        agent_examples[run_name].append(ex)

    agent_names = list(agent_examples.keys())
    n_agents = len(agent_names)

    if n_agents == 0:
        return {}, 0.0

    # Create full dataset with agent indices and goal indices
    all_goal_0_indices = []
    all_goal_1_indices = []
    all_observed_probs_0 = []
    all_observed_probs_1 = []
    all_observed_probs_no_goal = []
    all_weights = []
    agent_idx_for_example = []

    for agent_idx, agent_name in enumerate(agent_names):
        for ex in agent_examples[agent_name]:
            goal_0_tuple = tuple(ex['goal_0'].tolist())
            goal_1_tuple = tuple(ex['goal_1'].tolist())
            all_goal_0_indices.append(goal_to_idx[goal_0_tuple])
            all_goal_1_indices.append(goal_to_idx[goal_1_tuple])

            # Compute probabilities and weights based on no_goal_mode
            goal_0_rate = ex['goal_0_rate']
            goal_1_rate = ex['goal_1_rate']
            total_rate = goal_0_rate + goal_1_rate

            if is_three_way:
                # Use raw rates directly
                all_observed_probs_0.append(goal_0_rate)
                all_observed_probs_1.append(goal_1_rate)
                all_observed_probs_no_goal.append(1.0 - total_rate)
                all_weights.append(1.0)
            else:
                # Normalize to 2-way distribution
                if total_rate > 0:
                    all_observed_probs_0.append(goal_0_rate / total_rate)
                    all_observed_probs_1.append(goal_1_rate / total_rate)
                else:
                    all_observed_probs_0.append(0.5)
                    all_observed_probs_1.append(0.5)
                all_observed_probs_no_goal.append(0.0)
                all_weights.append(total_rate if no_goal_mode == "weighted_ignore_no_goal" else 1.0)

            agent_idx_for_example.append(agent_idx)

    goal_0_indices = jnp.array(all_goal_0_indices)
    goal_1_indices = jnp.array(all_goal_1_indices)
    observed_probs_0 = jnp.array(all_observed_probs_0)
    observed_probs_1 = jnp.array(all_observed_probs_1)
    observed_probs_no_goal = jnp.array(all_observed_probs_no_goal)
    sample_weights = jnp.array(all_weights)
    agent_indices = jnp.array(agent_idx_for_example)
    dataset_size = len(all_observed_probs_0)

    # Determine whether to use learned no-goal value
    learn_no_goal_value = is_three_way and include_no_goal_feature

    def batch_loss_fn(
        params: dict[str, jnp.ndarray],
        batch_goal_0_idx: jnp.ndarray,
        batch_goal_1_idx: jnp.ndarray,
        batch_observed_probs_0: jnp.ndarray,
        batch_observed_probs_1: jnp.ndarray,
        batch_observed_probs_no_goal: jnp.ndarray,
        batch_weights: jnp.ndarray,
        batch_agent_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute weighted loss for a mini-batch."""
        all_goal_values = params['all_goal_values']  # Shape: (n_agents, n_goals)

        # Gather goal values for each example
        values_0 = all_goal_values[batch_agent_indices, batch_goal_0_idx]
        values_1 = all_goal_values[batch_agent_indices, batch_goal_1_idx]

        if is_three_way:
            # Three-way distribution including no-goal
            if learn_no_goal_value:
                # Learned per-agent no-goal value
                no_goal_value = params['no_goal_value']  # Shape: (n_agents,)
                values_no_goal = no_goal_value[batch_agent_indices]
            else:
                # Fixed zero no-goal value
                values_no_goal = jnp.zeros_like(values_0)
            logits = jnp.stack([values_0, values_1, values_no_goal], axis=1)
            predicted_probs = jax.nn.softmax(logits, axis=1)
            observed_probs = jnp.stack([batch_observed_probs_0, batch_observed_probs_1, batch_observed_probs_no_goal], axis=1)
        else:
            # Two-way distribution
            logits = jnp.stack([values_0, values_1], axis=1)
            predicted_probs = jax.nn.softmax(logits, axis=1)
            observed_probs = jnp.stack([batch_observed_probs_0, batch_observed_probs_1], axis=1)

        # Compute KL divergence per example
        kl_divs = jnp.sum(
            observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON)),
            axis=1
        )

        # Compute weighted average loss
        total_weight = jnp.sum(batch_weights)
        return jnp.sum(batch_weights * kl_divs) / jnp.maximum(total_weight, EPSILON)

    # Initialize parameters
    params = {'all_goal_values': jnp.zeros((n_agents, n_goals))}
    if learn_no_goal_value:
        params['no_goal_value'] = jnp.zeros(n_agents)  # Per-agent no-goal value

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state, batch_g0, batch_g1, batch_obs0, batch_obs1, batch_obs_ng, batch_w, batch_agent):
        loss, grads = jax.value_and_grad(batch_loss_fn)(
            params, batch_g0, batch_g1, batch_obs0, batch_obs1, batch_obs_ng, batch_w, batch_agent
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Initialize RNG for shuffling
    rng = np.random.RandomState(0)

    for _ in range(num_epochs):
        shuffled_indices = rng.permutation(dataset_size)

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_idx = jnp.array(shuffled_indices[start_idx:end_idx])

            params, opt_state, _ = update_step(
                params, opt_state,
                goal_0_indices[batch_idx],
                goal_1_indices[batch_idx],
                observed_probs_0[batch_idx],
                observed_probs_1[batch_idx],
                observed_probs_no_goal[batch_idx],
                sample_weights[batch_idx],
                agent_indices[batch_idx],
            )

    # Extract per-agent goal values into dictionary
    idx_to_goal = {v: k for k, v in goal_to_idx.items()}
    per_agent_goal_values = {}
    for agent_idx, agent_name in enumerate(agent_names):
        per_agent_goal_values[agent_name] = {
            idx_to_goal[goal_idx]: float(params['all_goal_values'][agent_idx, goal_idx])
            for goal_idx in range(n_goals)
        }
        if learn_no_goal_value:
            # Include learned no-goal value in the output
            per_agent_goal_values[agent_name][('no_goal',)] = float(params['no_goal_value'][agent_idx])

    # Compute final loss on full dataset
    final_loss = float(batch_loss_fn(
        params, goal_0_indices, goal_1_indices, observed_probs_0, observed_probs_1,
        observed_probs_no_goal, sample_weights, agent_indices
    ))

    return per_agent_goal_values, final_loss


def fit_per_agent_per_feature_baseline(
    training_examples: list[dict],
    n_features: int,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
    include_no_goal_feature: bool = False,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    batch_size: int = 64,
) -> tuple[dict[str, jnp.ndarray], float]:
    """
    Fit optimal per-agent per-feature values as a baseline on model performance.

    This gives the best possible fit when each agent has completely independent
    feature weights, providing a ceiling on how well any constrained model can perform.

    Note: For full_distribution mode, the no_goal weight is handled based on
    include_no_goal_feature: if True, the last feature weight is the no_goal weight;
    if False, no_goal value is computed as dot(weights, zeros) = 0.

    Args:
        training_examples: List of training examples with goal_0_rate and goal_1_rate
        n_features: Total number of features
        no_goal_mode: How to handle no-goal (neither) choices:
            - "unweighted_ignore_no_goal": Normalize to 2-way distribution, weight=1
            - "weighted_ignore_no_goal": Normalize to 2-way distribution, weight=total_rate
            - "full_distribution": Use raw rates, include no-goal as 3rd option
        include_no_goal_feature: If True (default), no-goal features are [0,...,0,1].
            If False, no-goal features are zeros (no-goal value = 0).
        learning_rate: Learning rate for optimizer
        num_epochs: Number of passes through the dataset
        batch_size: Number of examples per mini-batch

    Returns:
        Tuple of:
            - per_agent_weights: Dict mapping agent name to optimal feature weights
            - total_loss: Total weighted loss across all agents
    """
    import numpy as np
    from src.preference_modelling.data import get_no_goal_features

    is_three_way = (no_goal_mode == "full_distribution")

    # Group examples by agent
    agent_examples: dict[str, list[dict]] = {}
    for ex in training_examples:
        run_name = ex['run_name']
        if run_name not in agent_examples:
            agent_examples[run_name] = []
        agent_examples[run_name].append(ex)

    agent_names = list(agent_examples.keys())
    n_agents = len(agent_names)

    if n_agents == 0:
        return {}, 0.0

    def create_simple_batch(examples: list[dict]) -> dict[str, jnp.ndarray]:
        """Create a simple batch without pipeline objects for baseline fitting."""
        observed_probs_0 = []
        observed_probs_1 = []
        observed_probs_no_goal = []
        weights = []

        for ex in examples:
            goal_0_rate = ex['goal_0_rate']
            goal_1_rate = ex['goal_1_rate']
            total_rate = goal_0_rate + goal_1_rate

            if is_three_way:
                observed_probs_0.append(goal_0_rate)
                observed_probs_1.append(goal_1_rate)
                observed_probs_no_goal.append(1.0 - total_rate)
                weights.append(1.0)
            else:
                if total_rate > 0:
                    observed_probs_0.append(goal_0_rate / total_rate)
                    observed_probs_1.append(goal_1_rate / total_rate)
                else:
                    observed_probs_0.append(0.5)
                    observed_probs_1.append(0.5)
                observed_probs_no_goal.append(0.0)
                weights.append(total_rate if no_goal_mode == "weighted_ignore_no_goal" else 1.0)

        return {
            'goals_0': jnp.stack([ex['goal_0'] for ex in examples]),
            'goals_1': jnp.stack([ex['goal_1'] for ex in examples]),
            'observed_probs_0': jnp.array(observed_probs_0),
            'observed_probs_1': jnp.array(observed_probs_1),
            'observed_probs_no_goal': jnp.array(observed_probs_no_goal),
            'weights': jnp.array(weights),
        }

    # Create full dataset with agent indices
    all_examples = []
    agent_idx_for_example = []
    for agent_idx, agent_name in enumerate(agent_names):
        for ex in agent_examples[agent_name]:
            all_examples.append(ex)
            agent_idx_for_example.append(agent_idx)

    full_batch = create_simple_batch(all_examples)
    agent_indices = jnp.array(agent_idx_for_example)
    dataset_size = len(all_examples)

    # Create no-goal feature vector for three-way loss
    no_goal_features = get_no_goal_features(n_features, include_no_goal_feature) if is_three_way else None

    def batch_loss_fn(params: dict[str, jnp.ndarray], batch_data: dict[str, jnp.ndarray], batch_agent_indices: jnp.ndarray) -> jnp.ndarray:
        """Compute weighted loss for a mini-batch."""
        all_weights = params['all_weights']  # Shape: (n_agents, n_features)

        goals_0 = batch_data['goals_0']
        goals_1 = batch_data['goals_1']
        observed_probs_0 = batch_data['observed_probs_0']
        observed_probs_1 = batch_data['observed_probs_1']
        observed_probs_no_goal = batch_data['observed_probs_no_goal']
        sample_weights = batch_data['weights']

        # Gather weights for each example based on agent index
        example_weights = all_weights[batch_agent_indices]  # Shape: (batch_size, n_features)

        # Compute goal values: (batch_size,)
        values_0 = jnp.sum(goals_0 * example_weights, axis=1)
        values_1 = jnp.sum(goals_1 * example_weights, axis=1)

        if is_three_way:
            # Compute no-goal value using the no_goal feature (last dimension)
            values_no_goal = jnp.sum(no_goal_features * example_weights, axis=1)
            logits = jnp.stack([values_0, values_1, values_no_goal], axis=1)
            predicted_probs = jax.nn.softmax(logits, axis=1)
            observed_probs = jnp.stack([observed_probs_0, observed_probs_1, observed_probs_no_goal], axis=1)
        else:
            logits = jnp.stack([values_0, values_1], axis=1)
            predicted_probs = jax.nn.softmax(logits, axis=1)
            observed_probs = jnp.stack([observed_probs_0, observed_probs_1], axis=1)

        # Compute KL divergence per example
        kl_divs = jnp.sum(
            observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON)),
            axis=1
        )

        # Compute weighted average loss
        total_weight = jnp.sum(sample_weights)
        return jnp.sum(sample_weights * kl_divs) / jnp.maximum(total_weight, EPSILON)

    # Initialize parameters: stacked per-agent weights only
    params = {
        'all_weights': jnp.zeros((n_agents, n_features)),
    }

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state, batch_data, batch_agent_indices):
        loss, grads = jax.value_and_grad(batch_loss_fn)(params, batch_data, batch_agent_indices)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def get_batch(indices: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """Extract a mini-batch from dataset using given indices."""
        batch_data = {k: v[indices] for k, v in full_batch.items()}
        batch_agent_idx = agent_indices[indices]
        return batch_data, batch_agent_idx

    # Initialize RNG for shuffling
    rng = np.random.RandomState(0)

    for _ in range(num_epochs):
        # Shuffle indices at the start of each epoch
        shuffled_indices = rng.permutation(dataset_size)

        # Iterate over mini-batches
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = jnp.array(shuffled_indices[start_idx:end_idx])
            batch_data, batch_agent_idx = get_batch(batch_indices)

            params, opt_state, loss = update_step(params, opt_state, batch_data, batch_agent_idx)

    # Extract per-agent weights into dictionary
    per_agent_weights = {
        agent_name: params['all_weights'][i]
        for i, agent_name in enumerate(agent_names)
    }

    # Compute final loss on full dataset
    final_loss = float(batch_loss_fn(params, full_batch, agent_indices))

    return per_agent_weights, final_loss


def get_random_choice_baseline(
    training_examples: list[dict],
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> float:
    """
    Compute random choice baseline assuming zero weights.

    When all feature weights are zero, the model predicts equal probability
    for all choices. This gives the worst reasonable baseline - any trained model
    should do at least this well.

    Args:
        training_examples: List of training examples with goal_0_rate and goal_1_rate
        no_goal_mode: How to handle no-goal (neither) choices:
            - "unweighted_ignore_no_goal": 2-way uniform (0.5, 0.5), weight=1
            - "weighted_ignore_no_goal": 2-way uniform (0.5, 0.5), weight=total_rate
            - "full_distribution": 3-way uniform (1/3, 1/3, 1/3)

    Returns:
        Total weighted KL divergence loss with uniform predictions
    """
    if len(training_examples) == 0:
        return 0.0

    is_three_way = (no_goal_mode == "full_distribution")

    # Compute observed probabilities and weights based on no_goal_mode
    observed_probs_0 = []
    observed_probs_1 = []
    observed_probs_no_goal = []
    weights = []

    for ex in training_examples:
        goal_0_rate = ex['goal_0_rate']
        goal_1_rate = ex['goal_1_rate']
        total_rate = goal_0_rate + goal_1_rate

        if is_three_way:
            observed_probs_0.append(goal_0_rate)
            observed_probs_1.append(goal_1_rate)
            observed_probs_no_goal.append(1.0 - total_rate)
            weights.append(1.0)
        else:
            if total_rate > 0:
                observed_probs_0.append(goal_0_rate / total_rate)
                observed_probs_1.append(goal_1_rate / total_rate)
            else:
                observed_probs_0.append(0.5)
                observed_probs_1.append(0.5)
            observed_probs_no_goal.append(0.0)
            weights.append(total_rate if no_goal_mode == "weighted_ignore_no_goal" else 1.0)

    observed_probs_0 = jnp.array(observed_probs_0)
    observed_probs_1 = jnp.array(observed_probs_1)
    observed_probs_no_goal = jnp.array(observed_probs_no_goal)
    weights = jnp.array(weights)

    if is_three_way:
        # With zero weights and no_goal feature, predicted probability is 1/3 for all choices
        predicted_prob = 1.0 / 3.0
        observed_probs = jnp.stack([observed_probs_0, observed_probs_1, observed_probs_no_goal], axis=1)
    else:
        # With zero weights, predicted probability is 0.5 for both choices
        predicted_prob = 0.5
        observed_probs = jnp.stack([observed_probs_0, observed_probs_1], axis=1)

    predicted_probs = jnp.full_like(observed_probs, predicted_prob)

    # Compute KL divergence: sum_i p_i * log(p_i / q_i)
    # where p is observed distribution and q is predicted (uniform)
    kl_divs = jnp.sum(
        observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON)),
        axis=1
    )

    # Compute weighted average loss
    total_weight = jnp.sum(weights)
    return float(jnp.sum(weights * kl_divs) / jnp.maximum(total_weight, EPSILON))


def train_preference_model(
    train_dataset: dict[str, jnp.ndarray | list],
    n_features: int,
    loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
    log_interval: int = 100,
    rng_seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """
    Train a preference model on the given dataset.
    
    Args:
        train_dataset: Dataset of training data (all examples)
        n_features: Total number of features
        loss_fn: Loss function to optimize
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for Adam optimizer
        num_epochs: Number of passes through the dataset
        batch_size: Number of examples per mini-batch
        show_progress: Whether to show progress bar
        log_interval: How often to log progress (in batches)
        rng_seed: Random seed for shuffling
    
    Returns:
        Trained model parameters dict with keys such as 'log_beta', 'log_theta', and 'log_q' (for kl/multi_kl models)
    """
    model_kwargs = model_kwargs or {}
    model = get_model(model_type, **model_kwargs)
    
    # Initialize parameters from model's hyperparameter spec
    params = model.init_params_from_spec(n_features)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Extract array-only dataset for JIT compatibility
    array_dataset = {k: v for k, v in train_dataset.items() if isinstance(v, jnp.ndarray)}
    dataset_size = array_dataset['weights'].shape[0]
    
    # Initialize RNG for shuffling
    rng = jax.random.PRNGKey(rng_seed)
    
    @jax.jit
    def update_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    def get_batch(dataset: dict[str, jnp.ndarray], indices: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Extract a mini-batch from dataset using given indices."""
        return {k: v[indices] if v.ndim > 0 else v for k, v in dataset.items()}
    
    global_batch_count = 0
    running_loss = 0.0
    running_weight = 0.0
    
    for epoch in tqdm(range(num_epochs), desc="Training", leave=show_progress):
        # Shuffle indices at the start of each epoch
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_rng, dataset_size)
        
        # Iterate over mini-batches
        for start_idx in tqdm(range(0, dataset_size, batch_size), desc="Epoch Batches", leave=False):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = shuffled_indices[start_idx:end_idx]
            batch = get_batch(array_dataset, batch_indices)
            
            params, opt_state, loss = update_step(params, opt_state, batch)
            
            # Accumulate for logging
            batch_weight = float(jnp.sum(batch['weights']))
            running_loss += float(loss) * batch_weight
            running_weight += batch_weight
            global_batch_count += 1
            
            if show_progress and global_batch_count % log_interval == 0:
                avg_loss = running_loss / max(running_weight, 1e-8)
                log.info(f"Batch {global_batch_count}, Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
                running_loss = 0.0
                running_weight = 0.0
    
    return params


def train_and_evaluate_model(
    train_examples: list[dict],
    val_examples: list[dict],
    n_features: int,
    loss_fn: Callable,
    data_loss_fn: Callable,
    model_type: ModelType = "rw",
    model_kwargs: dict | None = None,
    learning_rate: float = 0.03,
    num_epochs: int = 100,
    batch_size: int = DEFAULT_BATCH_SIZE,
    no_goal_mode: NoGoalMode = "unweighted_ignore_no_goal",
) -> tuple[dict[str, jnp.ndarray], dict[str, float | int]]:
    """
    Train a model and evaluate on validation set.

    Args:
        train_examples: Training examples
        val_examples: Validation examples
        n_features: Number of features
        loss_fn: Loss function with regularisation (for training)
        data_loss_fn: Data-only loss function (for evaluation)
        model_type: Which model formulation to use ("rw", "kl", or "multi_kl")
        model_kwargs: Optional kwargs for model construction
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Number of examples per mini-batch

    Returns:
        Tuple of (trained params, results dict with validation metrics)
    """
    train_dataset = examples_to_batch(train_examples, no_goal_mode=no_goal_mode)
    val_batch = examples_to_batch(val_examples, no_goal_mode=no_goal_mode)

    params = train_preference_model(
        train_dataset=train_dataset,
        n_features=n_features,
        loss_fn=loss_fn,
        model_type=model_type,
        model_kwargs=model_kwargs,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Use data-only loss (no regularisation) for validation, computed in mini-batches
    val_loss = compute_data_loss_batched(data_loss_fn, params, val_batch, batch_size=batch_size)

    results = {
        'val_loss': val_loss,
        'n_train_examples': len(train_examples),
        'n_val_examples': len(val_examples),
    }

    return params, results
