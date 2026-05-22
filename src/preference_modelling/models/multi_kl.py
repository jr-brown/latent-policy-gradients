"""
Multi-choice KL-divergence models and ablations.
"""
import jax
import jax.numpy as jnp
import logging

from typing import Any, Literal

from src.preference_modelling.data_structures import (
    TrainingPipeline,
    PaddedPipeline,
)

from .base import (
    EPSILON,
    Int,
    SimpleWeightsModel,
    NonLinearityMixin,
    PipelineModeMixin,
    HyperparameterSpec,
    ScalarSpec,
    CustomSpec,
    Q_SPEC,
    INIT_VALUE_SPEC,
    _compute_agent_params_from_pipeline_batched,
    _get_param_key,
    get_no_goal_features,
)
from .saliency import (
    ValueFunction,
    LinearValueFunction,
    make_value_function,
)


log = logging.getLogger(__name__)


def _expand_features_quadratic(features: jnp.ndarray) -> jnp.ndarray:
    """
    Expand feature vector(s) with quadratic terms.

    For original feature vector phi in R^n, expand to:
    phi_expanded = [phi, flatten(phi outer phi)] in R^(n + n^2)

    Args:
        features: shape (..., n) - original features

    Returns:
        shape (..., n + n^2) - expanded features [phi, flatten(phi outer phi)]
    """
    # Get original shape
    *batch_dims, n = features.shape

    # Compute outer product and flatten
    # For shape (..., n): outer product gives (..., n, n), flatten to (..., n^2)
    outer = jnp.einsum('...i,...j->...ij', features, features)
    outer_flat = outer.reshape(*batch_dims, n * n)

    # Concatenate: [phi, phi outer phi_flat]
    return jnp.concatenate([features, outer_flat], axis=-1)


class MultiChoiceKLModel(PipelineModeMixin, NonLinearityMixin, SimpleWeightsModel):
    """
    Multi-choice KL model with configurable value computation.

    Computes goal values via a unified ValueFunction interface:
        logit_i = value_function(w, phi^i)

    Value function types:
    - LinearValueFunction (default): value = (S @ w) . phi
    - MLPValueFunction: value = MLP(concat(w, phi))
    - BilinearValueFunction: value = sigma(B @ phi)^T A sigma(C @ w)

    The model supports configurable KL direction:
    - Forward KL (default): KL(target || policy)
    - Reverse KL: KL(policy || target)

    This model always requires numerical integration.
    """

    def __init__(
        self,
        reverse_kl: bool = True,
        value_function: ValueFunction | dict | None = None,
        pipeline_mode: Literal["sequential", "simultaneous", "memoryless"] = "sequential",
        non_linearity: str = "none",
        integration_kwargs: dict | None = None,
        include_no_goal_feature: bool = False,
    ):
        """
        Args:
            reverse_kl: If True, use KL(policy || target) instead of KL(target || policy)
            value_function: ValueFunction instance or config dict. Computes goal values
                from agent weights and goal features. If None, defaults to LinearValueFunction
                with full structure (learnable S matrix initialized to identity).
                Examples:
                - None: Linear with learnable S matrix
                - {"type": "linear", "structure": "diagonal"}: Diagonal saliency
                - {"type": "mlp", "hidden_sizes": [32]}: MLP value function
                - {"type": "bilinear", "rank": 16}: Bilinear value function
            pipeline_mode: How to process training pipeline stages:
                - "sequential": Process stages in order (default)
                - "simultaneous": Flatten all stages into one (no temporal ordering)
                - "memoryless": Use only the final stage (ignores history)
            non_linearity: Non-linearity to apply to weights ("none", "quadratic", "sigmoid", "power_law")
            integration_kwargs: Configuration for numerical integration
            include_no_goal_feature: If True (default), uses [0,...,0,1] for no-goal features.
                If False, uses zeros (original behavior).
        """
        # Create value function - always set, never None
        if value_function is None:
            self.value_function = LinearValueFunction()
        elif isinstance(value_function, dict):
            self.value_function = make_value_function(value_function)
        else:
            self.value_function = value_function

        # Cache latent dimension to avoid method calls in JIT context
        self._cached_latent_dimension = getattr(self.value_function, 'latent_dimension', None)

        self.reverse_kl = reverse_kl
        self.pipeline_mode = pipeline_mode
        super().__init__(
            non_linearity=non_linearity,
            integration_kwargs=integration_kwargs,
            include_no_goal_feature=include_no_goal_feature,
        )

    @property
    def name(self) -> str:
        base_name = "Multi-Choice KL"
        if self.reverse_kl:
            base_name += " (reverse)"
        if self.pipeline_mode != "sequential":
            base_name += f" [{self.pipeline_mode}]"
        # Only show value function name if it's not the default (full structure)
        if not isinstance(self.value_function, LinearValueFunction) or self.value_function.structure != "full":
            base_name += f", {self.value_function.name}"
        return self._format_name_with_non_linearity(base_name)

    def get_hyperparameter_specs(self) -> list[HyperparameterSpec]:
        """Return specs for all hyperparameters."""
        return [
            Q_SPEC,
            INIT_VALUE_SPEC,
            *self._get_non_linearity_specs(),
            *self.value_function.get_specs(),
        ]

    def init_params_from_spec(self, n_features: int) -> dict[str, jnp.ndarray]:
        """
        Initialize hyperparameters from the specification.

        Delegates value function parameter initialization to the value function itself.
        """
        params = {}

        # Initialize scalar parameters from specs
        for spec in self.get_hyperparameter_specs():
            if isinstance(spec, ScalarSpec):
                params.update(spec.init_params(n_features))

        # Delegate value function parameter initialization
        params.update(self.value_function.init_params(n_features))

        return params

    def compute_regularisation_loss(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Delegate regularisation to value_function."""
        return self.value_function.compute_regularisation_loss(params)

    def _get_initial_weights(
        self,
        n_features: int,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Get initial weights for pipeline computation.

        Uses cached latent dimension to avoid method calls in JIT context.
        """
        dim = self._cached_latent_dimension if self._cached_latent_dimension is not None else n_features
        if self._requires_nonzero_init():
            init_std = jnp.exp(params.get('log_init_std', jnp.array(-3.0)))
            return init_std * jnp.ones(dim)
        init_value = params.get('init_value', jnp.array(0.0))
        return init_value * jnp.ones(dim)

    def _compute_env_loss(
        self,
        params: dict[str, jnp.ndarray],
        w: jnp.ndarray,
        env_goals: jnp.ndarray,
        env_distractors: jnp.ndarray,
        env_goal_mask: jnp.ndarray,
        env_distractor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute KL loss for a single environment.

        Args:
            params: Model parameters
            w: Current weights, shape (latent_dim,) or (n_features,)
            env_goals: Goal features, shape (max_goals, n_features)
            env_distractors: Distractor features, shape (max_distractors, n_features)
            env_goal_mask: Goal validity mask, shape (max_goals,)
            env_distractor_mask: Distractor validity mask, shape (max_distractors,)

        Returns:
            Scalar KL loss
        """
        q = jax.nn.sigmoid(params['log_q'])

        # Apply non-linearity ONCE (not per-goal)
        transformed_w = self._apply_non_linearity(w, params)

        # Compute logits for goals and distractors using batched forward
        # Wrap single weight vector for new batched interface, unwrap result
        goal_logits = self.value_function.forward_batched(transformed_w[None, :], env_goals, params)[0]
        distractor_logits = self.value_function.forward_batched(transformed_w[None, :], env_distractors, params)[0]

        # Compute no-goal value using appropriate no-goal feature vector
        n_features = env_goals.shape[1]
        no_goal_features = get_no_goal_features(n_features, self.include_no_goal_feature)[None, :]
        no_goal_logit = self.value_function.forward_batched(transformed_w[None, :], no_goal_features, params)[0, 0]

        # Mask invalid entries with large negative values
        goal_logits = jnp.where(env_goal_mask > 0, goal_logits, -1e10)
        distractor_logits = jnp.where(env_distractor_mask > 0, distractor_logits, -1e10)

        # Compute partition function (including null option with computed no-goal value)
        Z = (jnp.sum(jnp.exp(goal_logits) * env_goal_mask) +
             jnp.sum(jnp.exp(distractor_logits) * env_distractor_mask) +
             jnp.exp(no_goal_logit))

        # Total goal probability
        pi_G = jnp.sum(jnp.exp(goal_logits) * env_goal_mask) / Z
        pi_G = jnp.clip(pi_G, EPSILON, 1.0 - EPSILON)

        # KL divergence (direction depends on reverse_kl flag)
        if self.reverse_kl:
            # KL(policy || target)
            return pi_G * jnp.log(pi_G / q) + (1 - pi_G) * jnp.log((1 - pi_G) / (1 - q))
        else:
            # KL(target || policy)
            return q * jnp.log(q / pi_G) + (1 - q) * jnp.log((1 - q) / (1 - pi_G))

    def _compute_stage_gradient_batched(
        self,
        params: dict[str, jnp.ndarray],
        w: jnp.ndarray,
        stage_goals: jnp.ndarray,
        stage_distractors: jnp.ndarray,
        stage_env_mask: jnp.ndarray,
        stage_goal_mask: jnp.ndarray,
        stage_distractor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute gradient using autodiff over environments.

        Unlike MultiChoiceKLModel, this uses raw gradients without saliency transformation.
        """
        def single_env_gradient(env_idx: Int) -> jnp.ndarray:
            # Gradient of loss w.r.t. weights for this environment
            grad_w = jax.grad(lambda w_: self._compute_env_loss(
                params, w_,
                stage_goals[env_idx],
                stage_distractors[env_idx],
                stage_goal_mask[env_idx],
                stage_distractor_mask[env_idx],
            ))(w)
            # Apply environment mask only (no saliency transformation)
            return stage_env_mask[env_idx] * grad_w

        # Compute gradients for all environments using vmap
        n_envs = stage_goals.shape[0]
        env_gradients = jax.vmap(single_env_gradient)(jnp.arange(n_envs))
        total_gradient = jnp.sum(env_gradients, axis=0)

        # Average over valid environments, negative for descent
        n_valid_envs = jnp.maximum(jnp.sum(stage_env_mask), 1.0)
        return -total_gradient / n_valid_envs

    def single_example_loss(
        self,
        params: dict[str, jnp.ndarray],
        padded: PaddedPipeline,
        goal_0: jnp.ndarray,
        goal_1: jnp.ndarray,
        observed_prob_0: jnp.ndarray,
        observed_prob_1: jnp.ndarray,
        observed_prob_no_goal: jnp.ndarray,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss for choice probabilities via value_function."""
        n_features = goal_0.shape[0]

        # Apply pipeline mode transformation
        modified_padded = self._apply_pipeline_mode(padded)

        # Compute agent weights through pipeline
        agent_weights = _compute_agent_params_from_pipeline_batched(
            initial_params=self._get_initial_weights(n_features, params),
            compute_agent_params_from_stage_equilibrium_batched_fn=self._get_compute_stage_equilibrium_batched_fn(params),
            padded=modified_padded,
        )

        # Apply non-linearity ONCE
        transformed_weights = self._apply_non_linearity(agent_weights, params)

        # Compute goal values using batched forward
        # Wrap single weight vector for new batched interface, unwrap result
        goals_batch = jnp.stack([goal_0, goal_1])  # (2, n_features)
        goal_logits = self.value_function.forward_batched(transformed_weights[None, :], goals_batch, params)[0]

        # Use 3-way loss (including no-goal) when observed_prob_no_goal >= 0
        # (full_distribution mode), 2-way loss when sentinel value < 0
        # (unweighted modes set no_goal to -1.0 sentinel)
        def two_way_loss():
            predicted_probs = jax.nn.softmax(goal_logits)
            observed_probs = jnp.array([observed_prob_0, observed_prob_1])
            return jnp.sum(
                observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON))
            )

        def three_way_loss():
            # Compute no-goal value using appropriate no-goal feature vector
            no_goal_features = get_no_goal_features(n_features, self.include_no_goal_feature)[None, :]
            no_goal_logit = self.value_function.forward_batched(transformed_weights[None, :], no_goal_features, params)[0, 0]
            logits = jnp.concatenate([goal_logits, no_goal_logit[None]])
            predicted_probs = jax.nn.softmax(logits)
            observed_probs = jnp.array([observed_prob_0, observed_prob_1, observed_prob_no_goal])
            return jnp.sum(
                observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON))
            )

        kl_div = jax.lax.cond(observed_prob_no_goal >= 0, three_way_loss, two_way_loss)

        return weight * kl_div

    def learn_agent_parameters_to_be_saved(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    ) -> dict[str, Any]:
        """Return agent parameters with pipeline mode applied."""
        agent_parameters = {}
        n_features = len(all_features)

        for agent_name, (_, padded) in agent_training_pipelines.items():
            # Apply pipeline mode transformation (using static version for save)
            modified_padded = self._apply_pipeline_mode_static(padded)

            agent_params = _compute_agent_params_from_pipeline_batched(
                initial_params=self._get_initial_weights(n_features, params),
                compute_agent_params_from_stage_equilibrium_batched_fn=self._get_compute_stage_equilibrium_batched_fn(params),
                padded=modified_padded,
            )
            # Determine labels for weights (features or latent dimensions)
            latent_dim = self._cached_latent_dimension if self._cached_latent_dimension is not None else n_features
            if latent_dim == n_features:
                labels = all_features
            else:
                labels = [f"z{i}" for i in range(latent_dim)]

            agent_weights = {
                label: float(value)
                for label, value in zip(labels, agent_params)
                if abs(float(value)) > EPSILON
            }
            agent_parameters[agent_name] = agent_weights

        return agent_parameters


# =============================================================================
# Diagonal Quadratic Multi-Choice KL Model
# =============================================================================

def _make_s_diag_spec(pair_saliency_init: float) -> CustomSpec:
    """Create a CustomSpec for diagonal saliency vector with quadratic feature expansion."""

    def init_fn(n_features: int) -> dict[str, jnp.ndarray]:
        # Base features (first n) = 1.0
        # Feature-pairs (next n^2) = pair_saliency_init (must be non-zero for gradient flow)
        base_saliencies = jnp.ones(n_features)
        pair_saliencies = pair_saliency_init * jnp.ones(n_features * n_features)
        return {'S_diag': jnp.concatenate([base_saliencies, pair_saliencies])}

    def log_fn(params: dict, all_features: list[str], features_in_training: set[str]) -> None:
        from tabulate import tabulate
        S_diag = params['S_diag']
        n_features = len(all_features)

        # Log base feature saliencies (first n entries)
        base_rows = []
        for i, feature in enumerate(all_features):
            val = float(S_diag[i])
            base_rows.append([feature, f"{val:.4f}"])

        log.info(f"Base Feature Saliencies:\n{tabulate(base_rows, headers=['Feature', 'Saliency'], tablefmt='simple_outline')}")

        # Log feature-pair saliencies as n x n matrix (entries n to n+n^2)
        # The quadratic terms are stored in row-major order: [f0*f0, f0*f1, ..., f0*fn, f1*f0, ...]
        matrix_rows = []
        for i in range(n_features):
            row_label = all_features[i]
            row_data = [row_label]
            for j in range(n_features):
                idx = n_features + i * n_features + j  # Offset by n for base features
                val = float(S_diag[idx])
                if i == j:
                    row_data.append(f"[{val:.3f}]")  # Highlight diagonal (squared terms)
                else:
                    row_data.append(f"{val:.3f}")
            matrix_rows.append(row_data)

        headers = ["phi_i*phi_j"] + all_features
        log.info(f"Feature-Pair Saliencies ([diagonal] = squared terms):\n{tabulate(matrix_rows, headers=headers, tablefmt='simple_outline', stralign='right')}")

    return CustomSpec(name='S_diag', init_fn=init_fn, log_fn=log_fn)


class DiagonalQuadraticMultiChoiceKLModel(PipelineModeMixin, NonLinearityMixin, SimpleWeightsModel):
    """
    Efficient diagonal-only quadratic feature expansion model.

    Like QuadraticNativeSaliencyMultiChoiceKLModel, expands features from n to n + n^2,
    but restricts saliency to a diagonal vector instead of a full matrix:

    effective_w = S_diag * f(w)  (element-wise, not matmul)

    This is O(n + n^2) per step instead of O((n + n^2)^2), making it ~100x faster
    for n=10 features.

    Use this when you want quadratic feature interactions but only need
    per-feature saliency scaling (no cross-feature saliency).
    """

    def __init__(
        self,
        reverse_kl: bool = True,
        pipeline_mode: Literal["sequential", "simultaneous", "memoryless"] = "sequential",
        non_linearity: str = "none",
        pair_saliency_init: float = 0.01,
        integration_kwargs: dict | None = None,
        include_no_goal_feature: bool = False,
    ):
        """
        Args:
            reverse_kl: If True, use KL(policy || target) instead of KL(target || policy)
            pipeline_mode: How to process training pipeline stages
            non_linearity: Non-linearity to apply to weights
            pair_saliency_init: Initial saliency value for feature-pair terms (default 0.01).
                Must be non-zero to allow gradient flow during training.
            integration_kwargs: Configuration for numerical integration
            include_no_goal_feature: If True (default), uses [0,...,0,1] for no-goal features.
                If False, uses zeros (original behavior).
        """
        self.reverse_kl = reverse_kl
        self.pipeline_mode = pipeline_mode
        self.pair_saliency_init = pair_saliency_init

        # Create the S_diag spec
        self._s_diag_spec = _make_s_diag_spec(pair_saliency_init)

        super().__init__(
            non_linearity=non_linearity,
            integration_kwargs=integration_kwargs,
            include_no_goal_feature=include_no_goal_feature,
        )

    @property
    def name(self) -> str:
        base_name = "Diagonal Quadratic Native Saliency Multi-Choice KL"
        if self.reverse_kl:
            base_name += " (reverse)"
        if self.pipeline_mode != "sequential":
            base_name += f" [{self.pipeline_mode}]"
        return self._format_name_with_non_linearity(base_name)

    def _expanded_n_features(self, n_base: int) -> int:
        """Compute expanded feature dimension: n + n^2."""
        return n_base + n_base * n_base

    def get_hyperparameter_specs(self) -> list[HyperparameterSpec]:
        """Return specs for all hyperparameters."""
        return [
            Q_SPEC,
            INIT_VALUE_SPEC,
            self._s_diag_spec,
            *self._get_non_linearity_specs(),
        ]

    def init_params_from_spec(self, n_features: int) -> dict[str, jnp.ndarray]:
        """Initialize with expanded feature dimension (diagonal only)."""
        params = {}
        for spec in self.get_hyperparameter_specs():
            params.update(spec.init_params(n_features))
        return params

    def compute_regularisation_loss(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """No regularisation for this model."""
        return jnp.array(0.0)

    def _get_initial_weights(self, n_features: int, params: dict) -> jnp.ndarray:
        """Initialize expanded weights."""
        n_expanded = self._expanded_n_features(n_features)
        if self._requires_nonzero_init():
            init_std = jnp.exp(params.get('log_init_std', jnp.array(-3.0)))
            return init_std * jnp.ones(n_expanded)
        init_value = params.get('init_value', jnp.array(0.0))
        return init_value * jnp.ones(n_expanded)

    def _compute_env_loss(
        self,
        params: dict[str, jnp.ndarray],
        w: jnp.ndarray,
        env_goals: jnp.ndarray,
        env_distractors: jnp.ndarray,
        env_goal_mask: jnp.ndarray,
        env_distractor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute KL loss with diagonal saliency (element-wise, O(n+n^2))."""
        q = jax.nn.sigmoid(params['log_q'])
        S_diag = params['S_diag']
        n_features = env_goals.shape[1]

        # Expand features to quadratic space
        expanded_goals = _expand_features_quadratic(env_goals)
        expanded_distractors = _expand_features_quadratic(env_distractors)

        # Apply non-linearity then diagonal saliency (element-wise multiply, not matmul!)
        effective_w = S_diag * self._apply_non_linearity(w, params)

        # Compute logits
        goal_logits = jnp.sum(expanded_goals * effective_w, axis=1)
        distractor_logits = jnp.sum(expanded_distractors * effective_w, axis=1)

        # Compute no-goal logit using appropriate no-goal feature vector
        no_goal_features = get_no_goal_features(n_features, self.include_no_goal_feature)
        expanded_no_goal = _expand_features_quadratic(no_goal_features)
        no_goal_logit = jnp.sum(expanded_no_goal * effective_w)

        # Mask invalid entries
        goal_logits = jnp.where(env_goal_mask > 0, goal_logits, -1e10)
        distractor_logits = jnp.where(env_distractor_mask > 0, distractor_logits, -1e10)

        # Compute partition function with learned no-goal value
        Z = (jnp.sum(jnp.exp(goal_logits) * env_goal_mask) +
             jnp.sum(jnp.exp(distractor_logits) * env_distractor_mask) +
             jnp.exp(no_goal_logit))

        # Total goal probability
        pi_G = jnp.sum(jnp.exp(goal_logits) * env_goal_mask) / Z
        pi_G = jnp.clip(pi_G, EPSILON, 1.0 - EPSILON)

        # KL divergence
        if self.reverse_kl:
            return pi_G * jnp.log(pi_G / q) + (1 - pi_G) * jnp.log((1 - pi_G) / (1 - q))
        else:
            return q * jnp.log(q / pi_G) + (1 - q) * jnp.log((1 - q) / (1 - pi_G))

    def _compute_stage_gradient_batched(
        self,
        params: dict[str, jnp.ndarray],
        w: jnp.ndarray,
        stage_goals: jnp.ndarray,
        stage_distractors: jnp.ndarray,
        stage_env_mask: jnp.ndarray,
        stage_goal_mask: jnp.ndarray,
        stage_distractor_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute gradient using autodiff over environments."""
        def single_env_gradient(env_idx: Int) -> jnp.ndarray:
            grad_w = jax.grad(lambda w_: self._compute_env_loss(
                params, w_,
                stage_goals[env_idx],
                stage_distractors[env_idx],
                stage_goal_mask[env_idx],
                stage_distractor_mask[env_idx],
            ))(w)
            return stage_env_mask[env_idx] * grad_w

        n_envs = stage_goals.shape[0]
        env_gradients = jax.vmap(single_env_gradient)(jnp.arange(n_envs))
        total_gradient = jnp.sum(env_gradients, axis=0)

        n_valid_envs = jnp.maximum(jnp.sum(stage_env_mask), 1.0)
        return -total_gradient / n_valid_envs

    def single_example_loss(
        self,
        params: dict[str, jnp.ndarray],
        padded: PaddedPipeline,
        goal_0: jnp.ndarray,
        goal_1: jnp.ndarray,
        observed_prob_0: jnp.ndarray,
        observed_prob_1: jnp.ndarray,
        observed_prob_no_goal: jnp.ndarray,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute loss with diagonal saliency and quadratic features."""
        S_diag = params['S_diag']
        n_features = goal_0.shape[0]

        modified_padded = self._apply_pipeline_mode(padded)

        agent_weights = _compute_agent_params_from_pipeline_batched(
            initial_params=self._get_initial_weights(n_features, params),
            compute_agent_params_from_stage_equilibrium_batched_fn=self._get_compute_stage_equilibrium_batched_fn(params),
            padded=modified_padded,
        )

        # Element-wise saliency (not matmul!)
        effective_weights = S_diag * self._apply_non_linearity(agent_weights, params)

        # Expand goal features
        expanded_goal_0 = _expand_features_quadratic(goal_0)
        expanded_goal_1 = _expand_features_quadratic(goal_1)

        value_0 = jnp.dot(effective_weights, expanded_goal_0)
        value_1 = jnp.dot(effective_weights, expanded_goal_1)

        # Use 3-way loss (including no-goal) when observed_prob_no_goal >= 0
        # (full_distribution mode), 2-way loss when sentinel value < 0
        # (unweighted modes set no_goal to -1.0 sentinel)
        def two_way_loss():
            logits = jnp.array([value_0, value_1])
            predicted_probs = jax.nn.softmax(logits)
            observed_probs = jnp.array([observed_prob_0, observed_prob_1])
            return jnp.sum(
                observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON))
            )

        def three_way_loss():
            # Compute no-goal value using appropriate no-goal feature vector
            no_goal_features = get_no_goal_features(n_features, self.include_no_goal_feature)
            expanded_no_goal = _expand_features_quadratic(no_goal_features)
            value_no_goal = jnp.dot(effective_weights, expanded_no_goal)
            logits = jnp.array([value_0, value_1, value_no_goal])
            predicted_probs = jax.nn.softmax(logits)
            observed_probs = jnp.array([observed_prob_0, observed_prob_1, observed_prob_no_goal])
            return jnp.sum(
                observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON))
            )

        kl_div = jax.lax.cond(observed_prob_no_goal >= 0, three_way_loss, two_way_loss)

        return weight * kl_div

    def learn_agent_parameters_to_be_saved(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    ) -> dict[str, Any]:
        """Return agent parameters in expanded space."""
        agent_parameters = {}
        n_features = len(all_features)

        expanded_features = list(all_features)
        for i, f_i in enumerate(all_features):
            for j, f_j in enumerate(all_features):
                expanded_features.append(f"{f_i}*{f_j}")

        for agent_name, (_, padded) in agent_training_pipelines.items():
            modified_padded = self._apply_pipeline_mode_static(padded)

            agent_params = _compute_agent_params_from_pipeline_batched(
                initial_params=self._get_initial_weights(n_features, params),
                compute_agent_params_from_stage_equilibrium_batched_fn=self._get_compute_stage_equilibrium_batched_fn(params),
                padded=modified_padded,
            )
            agent_weights = {
                feature: float(value)
                for feature, value in zip(expanded_features, agent_params)
                if abs(float(value)) > EPSILON
            }
            agent_parameters[agent_name] = agent_weights

        return agent_parameters
