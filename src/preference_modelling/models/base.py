"""
Base classes, mixins, and utilities for preference models.
"""
import jax
import jax.numpy as jnp
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Any, Generic, Callable
from jaxtyping import Array, Integer
from tabulate import tabulate
from functools import partial

from src.preference_modelling.data_structures import (
    TrainingPipeline,
    PaddedPipeline,
)
from src.preference_modelling.numerical_methods import (
    PT,
    IntegrationConfig,
    integrate_stage_batched,
)
from src.preference_modelling.data import get_no_goal_features


log = logging.getLogger(__name__)

# Module constant for numerical stability
EPSILON = 1e-8

Int = Integer[Array, ""] | int


# =============================================================================
# Hyperparameter Spec Classes
# =============================================================================

@dataclass
class HyperparameterSpec(ABC):
    """
    Base class for hyperparameter specifications.

    A spec encapsulates both initialization AND logging behavior for a hyperparameter,
    making the system extensible and eliminating hacky manual chaining.
    """
    name: str
    transform: Literal['exp', 'sigmoid', 'none'] = 'none'

    def param_key(self) -> str:
        """
        Derive param dict key from name and transform type.

        For exp/sigmoid transforms, params are stored in log-space with 'log_' prefix.
        For 'none' transform, params are stored directly with no prefix.
        """
        if self.transform in ('exp', 'sigmoid'):
            return f'log_{self.name}'
        return self.name

    @abstractmethod
    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        """
        Initialize parameters from this spec.

        Args:
            n_features: Number of features for matrix-valued parameters

        Returns:
            Dict of initialized parameters (may be empty, one, or multiple params)
        """
        ...

    @abstractmethod
    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        """
        Log parameters in human-readable format.

        Args:
            params: Full parameter dict
            all_features: List of all feature names
            features_in_training: Set of feature names that appear in training
        """
        ...


@dataclass
class ScalarSpec(HyperparameterSpec):
    """
    Single scalar hyperparameter (e.g., q, beta, gamma).

    Scalars are logged in a compact table format by the base class,
    so the log_params method here is a no-op.
    """
    init: float = 0.0
    description: str | None = None

    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        return {self.param_key(): jnp.array(float(self.init))}

    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        # Scalars are logged in batch by the base class log_hyperparameters
        pass


@dataclass
class MatrixSpec(HyperparameterSpec):
    """
    Matrix hyperparameter (e.g., theta, S).

    Supports identity or zeros initialization, and logs the matrix with
    feature labels and optional derived values.
    """
    init: Literal['identity', 'zeros'] = 'identity'
    header: str = 'M_ij'
    description: str = 'Matrix parameter'

    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        if self.init == 'identity':
            return {self.param_key(): jnp.eye(n_features)}
        elif self.init == 'zeros':
            return {self.param_key(): jnp.zeros((n_features, n_features))}
        else:
            raise ValueError(f"Unknown init: {self.init}")

    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        matrix = params.get(self.param_key())
        if matrix is None:
            return
        self._log_matrix(matrix, all_features, self.header, self.description)

    def _log_matrix(
        self,
        matrix: jnp.ndarray,
        all_features: list[str],
        header: str,
        description: str,
    ) -> None:
        """Helper to log a matrix with feature labels."""
        n_features = len(all_features)
        n_rows, n_cols = matrix.shape

        matrix_rows = []
        for i in range(n_rows):
            row_label = all_features[i] if i < n_features else f"z{i}"
            row_data = [row_label]
            for j in range(n_cols):
                val = float(matrix[i, j])
                if i == j:
                    row_data.append(f"[{val:.3f}]")
                else:
                    row_data.append(f"{val:.3f}")
            matrix_rows.append(row_data)

        if n_cols == n_features:
            col_headers = all_features
        else:
            col_headers = [f"z{j}" for j in range(n_cols)]

        headers = [header] + col_headers
        log.info(
            f"{description} ([diagonal] highlighted):\n"
            f"{tabulate(matrix_rows, headers=headers, tablefmt='simple_outline', stralign='right')}"
        )


@dataclass
class CompositeSpec(HyperparameterSpec):
    """
    Multiple related parameters managed as a unit (e.g., bilinear B, C, A -> M).

    Subclasses should implement init_params to return all related parameters
    and log_params to log them together with any derived values.
    """

    def param_names(self) -> list[str]:
        """Return names of all parameters managed by this spec."""
        return [self.name]


@dataclass
class CustomSpec(HyperparameterSpec):
    """
    Escape hatch for complex cases (e.g., S_diag with special structure).

    Allows arbitrary initialization and logging functions.
    """
    init_fn: Callable[[int], dict[str, jnp.ndarray]] = field(default=lambda n: {})
    log_fn: Callable[[dict, list[str], set[str]], None] = field(default=lambda p, f, t: None)

    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        return self.init_fn(n_features)

    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        self.log_fn(params, all_features, features_in_training)


# =============================================================================
# Shared Spec Constants
# =============================================================================

Q_SPEC = ScalarSpec(
    name='q',
    init=2.2,  # 0.90
    transform='sigmoid',
    description='Target goal probability'
)

INIT_VALUE_SPEC = ScalarSpec(
    name='init_value',
    init=-0.4,
    transform='none',
    description='Initial weight value'
)

BETA_SPEC = ScalarSpec(
    name='beta',
    init=0.0,
    transform='exp',
    description='Inverse temperature'
)


@dataclass
class ThetaSpec(MatrixSpec):
    """
    Cross-feature saliency matrix (theta).

    theta_ij: learning from feature j increases weight on feature i.
    """
    name: str = 'theta'
    init: Literal['identity', 'zeros'] = 'identity'
    transform: Literal['exp', 'sigmoid', 'none'] = 'none'
    header: str = 'theta_ij'
    description: str = 'Cross-Feature Saliency Matrix (theta_ij: learning from feature j increases weight on feature i, [diagonal] = self-saliency)'


THETA_SPEC = ThetaSpec()


# =============================================================================
# Utility Functions
# =============================================================================

def _get_param_key(name: str, transform: str) -> str:
    """
    Derive param dict key from name and transform type.

    For exp/sigmoid transforms, params are stored in log-space with 'log_' prefix.
    For 'none' transform, params are stored directly with no prefix.
    """
    if transform in ('exp', 'sigmoid'):
        return f'log_{name}'
    return name


def _transform_param(
    log_param: jnp.ndarray,
    transform: str,
) -> jnp.ndarray:
    """Transform log-space parameter to value-space."""
    if transform == 'exp':
        return jnp.exp(log_param)
    elif transform == 'sigmoid':
        return jax.nn.sigmoid(log_param)
    elif transform == 'none':
        return log_param
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _format_param_value(param: jnp.ndarray) -> str:
    """Format parameter value for logging."""
    if param.ndim == 0:
        return f"{param.item():.4f}"
    elif param.ndim == 1:
        return "[" + ", ".join(f"{val.item():.4f}" for val in param) + "]"
    elif param.ndim == 2:
        rows = []
        for row in param:
            row_str = ", ".join(f"{val.item():.4f}" for val in row)
            rows.append(f"[{row_str}]")
        return "[\n  " + ",\n  ".join(rows) + "\n]"
    else:
        raise ValueError("Parameter formatting only supports scalars, 1D, and 2D matrices.")


# =============================================================================
# Pipeline Mode Mixin
# =============================================================================

class PipelineModeMixin:
    """
    Mixin providing pipeline mode transformations.

    Supports three modes for processing multi-stage training pipelines:
    - "sequential": Process stages in order (default)
    - "simultaneous": Flatten all stages into one (no temporal ordering)
    - "memoryless": Use only the final stage (ignores history)
    """

    pipeline_mode: Literal["sequential", "simultaneous", "memoryless"]

    def _flatten_pipeline_to_single_stage(self, padded: PaddedPipeline) -> PaddedPipeline:
        """
        Flatten all stages into a single stage by concatenating environments.

        Args:
            padded: Original multi-stage padded pipeline

        Returns:
            New PaddedPipeline with single stage containing all environments
        """
        n_stages, max_envs, max_goals, n_features = padded.goals.shape
        _, _, max_distractors, _ = padded.distractors.shape

        # Reshape by merging stages and envs dimensions
        flat_goals = padded.goals.reshape(1, n_stages * max_envs, max_goals, n_features)
        flat_distractors = padded.distractors.reshape(1, n_stages * max_envs, max_distractors, n_features)

        # Flatten masks: env_mask needs to incorporate stage_mask
        # An env is valid only if both stage and env are active
        combined_env_mask = padded.stage_mask[:, None] * padded.env_mask
        flat_env_mask = combined_env_mask.reshape(1, n_stages * max_envs)

        flat_goal_mask = padded.goal_mask.reshape(1, n_stages * max_envs, max_goals)
        flat_distractor_mask = padded.distractor_mask.reshape(1, n_stages * max_envs, max_distractors)

        return PaddedPipeline(
            goals=flat_goals,
            distractors=flat_distractors,
            stage_mask=jnp.ones(1),
            env_mask=flat_env_mask,
            goal_mask=flat_goal_mask,
            distractor_mask=flat_distractor_mask,
        )

    def _extract_final_stage(self, padded: PaddedPipeline) -> PaddedPipeline:
        """
        Extract only the final active stage (memoryless mode).

        Uses dynamic slicing for JIT compatibility.

        Args:
            padded: Original multi-stage padded pipeline

        Returns:
            New PaddedPipeline with only the final active stage
        """
        stage_mask = padded.stage_mask
        n_stages = stage_mask.shape[0]

        # Get index of last active stage (or 0 if none active)
        stage_indices = jnp.arange(n_stages)
        last_active_idx = jnp.max(jnp.where(stage_mask > 0, stage_indices, 0))

        return PaddedPipeline(
            goals=jax.lax.dynamic_slice(padded.goals, (last_active_idx, 0, 0, 0), (1,) + padded.goals.shape[1:]),
            distractors=jax.lax.dynamic_slice(padded.distractors, (last_active_idx, 0, 0, 0), (1,) + padded.distractors.shape[1:]),
            stage_mask=jnp.ones(1),
            env_mask=jax.lax.dynamic_slice(padded.env_mask, (last_active_idx, 0), (1, padded.env_mask.shape[1])),
            goal_mask=jax.lax.dynamic_slice(padded.goal_mask, (last_active_idx, 0, 0), (1,) + padded.goal_mask.shape[1:]),
            distractor_mask=jax.lax.dynamic_slice(padded.distractor_mask, (last_active_idx, 0, 0), (1,) + padded.distractor_mask.shape[1:]),
        )

    def _extract_final_stage_static(self, padded: PaddedPipeline) -> PaddedPipeline:
        """
        Extract only the final active stage using static slicing.

        Not JIT-compatible, but works for learn_agent_parameters_to_be_saved.

        Args:
            padded: Original multi-stage padded pipeline

        Returns:
            New PaddedPipeline with only the final active stage
        """
        stage_mask = padded.stage_mask
        n_stages = stage_mask.shape[0]
        last_active_idx = int(jnp.max(jnp.where(stage_mask > 0, jnp.arange(n_stages), 0)))

        return PaddedPipeline(
            goals=padded.goals[last_active_idx:last_active_idx+1],
            distractors=padded.distractors[last_active_idx:last_active_idx+1],
            stage_mask=jnp.ones(1),
            env_mask=padded.env_mask[last_active_idx:last_active_idx+1],
            goal_mask=padded.goal_mask[last_active_idx:last_active_idx+1],
            distractor_mask=padded.distractor_mask[last_active_idx:last_active_idx+1],
        )

    def _apply_pipeline_mode(self, padded: PaddedPipeline) -> PaddedPipeline:
        """Apply pipeline mode transformation (JIT-compatible)."""
        if self.pipeline_mode == "sequential":
            return padded
        elif self.pipeline_mode == "simultaneous":
            return self._flatten_pipeline_to_single_stage(padded)
        elif self.pipeline_mode == "memoryless":
            return self._extract_final_stage(padded)
        else:
            raise ValueError(f"Unknown pipeline_mode: {self.pipeline_mode}")

    def _apply_pipeline_mode_static(self, padded: PaddedPipeline) -> PaddedPipeline:
        """Apply pipeline mode transformation using static slicing."""
        if self.pipeline_mode == "sequential":
            return padded
        elif self.pipeline_mode == "simultaneous":
            return self._flatten_pipeline_to_single_stage(padded)
        elif self.pipeline_mode == "memoryless":
            return self._extract_final_stage_static(padded)
        else:
            raise ValueError(f"Unknown pipeline_mode: {self.pipeline_mode}")


# =============================================================================
# Pipeline Computation Utilities
# =============================================================================

def _compute_agent_params_from_pipeline_batched[T](
    initial_params: T,
    compute_agent_params_from_stage_equilibrium_batched_fn: Callable[
        [T, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        T
    ],
    padded: PaddedPipeline,
) -> T:
    """
    Compute final agent parameters after training for a batch of pipelines using padded arrays.
    """
    goals = padded.goals
    distractors = padded.distractors
    stage_mask = padded.stage_mask
    env_mask = padded.env_mask
    goal_mask = padded.goal_mask
    distractor_mask = padded.distractor_mask

    def process_stage(w_prev: T, stage_data) -> tuple[T, None]:
        stage_goals, stage_distractors, stage_active, stage_env_mask, stage_goal_mask, stage_distractor_mask = stage_data

        w_out = compute_agent_params_from_stage_equilibrium_batched_fn(
            w_prev,
            stage_goals, stage_distractors,
            stage_env_mask, stage_goal_mask, stage_distractor_mask,
            stage_active,
        )
        return w_out, None

    stage_data = (goals, distractors, stage_mask, env_mask, goal_mask, distractor_mask)
    w_final, _ = jax.lax.scan(process_stage, initial_params, stage_data)

    return w_final


def _compute_agent_params_from_stage_equilibrium_batched[T](
    compute_stage_gradient_batched_fn: Callable[
        [T, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        T
    ],
    integration_config: IntegrationConfig,
    w_prev: T,
    stage_goals: jnp.ndarray,
    stage_distractors: jnp.ndarray,
    stage_env_mask: jnp.ndarray,
    stage_goal_mask: jnp.ndarray,
    stage_distractor_mask: jnp.ndarray,
    stage_active: jnp.ndarray,
) -> T:
    """
    Compute equilibrium agent parameters for a single stage using numerical integration (batched).
    """
    def gradient_fn(w):
        return compute_stage_gradient_batched_fn(
            w,
            stage_goals, stage_distractors,
            stage_env_mask, stage_goal_mask, stage_distractor_mask,
        )

    return integrate_stage_batched(gradient_fn, w_prev, stage_active, integration_config)


# =============================================================================
# Saliency Model Mixin
# =============================================================================

class SaliencyModelMixin:
    """
    Mixin providing shared saliency matrix (theta) hyperparameter.
    """

    def __init__(
        self,
        saliency_regularisation_strength: float = 0.01,
        **kwargs
    ):
        self.saliency_regularisation_strength = saliency_regularisation_strength
        super().__init__(**kwargs)

    def _get_saliency_specs(self) -> list[HyperparameterSpec]:
        """Return the saliency hyperparameter specs."""
        return [THETA_SPEC]

    def _compute_saliency_regularisation_loss(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute saliency matrix regularization loss."""
        theta = params['theta']

        # Kinda cursed?
        froebenius_norm = jnp.linalg.norm(theta, ord='fro')
        target_norm = jnp.sqrt(theta.shape[0])
        f_control = (froebenius_norm - target_norm)**2

        # Seems better but nan's
        froebenius_norm_from_eye = jnp.linalg.norm(theta - jnp.eye(theta.shape[0]), ord='fro')
        # f_control = froebenius_norm_from_eye**2

        return self.saliency_regularisation_strength * f_control


# =============================================================================
# Non-Linearity Mixin
# =============================================================================

class NonLinearityMixin:
    """
    Mixin providing optional non-linearity applied to weights.

    Non-linearity options:
    - "none": Identity (default)
    - "quadratic": f(w) = w^2
    - "sigmoid": f(w) = sigmoid(w)
    - "power_law": f(w) = sign(w) * |w|^gamma (gamma is learnable)

    Note: For quadratic and power_law non-linearities, weights must be initialized
    with non-zero values since the gradient at w=0 is zero (f'(0) = 0).
    This mixin adds an init_std hyperparameter for such cases.
    """

    def __init__(
        self,
        non_linearity: Literal["none", "quadratic", "sigmoid", "power_law"] = "none",
        **kwargs
    ):
        self.non_linearity = non_linearity
        super().__init__(**kwargs)

    def _requires_nonzero_init(self) -> bool:
        """Check if the non-linearity requires non-zero weight initialization."""
        # quadratic: f(w) = w^2, f'(w) = 2w, f'(0) = 0
        # power_law: f(w) = sign(w)|w|^gamma, f'(0) = 0 for gamma > 1
        return self.non_linearity in ("quadratic", "power_law")

    def _apply_non_linearity(self, x: jnp.ndarray, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Apply the configured non-linearity to weights."""
        if self.non_linearity == "none":
            return x
        elif self.non_linearity == "quadratic":
            return x ** 2
        elif self.non_linearity == "sigmoid":
            return jax.nn.sigmoid(x)
        elif self.non_linearity == "power_law":
            gamma = jnp.exp(params['log_gamma'])
            return jnp.sign(x) * jnp.abs(x) ** gamma
        else:
            raise ValueError(f"Unknown non-linearity: {self.non_linearity}")

    def _get_non_linearity_specs(self) -> list[HyperparameterSpec]:
        """Return hyperparameter specs for non-linearity parameters."""
        specs = []
        if self.non_linearity == "power_law":
            specs.append(ScalarSpec(
                name='gamma',
                init=0.0,
                transform='exp',
                description='Power law exponent'
            ))
        if self._requires_nonzero_init():
            specs.append(ScalarSpec(
                name='init_std',
                init=-3.0,
                transform='exp',
                description='Weight initialization std'
            ))
        return specs

    def _get_initial_weights(
        self,
        n_features: int,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Get initial weights for pipeline computation.

        For non-linearities that require non-zero initialization (quadratic, power_law),
        initializes with small positive values. Otherwise uses init_value (defaults to 0).
        """
        if self._requires_nonzero_init():
            init_std = jnp.exp(params.get('log_init_std', jnp.array(-3.0)))
            # Use deterministic small positive values (avoid randomness for reproducibility)
            # Initialize with init_std * ones, which gives non-zero gradient
            return init_std * jnp.ones(n_features)
        # Use init_value if present, otherwise default to 0
        init_value = params.get('init_value', jnp.array(0.0))
        return init_value * jnp.ones(n_features)

    def _format_name_with_non_linearity(self, base_name: str) -> str:
        """Format model name to include non-linearity if not 'none'."""
        if self.non_linearity != "none":
            return f"{base_name}, f={self.non_linearity}"
        return base_name

    def _transform_weights_for_choice(
        self,
        w: jnp.ndarray,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Override to apply non-linearity when transforming weights for choice."""
        return self._apply_non_linearity(w, params)


# =============================================================================
# Base Model Classes
# =============================================================================

class PhenomenologicalModel(ABC, Generic[PT]):
    """
    Abstract base class for weight evolution models.

    Subclasses must implement:
        - name: Model name for logging
        - get_hyperparameter_specs: List of HyperparameterSpec for model hyperparameters
        - _compute_stage_gradient_batched: Batched gradient computation

    The base class provides:
        - compute_pipeline_weights: Non-batched pipeline computation
        - compute_pipeline_weights_batched: Batched pipeline computation
        - init_params_from_spec: Initialize parameters from hyperparameter specs
        - log_hyperparameters: Log all hyperparameters via their specs
    """

    def __init__(
        self,
        integration_kwargs: dict | None = None,
        include_no_goal_feature: bool = False,
    ):
        """
        Args:
            integration_kwargs: Configuration for numerical integration
            include_no_goal_feature: If True, the feature vector includes a
                dedicated 'no_goal' feature as the last dimension. When computing no-goal
                values in three-way loss, uses [0,...,0,1]. If False, uses zeros for
                no-goal features (original behavior before no_goal feature was added).
        """
        if integration_kwargs is not None:
            self.integration_config = IntegrationConfig(**integration_kwargs)
        else:
            log.info("No integration_kwargs provided, using default IntegrationConfig (max_steps=100)")
            self.integration_config = IntegrationConfig()

        self.include_no_goal_feature = include_no_goal_feature

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
        ...

    @abstractmethod
    def get_hyperparameter_specs(self) -> list[HyperparameterSpec]:
        """
        Return specification for model hyperparameters (level 2 params).

        These are parameters optimized by the outer training loop but fixed
        during pipeline weight computation.

        Returns:
            List of HyperparameterSpec instances
        """
        ...

    def get_hyperparameter_spec(self, n_features: int | None = None) -> dict[str, dict]:
        """
        Return specification for model hyperparameters as a dict (legacy interface).

        This method provides backward compatibility with the old dict-based interface.

        Args:
            n_features: Number of features (ignored, for backward compatibility)

        Returns:
            Dict mapping param name to spec with 'init' and 'transform' keys
        """
        spec = {}
        for hp_spec in self.get_hyperparameter_specs():
            if isinstance(hp_spec, ScalarSpec):
                spec[hp_spec.name] = {'init': hp_spec.init, 'transform': hp_spec.transform}
            elif isinstance(hp_spec, MatrixSpec):
                init_str = 'identity_matrix' if hp_spec.init == 'identity' else 'zeros_matrix'
                spec[hp_spec.name] = {'init': init_str, 'transform': hp_spec.transform}
            elif isinstance(hp_spec, (CompositeSpec, CustomSpec)):
                # For composite/custom specs, we mark as custom
                spec[hp_spec.name] = {'init': 'custom', 'transform': hp_spec.transform}
        return spec

    def init_params_from_spec(self, n_features: int) -> dict[str, jnp.ndarray]:
        """
        Initialize hyperparameters from the specifications.

        Args:
            n_features: Number of features for matrix-valued parameters

        Returns:
            Dict of initialized parameters
        """
        params = {}
        for spec in self.get_hyperparameter_specs():
            params.update(spec.init_params(n_features))
        return params

    def compute_regularisation_loss(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute regularization loss for the model.

        Args:
            params: Model parameters

        Returns:
            Regularization loss scalar
        """
        return jnp.array(0.0)

    def log_hyperparameters(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ):
        """
        Log all hyperparameters via their specs.

        Scalar specs are grouped into a compact table.
        Non-scalar specs are logged individually.

        Args:
            params: Fitted parameters in log-space
            all_features: List of all feature names
            features_in_training: Set of feature names that appear in training
        """
        specs = self.get_hyperparameter_specs()

        # Group scalars for compact table display
        scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]
        if scalar_specs:
            rows = []
            for spec in scalar_specs:
                param_key = spec.param_key()
                if param_key in params:
                    raw_value = params[param_key]
                    transformed_value = _transform_param(raw_value, spec.transform)
                    rows.append([spec.name, _format_param_value(transformed_value)])
            if rows:
                log.info(f"Scalar Parameters:\n{tabulate(rows, headers=['Parameter', 'Value'], tablefmt='simple_outline')}")

        # Log non-scalar specs individually
        for spec in specs:
            if not isinstance(spec, ScalarSpec):
                spec.log_params(params, all_features, features_in_training)

    @abstractmethod
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
        """
        Compute loss for a single training example using padded pipeline.

        Args:
            params: Model parameters
            padded: PaddedPipeline for the training example
            goal_0: Features of goal 0
            goal_1: Features of goal 1
            observed_prob_0: Observed probability of choosing goal 0
            observed_prob_1: Observed probability of choosing goal 1
            observed_prob_no_goal: Observed probability of choosing neither goal (no-goal)
            weight: Weight for this example

        Returns:
            Loss scalar
        """
        ...

    @abstractmethod
    def learn_agent_parameters_to_be_saved(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    ) -> dict[str, Any]:
        """
        Return agent-specific parameters to be saved after training.

        Returns:
            Dict of agent-specific parameters
        """
        ...


class SimpleWeightsModel(PhenomenologicalModel):
    """
    Base class for models which use a simple vector of agent weights.
    """

    def _get_initial_weights(
        self,
        n_features: int,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Get initial weights for pipeline computation.

        Override in subclasses (e.g., NonLinearityMixin) to use non-zero initialization
        when required by the model (e.g., for quadratic non-linearity where f'(0) = 0).

        Args:
            n_features: Number of features
            params: Model parameters

        Returns:
            Initial weight vector, shape (n_features,)
        """
        # Use init_value if present in params, otherwise default to 0
        init_value = params.get('init_value', jnp.array(0.0))
        return init_value * jnp.ones(n_features)

    def learn_agent_parameters_to_be_saved(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    ) -> dict[str, Any]:

        agent_parameters = {}

        for agent_name, (_, padded) in agent_training_pipelines.items():
            agent_params = _compute_agent_params_from_pipeline_batched(
                initial_params=self._get_initial_weights(len(all_features), params),
                compute_agent_params_from_stage_equilibrium_batched_fn=self._get_compute_stage_equilibrium_batched_fn(params),
                padded=padded,
            )
            agent_weights = {
                feature: float(value)
                for feature, value in zip(all_features, agent_params)
                if abs(float(value)) > EPSILON
            }
            agent_parameters[agent_name] = agent_weights

        return agent_parameters

    def _transform_weights_for_choice(
        self,
        w: jnp.ndarray,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Transform weights for choice prediction.

        Override in subclasses to apply non-linearity or other transformations.

        Args:
            w: Learned weights, shape (n_features,)
            params: Model parameters

        Returns:
            Transformed weights for computing goal values
        """
        return w

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

        beta = jnp.exp(params.get('log_beta', jnp.array(0.0)))
        n_features = goal_0.shape[0]
        # agent_weights = self._compute_pipeline_weights_batched(params, padded)
        agent_weights = _compute_agent_params_from_pipeline_batched(
            initial_params=self._get_initial_weights(n_features, params),
            compute_agent_params_from_stage_equilibrium_batched_fn=self._get_compute_stage_equilibrium_batched_fn(params),
            padded=padded,
        )

        # Transform weights for choice prediction (applies non-linearity if configured)
        effective_weights = self._transform_weights_for_choice(agent_weights, params)

        # Compute goal values
        value_0 = jnp.dot(effective_weights, goal_0)
        value_1 = jnp.dot(effective_weights, goal_1)

        # Use 3-way loss (including no-goal) when observed_prob_no_goal >= 0
        # (full_distribution mode), 2-way loss when sentinel value < 0
        # (unweighted modes set no_goal to -1.0 sentinel)
        def two_way_loss():
            logits = jnp.array([beta * value_0, beta * value_1])
            predicted_probs = jax.nn.softmax(logits)
            observed_probs = jnp.array([observed_prob_0, observed_prob_1])
            return jnp.sum(
                observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON))
            )

        def three_way_loss():
            # Create no-goal feature vector based on include_no_goal_feature setting
            no_goal_features = get_no_goal_features(n_features, self.include_no_goal_feature)
            value_no_goal = jnp.dot(effective_weights, no_goal_features)
            logits = jnp.array([beta * value_0, beta * value_1, beta * value_no_goal])
            predicted_probs = jax.nn.softmax(logits)
            observed_probs = jnp.array([observed_prob_0, observed_prob_1, observed_prob_no_goal])
            return jnp.sum(
                observed_probs * (jnp.log(observed_probs + EPSILON) - jnp.log(predicted_probs + EPSILON))
            )

        kl_div = jax.lax.cond(observed_prob_no_goal >= 0, three_way_loss, two_way_loss)

        return weight * kl_div

    def _get_compute_stage_equilibrium_batched_fn(
        self,
        params: dict[str, jnp.ndarray],
    ) -> Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        jnp.ndarray
    ]:
        """
        Can be overwritten, especially useful for closed-form models
        """
        return partial(
            _compute_agent_params_from_stage_equilibrium_batched,
            partial(self._compute_stage_gradient_batched, params),
            self.integration_config,
        )

    @abstractmethod
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
        Compute weight gradient for a single stage in batched format.

        This method computes dw/dt for numerical integration in the batched case.

        Args:
            params: Model parameters
            w: Current weights, shape (n_features,)
            stage_goals: Goal features, shape (max_envs, max_goals, n_features)
            stage_distractors: Distractor features, shape (max_envs, max_distractors, n_features)
            stage_env_mask: Environment validity mask, shape (max_envs,)
            stage_goal_mask: Goal validity mask, shape (max_envs, max_goals)
            stage_distractor_mask: Distractor validity mask, shape (max_envs, max_distractors)

        Returns:
            Gradient dw/dt, shape (n_features,)
        """
        ...
