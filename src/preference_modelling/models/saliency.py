"""
Value functions for preference models.

This module provides a unified ValueFunction interface for computing goal values
from agent weights and goal features: (w, phi) -> scalar.

Public value function:
- LinearValueFunction: effective_w = S @ w, value = effective_w . phi

(MLPValueFunction and BilinearValueFunction live in src/private/ and are
loaded on demand by make_value_function below.)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Callable
import logging

import jax
import jax.numpy as jnp
from tabulate import tabulate

from .base import (
    HyperparameterSpec,
    CompositeSpec,
    CustomSpec,
    MatrixSpec,
)


log = logging.getLogger(__name__)


class ValueFunction(ABC):
    """
    Abstract base class for value functions.

    A value function computes a scalar value from agent weights and goal features:
    forward(w, goal_features, params) -> scalar

    All value functions must implement:
        - name: Human-readable name for logging
        - get_specs: Return HyperparameterSpec list for value function params
        - init_params: Initialize parameters from specification
        - forward: Compute value from weights and features
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def get_specs(self) -> list[HyperparameterSpec]:
        """
        Return hyperparameter specs for value function parameters.

        Returns:
            List of HyperparameterSpec instances
        """
        ...

    def get_param_spec(self, n_features: int) -> dict[str, dict]:
        """
        Return hyperparameter specification as a dict (legacy interface).

        Args:
            n_features: Number of features

        Returns:
            Dict mapping param name to spec with 'init' and 'transform' keys.
        """
        specs = self.get_specs()
        result = {}
        for spec in specs:
            result[spec.name] = {'init': 'custom', 'transform': spec.transform}
        return result

    @abstractmethod
    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        """
        Initialize value function parameters.

        Args:
            n_features: Number of features

        Returns:
            Dict of initialized parameters
        """
        ...

    @abstractmethod
    def forward(
        self,
        w: jnp.ndarray,
        goal_features: jnp.ndarray,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute goal value from agent weights and goal features.

        Args:
            w: Agent weights, shape (n_features,) or (latent_dim,)
            goal_features: Goal feature vector, shape (n_features,)
            params: Value function parameters

        Returns:
            Scalar goal value
        """
        ...

    @abstractmethod
    def forward_batched(
        self,
        ws: jnp.ndarray,
        goals_batch: jnp.ndarray,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute goal values for multiple weight vectors and goals efficiently.

        This method enables fused matrix operations by batching over both
        weight vectors and goals simultaneously.

        Args:
            ws: Agent weights, shape (n_weights, latent_dim) or (n_weights, n_features)
            goals_batch: Goal features, shape (n_goals, n_features)
            params: Value function parameters

        Returns:
            Goal values, shape (n_weights, n_goals)
        """
        ...

    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        """
        Log value function parameters in human-readable format.

        Default implementation delegates to each spec's log_params method.
        """
        for spec in self.get_specs():
            spec.log_params(params, all_features, features_in_training)

    def compute_regularisation_loss(
        self,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute regularization loss for value function parameters.

        Default implementation returns 0. Override in subclasses.

        Args:
            params: Model parameters

        Returns:
            Scalar regularization loss
        """
        return jnp.array(0.0)

    def get_latent_dim(self, n_features: int) -> int:
        """
        Return the dimension of the weight vector w.

        Default is n_features. Override for value functions that use
        a different latent dimension.

        Args:
            n_features: Number of features

        Returns:
            Dimension of weight vector
        """
        return n_features


# =============================================================================
# Linear Saliency Spec
# =============================================================================

@dataclass
class LinearSaliencySpec(CompositeSpec):
    """
    S matrix spec that also logs SS^T.

    The S matrix transforms weights: effective_w = S @ w
    """
    name: str = 'S'
    transform: Literal['exp', 'sigmoid', 'none'] = 'none'
    structure: Literal["full", "upper_triangular", "diagonal"] = "full"
    init_type: Literal["identity", "random_gaussian"] = "identity"
    seed: int = 42
    latent_dimension: int | None = None

    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        latent_dim = self.latent_dimension if self.latent_dimension is not None else n_features

        if self.init_type == "identity":
            S = jnp.eye(n_features, latent_dim)
        else:  # random_gaussian
            # Compute number of non-zero entries based on structure
            if self.structure == "diagonal":
                k = min(n_features, latent_dim)
            elif self.structure == "upper_triangular":
                k = n_features * (n_features + 1) // 2
            else:  # full
                k = n_features * latent_dim

            # Scale so E[||S||_F^2] = n_features (matching identity norm)
            std = jnp.sqrt(n_features / k)
            key = jax.random.PRNGKey(self.seed)
            S = std * jax.random.normal(key, shape=(n_features, latent_dim))

            if self.structure == "upper_triangular":
                S = jnp.triu(S)
            elif self.structure == "diagonal":
                S = jnp.diag(jnp.diag(S))

        return {'S': S}

    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        S = self._get_constrained_S(params['S'])
        n_features = len(all_features)
        latent_dim = S.shape[1]

        # Log S matrix
        matrix_rows = []
        for i in range(n_features):
            row_data = [all_features[i]]
            for j in range(latent_dim):
                val = float(S[i, j])
                row_data.append(f"{val:.3f}")
            matrix_rows.append(row_data)

        if latent_dim == n_features:
            col_headers = all_features
        else:
            col_headers = [f"z{j}" for j in range(latent_dim)]

        headers = ["S_ij"] + col_headers
        structure_info = f", structure={self.structure}" if self.structure != "full" else ""
        latent_info = f", latent_dim={latent_dim}" if latent_dim != n_features else ""
        log.info(
            f"Saliency Matrix (effective_w = S @ w{structure_info}{latent_info}):\n"
            f"{tabulate(matrix_rows, headers=headers, tablefmt='simple_outline', stralign='right')}"
        )

        # Log SS^T (effective feature similarity matrix)
        SST = S @ S.T
        sst_rows = []
        for i in range(n_features):
            row_data = [all_features[i]]
            for j in range(n_features):
                val = float(SST[i, j])
                if i == j:
                    row_data.append(f"[{val:.3f}]")
                else:
                    row_data.append(f"{val:.3f}")
            sst_rows.append(row_data)

        sst_headers = ["SS^T_ij"] + all_features
        log.info(
            f"Feature Similarity Matrix SS^T ([diagonal] highlighted):\n"
            f"{tabulate(sst_rows, headers=sst_headers, tablefmt='simple_outline', stralign='right')}"
        )

    def _get_constrained_S(self, S: jnp.ndarray) -> jnp.ndarray:
        """Apply structure constraint to S matrix."""
        if self.structure == "full":
            return S
        elif self.structure == "upper_triangular":
            return jnp.triu(S)
        elif self.structure == "diagonal":
            return jnp.diag(jnp.diag(S))
        else:
            raise ValueError(f"Unknown structure: {self.structure}")


class LinearValueFunction(ValueFunction):
    """
    Linear value function: value = (S @ w) . phi

    Computes effective weights via a saliency matrix S, then dots with features.
    This is the simplest value function with interpretable structure.

    Supports:
    - Structure constraints: full, upper_triangular, diagonal
    - Initialization: identity, random_gaussian
    - Non-square matrices via latent_dimension
    """

    def __init__(
        self,
        structure: Literal["full", "upper_triangular", "diagonal"] = "full",
        init: Literal["identity", "random_gaussian"] = "identity",
        seed: int = 42,
        latent_dimension: int | None = None,
    ):
        """
        Args:
            structure: Constraint on S matrix structure:
                - "full": No restriction (default)
                - "upper_triangular": Only upper triangular entries
                - "diagonal": Only diagonal entries
            init: Initialization method for S matrix:
                - "identity": Identity matrix (default)
                - "random_gaussian": i.i.d. N(0, sigma^2) with sigma scaled for norm preservation
            seed: Random seed for Gaussian initialization
            latent_dimension: Dimension of weight vector w. If None, defaults to n_features.
                When specified, S becomes (n_features, latent_dimension).
                Only compatible with structure="full".
        """
        if latent_dimension is not None:
            if structure != "full":
                raise ValueError(
                    f"latent_dimension requires structure='full', got '{structure}'"
                )
            if init != "random_gaussian":
                raise ValueError(
                    f"latent_dimension requires init='random_gaussian', got '{init}'"
                )

        self.structure = structure
        self.init = init
        self.seed = seed
        self.latent_dimension = latent_dimension

        # Create the spec for this value function
        self._spec = LinearSaliencySpec(
            structure=structure,
            init_type=init,
            seed=seed,
            latent_dimension=latent_dimension,
        )

    @property
    def name(self) -> str:
        parts = ["Linear"]
        if self.structure != "full":
            parts.append(f"({self.structure})")
        if self.latent_dimension is not None:
            parts.append(f"[latent={self.latent_dimension}]")
        return "".join(parts)

    def get_latent_dim(self, n_features: int) -> int:
        return self.latent_dimension if self.latent_dimension is not None else n_features

    def get_specs(self) -> list[HyperparameterSpec]:
        return [self._spec]

    def get_param_spec(self, n_features: int) -> dict[str, dict]:
        return {
            'S': {'init': 'custom', 'transform': 'none'},
        }

    def init_params(self, n_features: int) -> dict[str, jnp.ndarray]:
        return self._spec.init_params(n_features)

    def _get_constrained_S(self, S: jnp.ndarray) -> jnp.ndarray:
        """Apply structure constraint to S matrix."""
        return self._spec._get_constrained_S(S)

    def forward(
        self,
        w: jnp.ndarray,
        goal_features: jnp.ndarray,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute value = (S @ w) . phi"""
        S = self._get_constrained_S(params['S'])
        effective_w = S @ w
        return jnp.dot(effective_w, goal_features)

    def forward_batched(
        self,
        ws: jnp.ndarray,
        goals_batch: jnp.ndarray,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute values for multiple weight vectors and goals.

        Uses fused matrix operations: (ws @ S.T) @ goals.T

        Args:
            ws: Weight vectors, shape (n_weights, latent_dim)
            goals_batch: Goal features, shape (n_goals, n_features)
            params: Value function parameters

        Returns:
            Goal values, shape (n_weights, n_goals)
        """
        S = self._get_constrained_S(params['S'])
        effective_ws = ws @ S.T  # (n_weights, n_features)
        return effective_ws @ goals_batch.T  # (n_weights, n_goals)

    def log_params(
        self,
        params: dict[str, jnp.ndarray],
        all_features: list[str],
        features_in_training: set[str],
    ) -> None:
        self._spec.log_params(params, all_features, features_in_training)


def make_value_function(config: dict | None) -> ValueFunction:
    """
    Factory function to create value functions from config dicts.

    Args:
        config: Configuration dict with 'type' key and type-specific parameters.
                If None, returns LinearValueFunction with default settings
                (learnable S matrix initialized to identity).

    Returns:
        ValueFunction instance

    Examples:
        >>> make_value_function(None)  # Learnable S matrix initialized to identity
        >>> make_value_function({"type": "linear"})  # Same as above
        >>> make_value_function({"type": "linear", "structure": "diagonal"})
        >>> make_value_function({"type": "mlp", "hidden_sizes": [32]})
        >>> make_value_function({"type": "bilinear", "rank": 16})
    """
    if config is None:
        return LinearValueFunction()

    config = config.copy()
    vf_type = config.pop("type", "linear")

    if vf_type == "linear":
        return LinearValueFunction(**config)
    elif vf_type == "mlp":
        try:
            from src.private.preference_modelling.models.mlp_value_function import MLPValueFunction
        except ImportError as e:
            raise ValueError(
                "value_function type 'mlp' requires the private MLP module "
                "(src/private/preference_modelling/models/mlp_value_function.py); "
                "not included in the public release."
            ) from e
        return MLPValueFunction(**config)
    elif vf_type == "bilinear":
        try:
            from src.private.preference_modelling.models.bilinear import BilinearValueFunction
        except ImportError as e:
            raise ValueError(
                "value_function type 'bilinear' requires the private Bilinear module "
                "(src/private/preference_modelling/models/bilinear.py); "
                "not included in the public release."
            ) from e
        return BilinearValueFunction(**config)
    else:
        raise ValueError(f"Unknown value_function type: {vf_type}")
