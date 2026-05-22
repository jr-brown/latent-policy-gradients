"""
Closed-form equilibrium models: ClosedFormModel, RWModel, KLModel.
"""
import jax
import jax.numpy as jnp
import logging

from abc import abstractmethod
from typing import Callable

from .base import (
    EPSILON,
    SimpleWeightsModel,
    SaliencyModelMixin,
    HyperparameterSpec,
    ScalarSpec,
    BETA_SPEC,
    Q_SPEC,
)


log = logging.getLogger(__name__)


class ClosedFormModel(SimpleWeightsModel):
    """
    Base class for models with closed-form equilibrium solutions.

    These models lump all goal features together and ignore distractors.
    Subclasses implement the core equilibrium computation and either:
    - Override _compute_gradient_from_phi for explicit gradients, OR
    - Implement _compute_loss_from_phi for autodiff-based gradients
    """
    has_warned_about_integration = False

    def __init__(self, integration_kwargs: dict | None = None, **kwargs):
        # ClosedFormModel uses closed-form solutions by default.
        # Only use numerical integration if explicitly requested.
        super().__init__(integration_kwargs=integration_kwargs, **kwargs)
        if integration_kwargs is None:
            # Override the default to use closed-form solution
            self.integration_config = None

    @abstractmethod
    def _compute_equilibrium_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w_prev: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute equilibrium weights given combined goal features.

        Args:
            params: Model parameters
            phi: Combined goal features (binary indicator of active features)
            w_prev: Weights at start of stage

        Returns:
            Equilibrium weights after this stage
        """
        ...

    def _compute_loss_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute scalar loss given combined goal features.

        Override this method to use autodiff-based gradient computation.
        The gradient will be computed as: -theta @ jax.grad(loss)(w)

        Args:
            params: Model parameters
            phi: Combined goal features
            w: Current weights

        Returns:
            Scalar loss value
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either "
            "_compute_loss_from_phi or _compute_gradient_from_phi"
        )

    def _compute_gradient_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute weight gradient given combined goal features.

        Default implementation uses autodiff on _compute_loss_from_phi.
        Override for explicit gradient computation (e.g., RWModel).

        Args:
            params: Model parameters
            phi: Combined goal features
            w: Current weights

        Returns:
            Gradient dw/dt
        """
        theta = params['theta']

        def loss_fn(w_):
            return self._compute_loss_from_phi(params, phi, w_)

        grad_w = jax.grad(loss_fn)(w)
        # Negative gradient for descent, with saliency metric
        return -theta @ grad_w

    def _combine_goal_features_batched(
        self,
        stage_goals: jnp.ndarray,
        stage_env_mask: jnp.ndarray,
        stage_goal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Combine all goal features from a stage (batched)."""
        n_features = stage_goals.shape[-1]
        combined_mask = stage_env_mask[:, None] * stage_goal_mask
        masked_goals = stage_goals * combined_mask[:, :, None]
        return jnp.max(masked_goals.reshape(-1, n_features), axis=0)

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
        phi = self._combine_goal_features_batched(stage_goals, stage_env_mask, stage_goal_mask)
        return self._compute_gradient_from_phi(params, phi, w)

    def _get_compute_stage_equilibrium_batched_fn(
        self,
        params: dict[str, jnp.ndarray],
    ) -> Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        jnp.ndarray
    ]:
        if self.integration_config is None:
            def fn(
                w_prev: jnp.ndarray,
                stage_goals: jnp.ndarray,
                stage_distractors: jnp.ndarray,
                stage_env_mask: jnp.ndarray,
                stage_goal_mask: jnp.ndarray,
                stage_distractor_mask: jnp.ndarray,
                stage_active: jnp.ndarray,
            ):
                phi = self._combine_goal_features_batched(stage_goals, stage_env_mask, stage_goal_mask)
                has_features = jnp.sum(phi) > 0
                w_new = self._compute_equilibrium_from_phi(params, phi, w_prev)

                return jnp.where(stage_active * has_features, w_new, w_prev)

            return fn

        else:
            if not self.has_warned_about_integration:
                log.warning(f"{self.name} has closed-form solution but numerical integration requested.")
                self.has_warned_about_integration = True
            return super()._get_compute_stage_equilibrium_batched_fn(params)


class RWModel(SaliencyModelMixin, ClosedFormModel):
    """
    Rescorla-Wagner model with saliency-weighted feature learning.

    This model uses an explicit learning rule rather than gradient descent on a loss,
    so it overrides _compute_gradient_from_phi directly.

    Equilibrium: w* = w_prev + (1 - w_prev . phi) / ||s||_1 . s
    where s = theta @ phi and phi is the combined goal features.
    """

    @property
    def name(self) -> str:
        return "Rescorla-Wagner"

    def get_hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            BETA_SPEC,
            *self._get_saliency_specs(),
        ]

    def compute_regularisation_loss(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return self._compute_saliency_regularisation_loss(params)

    def _compute_equilibrium_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w_prev: jnp.ndarray,
    ) -> jnp.ndarray:
        theta = params['theta']

        s = theta @ phi
        s_sum = jnp.sum(s)

        w_active_sum = jnp.sum(w_prev * phi)
        update_scaling = (1.0 - w_active_sum) / jnp.maximum(s_sum, EPSILON)

        return w_prev + s * update_scaling

    def _compute_gradient_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """Explicit Rescorla-Wagner gradient (not derived from a loss function)."""
        theta = params['theta']
        s = theta @ phi
        prediction_error = 1.0 - jnp.dot(w, phi)
        return s * prediction_error


class KLModel(SaliencyModelMixin, ClosedFormModel):
    """
    KL-divergence gradient descent model with saliency metric.

    Uses autodiff to compute gradients from the KL loss.

    Equilibrium: w* = alpha* . theta @ phi + w_perp
    where alpha* = (logit(q) - w_perp . phi) / (phi^T @ theta @ phi)
    """

    @property
    def name(self) -> str:
        return "KL-Divergence"

    def get_hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            Q_SPEC,
            *self._get_saliency_specs(),
        ]

    def compute_regularisation_loss(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return self._compute_saliency_regularisation_loss(params)

    def _compute_equilibrium_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w_prev: jnp.ndarray,
    ) -> jnp.ndarray:
        theta = params['theta']
        logit_q = params['log_q']

        d = theta @ phi
        d_norm_sq = jnp.dot(d, d)
        phi_theta_phi = jnp.dot(phi, d)

        alpha_prev = jnp.dot(w_prev, d) / jnp.maximum(d_norm_sq, EPSILON)
        w_perp = w_prev - alpha_prev * d
        w_perp_dot_phi = jnp.dot(w_perp, phi)

        alpha_star = (logit_q - w_perp_dot_phi) / jnp.maximum(phi_theta_phi, EPSILON)

        return alpha_star * d + w_perp

    def _compute_loss_from_phi(
        self,
        params: dict[str, jnp.ndarray],
        phi: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """KL divergence between target q and policy sigma(w . phi)."""
        q = jax.nn.sigmoid(params['log_q'])
        sigma = jax.nn.sigmoid(jnp.dot(w, phi))

        # KL(target || policy) = q*log(q/sigma) + (1-q)*log((1-q)/(1-sigma))
        return (
            q * jnp.log(q / (sigma + EPSILON) + EPSILON) +
            (1 - q) * jnp.log((1 - q) / (1 - sigma + EPSILON) + EPSILON)
        )
