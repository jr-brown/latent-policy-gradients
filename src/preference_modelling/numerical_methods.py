import jax
import jax.numpy as jnp
import logging

from typing import TypeVar, Union
from jaxtyping import PyTree, Array, Float
from dataclasses import dataclass
from collections.abc import Callable


log = logging.getLogger(__name__)

EPSILON = 1e-8


# Restricted PyTree type
PT = TypeVar('PT', bound=Union[Float[Array, "..."], dict[str, Float[Array, "..."]]])


@dataclass
class IntegrationConfig:
    """Configuration for numerical integration of weight dynamics."""
    max_steps: int | None = 100
    convergence_tolerance: float | None = None
    dt: float = 0.01  # Time step for RK4
    
    def __post_init__(self):
        if self.max_steps is None and self.convergence_tolerance is None:
            raise ValueError(
                "At least one of max_steps or convergence_tolerance must be specified "
                "for numerical integration"
            )
    
    def should_stop(self, step: int, relative_change: float) -> bool:
        """Check if integration should stop based on configured criteria."""
        if self.max_steps is not None and step >= self.max_steps:
            return True
        if self.convergence_tolerance is not None and relative_change < self.convergence_tolerance:
            return True
        return False


def rk4_step(
    gradient_fn: Callable[[PT], PT],
    w: PT,
    dt: float,
) -> PT:
    """
    Perform a single RK4 integration step.
    
    Args:
        gradient_fn: Function computing dw/dt given current parameters
        w: Current parameters
        dt: Time step
    
    Returns:
        Updated parameters after one RK4 step
    """
    k1 = gradient_fn(w)
    k2 = gradient_fn(jax.tree.map(lambda wi, k1i: wi + 0.5 * dt * k1i, w, k1))
    k3 = gradient_fn(jax.tree.map(lambda wi, k2i: wi + 0.5 * dt * k2i, w, k2))
    k4 = gradient_fn(jax.tree.map(lambda wi, k3i: wi + dt * k3i, w, k3))
    return jax.tree.map(
        lambda wi, k1i, k2i, k3i, k4i: wi + (dt / 6.0) * (k1i + 2*k2i + 2*k3i + k4i),
        w, k1, k2, k3, k4
    )


def _pytree_norm(tree: PyTree) -> Array:
    """Compute the L2 norm of a PyTree of arrays."""
    leaves = jax.tree.leaves(tree)
    sum_sq = sum(jnp.sum(leaf ** 2) for leaf in leaves)
    return jnp.sqrt(sum_sq)


def _pytree_diff_norm(tree1: PT, tree2: PT) -> Array:
    """Compute the L2 norm of the difference between two PyTrees."""
    diff = jax.tree.map(lambda a, b: a - b, tree1, tree2)
    return _pytree_norm(diff)


def integrate_to_equilibrium(
    gradient_fn: Callable[[PT], PT],
    initial_state: PT,
    config: IntegrationConfig,
) -> tuple[PT, dict]:
    """
    Integrate dynamics to equilibrium using RK4.
    
    This is the non-JIT-compatible version that supports early stopping
    based on convergence tolerance. Use `integrate_fixed_steps` for
    JIT-compatible integration.
    
    Args:
        gradient_fn: Function computing d(state)/dt given current state
        initial_state: Starting state (PyTree of arrays)
        config: Integration configuration
    
    Returns:
        Tuple of:
            - Final state
            - Metadata dict containing:
                - steps_taken: int
                - converged: bool
                - final_relative_change: float
    """
    w = initial_state
    
    steps_taken = 0
    convergence_steps = None
    relative_change = float('inf')
    
    max_steps = config.max_steps if config.max_steps is not None else 1000000
    
    for step in range(max_steps):
        w_new = rk4_step(gradient_fn, w, config.dt)
        
        # Compute relative change using PyTree norms
        w_new_norm = _pytree_norm(w_new)
        if w_new_norm > EPSILON:
            relative_change = float(_pytree_diff_norm(w_new, w) / w_new_norm)
        else:
            relative_change = float(_pytree_diff_norm(w_new, w))
        
        w = w_new
        steps_taken = step + 1

        if config.convergence_tolerance is not None and relative_change < config.convergence_tolerance:
            convergence_steps = steps_taken
        
        if config.should_stop(steps_taken, relative_change):
            break
    
    # Log convergence information
    log.debug(
        f"Integration completed: {steps_taken=}, {convergence_steps=}, "
        f"final_relative_change={relative_change:.2e}"
    )
    
    if convergence_steps is None and config.convergence_tolerance is not None:
        log.warning(
            f"Integration did not converge after {steps_taken} steps. "
            f"Final relative change: {relative_change:.2e}, "
            f"tolerance: {config.convergence_tolerance:.2e}"
        )
    
    return w, {
        'steps_taken': steps_taken,
        'convergence_steps': convergence_steps,
        'final_relative_change': relative_change,
    }

def integrate_fixed_steps(
    gradient_fn: Callable[[PT], PT],
    initial_state: PT,
    max_steps: int,
    dt: float,
) -> PT:
    """
    Integrate dynamics for a fixed number of steps using RK4.
    
    This is the JIT-compatible version for use in batched operations.
    Uses `jax.lax.fori_loop` for efficient compilation.
    
    Args:
        gradient_fn: Function computing d(state)/dt given current state
        initial_state: Starting state (PyTree of arrays)
        max_steps: Number of integration steps
        dt: Time step
    
    Returns:
        Final state after integration
    """
    def body_fn(_, w):
        return rk4_step(gradient_fn, w, dt)
    
    return jax.lax.fori_loop(0, max_steps, body_fn, initial_state)

def integrate_stage_batched(
    gradient_fn: Callable[[PT], PT],
    initial_state: PT,
    stage_active: Array,
    config: IntegrationConfig,
) -> PT:
    """
    Integrate a single stage with conditional execution for batched operations.
    
    This is a JIT-compatible helper that integrates only if the stage is active.
    
    Args:
        gradient_fn: Function computing d(state)/dt given current state
        initial_state: Starting state (PyTree of arrays)
        stage_active: Scalar indicating if this stage should be processed
        config: Integration configuration (uses max_steps and dt)
    
    Returns:
        Final state (unchanged if stage_active is 0)
    """
    max_steps = config.max_steps if config.max_steps is not None else 1000

    def do_integrate(w):
        return integrate_fixed_steps(gradient_fn, w, max_steps, config.dt)

    return jax.lax.cond(
        stage_active > 0,
        do_integrate,
        lambda w: w,
        initial_state
    )
