"""
Preference modelling models package.

This package provides phenomenological models for preference evolution during training.
"""
from typing import Literal

from .base import (
    # Spec classes
    HyperparameterSpec,
    ScalarSpec,
    MatrixSpec,
    CompositeSpec,
    CustomSpec,
    ThetaSpec,
    # Shared specs
    Q_SPEC,
    INIT_VALUE_SPEC,
    BETA_SPEC,
    THETA_SPEC,
    # Base classes and utilities
    EPSILON,
    Int,
    PhenomenologicalModel,
    SimpleWeightsModel,
    NonLinearityMixin,
    PipelineModeMixin,
    SaliencyModelMixin,
    _compute_agent_params_from_pipeline_batched,
    _compute_agent_params_from_stage_equilibrium_batched,
    _transform_param,
    _format_param_value,
    _get_param_key,
    get_no_goal_features,
)

from .closed_form import (
    ClosedFormModel,
    RWModel,
    KLModel,
)

from .multi_kl import (
    MultiChoiceKLModel,
    DiagonalQuadraticMultiChoiceKLModel,
)

from .saliency import (
    ValueFunction,
    LinearValueFunction,
    LinearSaliencySpec,
    make_value_function,
)

# Private model implementations — gracefully absent on a public release. When
# the src/private/ tree is missing, the corresponding registry entries simply
# aren't added; get_model("sharded_multi_kl") etc. then raises the usual
# "Unknown model type" with the publicly-registered list.
_HAS_SHARDED = False
try:
    from src.private.preference_modelling.models.sharded import (
        ShardedAgentParams,
        ShardedMultiChoiceKLModel,
        GAMMA_SPEC,
        VALUE_SCALE_SPEC,
        ACTIVATION_WEIGHT_SCALE_SPEC,
        ACTIVATION_BIAS_SCALE_SPEC,
        ACTIVATION_WEIGHT_INIT_SCALE_SPEC,
    )
    _HAS_SHARDED = True
except ImportError:
    pass

_HAS_BILINEAR = False
try:
    from src.private.preference_modelling.models.bilinear import (
        BilinearValueFunction,
        BilinearSpec,
    )
    _HAS_BILINEAR = True
except ImportError:
    pass

_HAS_MLP = False
try:
    from src.private.preference_modelling.models.mlp_value_function import MLPValueFunction
    _HAS_MLP = True
except ImportError:
    pass


# Model type registry. ModelType keeps all known string keys so type-checkers
# accept any of them; runtime availability is gated by MODEL_REGISTRY membership.
ModelType = Literal[
    "rw", "kl", "multi_kl", "sharded_multi_kl", "contextual_sharded_multi_kl",
    "diagonal_quadratic_multi_kl",
]

MODEL_REGISTRY: dict[ModelType, type[PhenomenologicalModel]] = {
    "rw": RWModel,
    "kl": KLModel,
    "multi_kl": MultiChoiceKLModel,
    "diagonal_quadratic_multi_kl": DiagonalQuadraticMultiChoiceKLModel,
}
if _HAS_SHARDED:
    MODEL_REGISTRY["sharded_multi_kl"] = ShardedMultiChoiceKLModel
    MODEL_REGISTRY["contextual_sharded_multi_kl"] = ShardedMultiChoiceKLModel  # Alias


def get_model(model_type: ModelType, **model_kwargs) -> PhenomenologicalModel:
    """Get a weight evolution model instance by type with optional kwargs."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**model_kwargs)


__all__ = [
    # Registry and factory
    "get_model",
    "ModelType",
    "MODEL_REGISTRY",
    # Spec classes
    "HyperparameterSpec",
    "ScalarSpec",
    "MatrixSpec",
    "CompositeSpec",
    "CustomSpec",
    "ThetaSpec",
    # Shared specs (public)
    "Q_SPEC",
    "INIT_VALUE_SPEC",
    "BETA_SPEC",
    "THETA_SPEC",
    # Base classes and utilities
    "EPSILON",
    "Int",
    "PhenomenologicalModel",
    "SimpleWeightsModel",
    "NonLinearityMixin",
    "PipelineModeMixin",
    "SaliencyModelMixin",
    "_compute_agent_params_from_pipeline_batched",
    "_compute_agent_params_from_stage_equilibrium_batched",
    "_transform_param",
    "_format_param_value",
    "_get_param_key",
    "get_no_goal_features",
    # Closed-form models
    "ClosedFormModel",
    "RWModel",
    "KLModel",
    # Multi-choice KL models
    "MultiChoiceKLModel",
    "DiagonalQuadraticMultiChoiceKLModel",
    # Value functions (public)
    "ValueFunction",
    "LinearValueFunction",
    "LinearSaliencySpec",
    "make_value_function",
    # Private model implementations (MLPValueFunction, BilinearValueFunction +
    # BilinearSpec, ShardedMultiChoiceKLModel + GAMMA_SPEC etc.) live in
    # src/private/preference_modelling/models/ and are imported here on a
    # best-effort basis. Import them directly from src.private.* if needed.
]
