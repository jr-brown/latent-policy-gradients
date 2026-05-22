import jax.numpy as jnp
import json
import logging
import os

from src.preference_modelling.models import ModelType, get_model
from src.preference_modelling.data_structures import TrainingPipeline, PaddedPipeline


log = logging.getLogger(__name__)

# Module constant for numerical stability
EPSILON = 1e-8


def get_features_in_training(
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    all_features: list[str],
) -> set[str]:
    """
    Determine which features are present in training pipelines.
    
    Args:
        agent_training_pipelines: Dict mapping agent names to (pipeline, padded_pipeline)
        all_features: Ordered list of all feature names
    
    Returns:
        Set of feature names that appear in training
    """
    n_features = len(all_features)
    features_in_training = set()
    
    for pipeline, _ in agent_training_pipelines.values():
        for stage in pipeline.stages:
            for env in stage.environments:
                for goal in env.goals:
                    for feature_idx in range(n_features):
                        if goal[feature_idx] > 0:
                            features_in_training.add(all_features[feature_idx])
                for distractor in env.distractors:
                    for feature_idx in range(n_features):
                        if distractor[feature_idx] > 0:
                            features_in_training.add(all_features[feature_idx])
    
    return features_in_training


def log_fitted_parameters(
    params: dict[str, jnp.ndarray],
    all_features: list[str],
    agent_training_pipelines: dict[str, tuple[TrainingPipeline, PaddedPipeline]],
    model_type: ModelType,
    model_kwargs: dict | None = None,
    save_name: str | None = None,
) -> None:
    """
    Log fitted model parameters in a structured format.
    
    Args:
        params: Fitted parameters in log-space
        all_features: List of all feature names
        agent_training_pipelines: Dict mapping agent names to (pipeline, padded_pipeline)
        agent_feature_weights: Dict mapping agent names to their feature weight arrays
        model_type: Which model formulation was used
        model_kwargs: Optional kwargs used for model construction
    """
    model_kwargs = model_kwargs or {}
    model = get_model(model_type, **model_kwargs)
    
    features_in_training = get_features_in_training(agent_training_pipelines, all_features)
    log.info(f"Fitted {model.name} model parameters:")

    model.log_hyperparameters(params, all_features, features_in_training)
    
    if save_name is not None:
        agent_parameters = model.learn_agent_parameters_to_be_saved(
            params,
            all_features,
            agent_training_pipelines,
        )
        output_dir = "local/per_agent_feature_weights"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{save_name}.json")
        
        with open(output_path, "w") as f:
            json.dump(agent_parameters, f, indent=2)
    
        log.info(f"Saved per-agent feature weights to {output_path}")

