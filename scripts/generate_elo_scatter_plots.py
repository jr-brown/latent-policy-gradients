"""Generate Elo vs Model Values scatter plots for all agents with paper-friendly labels."""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from src.preference_modelling.metrics import (
    load_saved_params, compute_per_agent_weights, compute_per_agent_per_goal_values,
)
from src.preference_modelling.data import prepare_training_data, prepare_examples
from src.preference_analysis import compute_elo_scores, plot_elo_vs_model_values
from src.util import load_runs_from_cache

# Load data (same filtering as run_validation.py)
print('Loading data...', flush=True)
run_env_metrics, possible_goals = load_runs_from_cache(
    get_runs_from_wandb_kwargs={'name_regex_filter': 'maze_eval_'},
    cache_dir='local/cache', offline=True,
)
filtered = {k: v for k, v in run_env_metrics.items() if 'distill' in k and 'and' not in k}

agent_pipelines, feature_to_idx, all_features, n_features = prepare_training_data(
    run_env_metrics=filtered, possible_goals=possible_goals, include_no_goal_feature=False,
)
print(f'Data: {len(filtered)} agents', flush=True)

# Load saved model params
model_key = 'val_multi_kl_full_dist'
params_path = 'local/validation_sweep_results/merged_summaries.json'
print(f'Loading model params from {params_path} ({model_key})...', flush=True)
params, model_type, model_kwargs = load_saved_params(params_path, model_key)
model_kwargs['include_no_goal_feature'] = False

# Compute per-agent weights by running model forward through each pipeline
print('Computing per-agent weights...', flush=True)
agent_weights = compute_per_agent_weights(
    params=params, model_type=model_type, model_kwargs=model_kwargs,
    all_features=all_features, agent_training_pipelines=agent_pipelines,
)
print(f'Computed weights for {len(agent_weights)} agents', flush=True)

# Compute per-agent per-goal model values
print('Computing per-agent per-goal values...', flush=True)
agent_goal_values = compute_per_agent_per_goal_values(
    params=params, model_type=model_type, model_kwargs=model_kwargs,
    agent_weights=agent_weights, possible_goals=possible_goals,
    feature_to_idx=feature_to_idx, n_features=n_features,
)

# Compute Elo scores for ALL agents
print('Computing Elo scores for all agents...', flush=True)
agent_elo_scores: dict[str, dict[str, float]] = {}
for run_name, env_metrics in filtered.items():
    if run_name in agent_weights:
        elo = compute_elo_scores(
            env_metrics=env_metrics,
            possible_goals=possible_goals,
        )
        elo.pop("no_goal", None)
        agent_elo_scores[run_name] = elo

print(f'Computed Elo for {len(agent_elo_scores)} agents', flush=True)

# Generate plots with paper-friendly labels
output_dir = 'local/plots/elo_vs_model'
print(f'Generating plots in {output_dir}...', flush=True)
results = plot_elo_vs_model_values(
    agent_elo_scores=agent_elo_scores,
    agent_goal_values=agent_goal_values,
    possible_goals=possible_goals,
    output_dir=output_dir,
    filename='elo_vs_model_values.png',
    model_name='Predicted',
    show_title=False,
)

print(f'\nResults:', flush=True)
for k, v in results.items():
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}', flush=True)

print('\nDone!', flush=True)
