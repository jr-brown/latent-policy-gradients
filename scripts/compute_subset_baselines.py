"""Compute uniform and Dirichlet random baselines for data subsets used in Table 1."""
import json
import logging
import sys
import os
import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from src.preference_modelling.metrics import (
    compute_random_baseline_metrics, compute_uniform_baseline_metrics,
)
from src.preference_modelling.data import prepare_training_data, prepare_examples
from src.util import load_runs_from_cache


def ser(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [ser(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


# Load data (same as run_validation.py)
print('Loading data...', flush=True)
run_env_metrics, possible_goals = load_runs_from_cache(
    get_runs_from_wandb_kwargs={'name_regex_filter': 'maze_eval_'},
    cache_dir='local/cache', offline=True,
)
filtered = {k: v for k, v in run_env_metrics.items() if 'distill' in k and 'and' not in k}
agent_pipelines, feature_to_idx, all_features, n_features = prepare_training_data(
    run_env_metrics=filtered, possible_goals=possible_goals, include_no_goal_feature=False,
)
training_examples = prepare_examples(
    run_metrics_dict=filtered, agent_training_pipelines=agent_pipelines,
    feature_to_idx=feature_to_idx, n_features=n_features, include_no_goal_feature=False,
)
print(f'Data: {len(filtered)} agents, {len(training_examples)} examples', flush=True)

# Define subsets matching the validation.py filtering logic
subsets = {
    'all': training_examples,
    'multi_stage': [
        ex for ex in training_examples
        if '_then_' in ex['run_name'] or '_and_' in ex['run_name']
    ],
    'with_distractor': [
        ex for ex in training_examples
        if '_distractor' in ex['run_name']
    ],
}

# Compute baselines for each subset
results = {}
for subset_name, subset_examples in subsets.items():
    n = len(subset_examples)
    print(f'\n=== {subset_name} (n={n}) ===', flush=True)

    uniform = compute_uniform_baseline_metrics(subset_examples, directional_threshold=0.10)
    random = compute_random_baseline_metrics(subset_examples, directional_threshold=0.10)

    results[subset_name] = ser({
        'n_examples': n,
        'uniform': uniform,
        'random': random,
    })

    print(f'  Uniform 3-way KL: {uniform["3way_kl_mean"]:.4f}', flush=True)
    print(f'  Random  3-way KL: {random["3way_kl_mean"]:.4f}', flush=True)

# Print summary table for paper
print('\n\n=== SUMMARY FOR TABLE 1 ===', flush=True)
print(f'{"Subset":<20} {"Uniform KL":>12} {"Dirichlet KL":>14}', flush=True)
print('-' * 48, flush=True)
for subset_name, data in results.items():
    u_kl = data['uniform']['3way_kl_mean']
    r_kl = data['random']['3way_kl_mean']
    print(f'{subset_name:<20} {u_kl:>12.4f} {r_kl:>14.4f}', flush=True)

# Save results
out_path = 'local/plots/validation/subset_baselines.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out_path}', flush=True)
