"""Run post-hoc validation metrics for all relevant models, one at a time."""
import json
import logging
import sys
import os
import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from src.preference_modelling.metrics import (
    load_saved_params, compute_post_hoc_metrics,
    compute_random_baseline_metrics, compute_uniform_baseline_metrics,
)
from src.preference_modelling.data import prepare_training_data, prepare_examples
from src.preference_analysis import validate_elo_scores
from src.util import load_runs_from_cache

def ser(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [ser(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# Load data once
run_env_metrics, possible_goals = load_runs_from_cache(
    get_runs_from_wandb_kwargs={'name_regex_filter': 'maze_eval_'},
    cache_dir='local/cache', offline=True,
)
filtered = {k: v for k, v in run_env_metrics.items() if 'distill' in k and 'and' not in k}
# Drop agents whose every eval rate is (0, 0) — no valid matchups, Elo has no signal
_before = len(filtered)
filtered = {
    k: v for k, v in filtered.items()
    if any(r is not None and r != (0.0, 0.0) for r in v.values())
}
print(f'Dropped {_before - len(filtered)} agents with all-zero eval rates', flush=True)
agent_pipelines, feature_to_idx, all_features, n_features = prepare_training_data(
    run_env_metrics=filtered, possible_goals=possible_goals, include_no_goal_feature=False,
)
training_examples = prepare_examples(
    run_metrics_dict=filtered, agent_training_pipelines=agent_pipelines,
    feature_to_idx=feature_to_idx, n_features=n_features, include_no_goal_feature=False,
)
print(f'Data: {len(filtered)} agents, {len(training_examples)} examples', flush=True)

os.makedirs('local/plots/validation', exist_ok=True)
results_path = 'local/plots/validation/full_validation_results.json'

# Load existing results if any
if os.path.exists(results_path):
    with open(results_path) as f:
        output = json.load(f)
else:
    output = {'elo_validation': None, 'model_metrics': {}}

# Elo validation
if output['elo_validation'] is None:
    print('\n=== Elo 4-Fold CV ===', flush=True)
    elo_result = validate_elo_scores(
        run_env_metrics=filtered, possible_goals=possible_goals,
        n_folds=4, directional_threshold=0.10,
    )
    output['elo_validation'] = ser(elo_result)
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    s = elo_result['summary']
    print(f'  KL: {s["kl_mean"]:.4f}, TV: {s["tv_mean"]:.4f}, Brier: {s["brier_mean"]:.4f}, Dir: {s["directional_accuracy_mean"]:.2%}', flush=True)
else:
    print('Elo validation already done, skipping', flush=True)

# Models
models_to_eval = [
    ('val_multi_kl_full_dist', 'local/validation_sweep_results/merged_summaries.json'),
    ('val_lbr_simultaneous_full_dist', 'local/validation_sweep_results/merged_summaries.json'),
    ('val_lbr_diagonal_full_dist', 'local/validation_sweep_results/merged_summaries.json'),
    ('val_lbr_memoryless_full_dist', 'local/validation_sweep_results/merged_summaries.json'),
    # dq_lbr last — quadratic expansion uses ~500MB, can OOM locally
    ('val_dq_lbr_full_dist', 'local/validation_sweep_results/merged_summaries.json'),
    ('without_feature_full_dist', 'local/developmental_model_fit_summaries.json'),
]

# Baselines
if 'uniform_baseline' not in output['model_metrics']:
    print('\n=== uniform_baseline ===', flush=True)
    m = compute_uniform_baseline_metrics(training_examples, directional_threshold=0.10)
    output['model_metrics']['uniform_baseline'] = ser(m)
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  3way KL={m["3way_kl_mean"]:.4f}, TV={m["3way_tv_mean"]:.4f}, Dir={m["3way_directional_accuracy"]:.2%}', flush=True)
    print(f'  2way KL={m["2way_kl_mean"]:.4f}, TV={m["2way_tv_mean"]:.4f}, Dir={m["2way_directional_accuracy"]:.2%}', flush=True)
else:
    print('\nuniform_baseline: already done, skipping', flush=True)

if 'random_baseline' not in output['model_metrics']:
    print('\n=== random_baseline ===', flush=True)
    baseline_metrics = compute_random_baseline_metrics(
        training_examples, directional_threshold=0.10,
    )
    output['model_metrics']['random_baseline'] = ser(baseline_metrics)
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    m = baseline_metrics
    print(f'  3way KL={m["3way_kl_mean"]:.4f}, TV={m["3way_tv_mean"]:.4f}, Dir={m["3way_directional_accuracy"]:.2%}', flush=True)
    print(f'  2way KL={m["2way_kl_mean"]:.4f}, TV={m["2way_tv_mean"]:.4f}, Dir={m["2way_directional_accuracy"]:.2%}', flush=True)
else:
    print('\nrandom_baseline: already done, skipping', flush=True)

for model_key, path in models_to_eval:
    if model_key in output['model_metrics']:
        print(f'\n{model_key}: already done, skipping', flush=True)
        continue
    print(f'\n=== {model_key} ===', flush=True)
    try:
        params, model_type, model_kwargs = load_saved_params(path, model_key)
        model_kwargs['include_no_goal_feature'] = False
        result = compute_post_hoc_metrics(
            params=params, model_type=model_type, model_kwargs=model_kwargs,
            training_examples=training_examples, agent_training_pipelines=agent_pipelines,
            all_features=all_features, feature_to_idx=feature_to_idx, n_features=n_features,
            include_no_goal_feature=False,
        )
        output['model_metrics'][model_key] = ser(result['metrics'])
        # Save incrementally
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        m = result['metrics']
        print(f'  3way KL={m["3way_kl_mean"]:.4f}, TV={m["3way_tv_mean"]:.4f}, Dir={m["3way_directional_accuracy"]:.2%}', flush=True)
        print(f'  2way KL={m["2way_kl_mean"]:.4f}, TV={m["2way_tv_mean"]:.4f}, Dir={m["2way_directional_accuracy"]:.2%}', flush=True)
    except Exception as e:
        print(f'  FAILED: {e}', flush=True)
        import traceback; traceback.print_exc()

# Final summary
print('\n\n=== FINAL SUMMARY ===', flush=True)
for variant in ('3way', '2way'):
    print(f'\n{variant} metrics:', flush=True)
    print(f'{"Model":<35} {"KL":>8} {"TV":>8} {"Brier":>8} {"Dir%":>8}', flush=True)
    print('-' * 75, flush=True)
    for key, m in output['model_metrics'].items():
        da = m.get(f'{variant}_directional_accuracy', float('nan'))
        da_str = f'{da:.2%}' if not np.isnan(da) else 'N/A'
        print(f'{key:<35} {m[f"{variant}_kl_mean"]:>8.4f} {m[f"{variant}_tv_mean"]:>8.4f} {m[f"{variant}_brier_mean"]:>8.4f} {da_str:>8}', flush=True)

print('\nDone!', flush=True)
