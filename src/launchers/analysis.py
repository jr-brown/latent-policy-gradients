import re
import math
import numpy as np
import jax.numpy as jnp
import wandb
import logging

from tqdm import tqdm
from typing import Literal, Any
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_scaffold.file_io import transform

from src.util import Kwargs, load_runs_from_cache
from src.preference_analysis import (
    compute_btl_scores,
    compute_elo_scores,
    compute_shape_colour_preferences,
    plot_scores,
    plot_run_env_metrics,
    plot_shape_colour_preferences,
    plot_grouped_preference_comparison,
    plot_feature_saliency_matrix,
    plot_stage_comparison,
    validate_elo_scores,
    plot_elo_vs_model_values,
)
from src.two_stage_analysis import analyze_two_stage_pipelines
from src.preference_modelling.data import (
    prepare_training_data,
    prepare_examples,
    examples_to_batch,
)
from src.preference_modelling.loss import create_loss_functions, compute_data_loss_batched
from src.preference_modelling.models import ModelType
from src.preference_modelling.logging import log_fitted_parameters
from src.preference_modelling.training import(
    fit_per_agent_per_feature_baseline, fit_per_agent_per_goal_baseline, get_random_choice_baseline,
    train_preference_model, 
)
from src.preference_modelling.validation import (
    compute_all_training_losses,
    perform_env_holdout_validation,
    perform_no_distractor_to_distractor_validation,
    perform_single_stage_to_multi_stage_validation,
    perform_kfold_agent_validation,
    perform_few_shot_validation,
    perform_few_shot_sweep_validation,
    create_env_holdout_splits,
)


log = logging.getLogger(__name__)


ScoreType = Literal["ELO", "BTL"]


def analyse_agent_preferences(
    get_runs_from_wandb_kwargs: Kwargs,
    cache_dir: str = "local/cache",
    max_workers: int = 10,
    score_type: ScoreType = "ELO",
    plot_per_run_scores: bool = True,
    plot_feature_saliency: bool = True,
    run_categories: dict[str, dict[str, list[str] | bool]] | None = None,
    comparison_groups: dict[str, dict] | None = None,
    zero_mean_scores: bool = False,
    two_stage_plot_config: dict | None = None,
    show_title: bool = True,
    max_prefs_per_plot: int | None = None,
):
    """
    Analyse agent preferences from evaluation runs.

    Args:
        get_runs_from_wandb_kwargs: Kwargs for fetching runs from wandb
        cache_dir: Directory for caching run metrics
        max_workers: Number of parallel workers
        score_type: Type of score to compute ("ELO" or "BTL")
        plot_per_run_scores: Whether to plot per-run score charts
        plot_feature_saliency: Whether to plot feature saliency matrix for single-stage runs
        run_categories: Dictionary mapping category names to dicts with keys:
            - required: list of keywords that must be in run name
            - excluded: list of keywords that must not be in run name
            - group_finetune: whether to group by finetune target
            If None, uses default categories.
        comparison_groups: Dictionary mapping group names to dicts with keys:
            - patterns: list of run name patterns to compare (required)
            - features_to_show: list of features to display (optional)
            Each group will produce a grouped bar chart comparing shape/colour preferences.
        show_title: Whether to display titles on plots. Set to False for figures with captions.
        max_prefs_per_plot: Maximum number of preferences to show per plot. If set, plots
            with more preferences will be split into multiple files with numbered suffixes.
            If None (default), all preferences are shown on a single plot.
    """
    run_env_metrics, possible_goals = load_runs_from_cache(
        get_runs_from_wandb_kwargs=get_runs_from_wandb_kwargs,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
    
    run_env_metrics = {
        run_name: env_metrics
        for run_name, env_metrics in run_env_metrics.items()
        # if "distractor" not in run_name and "distill" in run_name and "and" not in run_name
        if "distill" in run_name and "and" not in run_name
    }
    # Separate parallel computation from sequential plotting
    def compute_run_scores(run_name_and_metrics):
        run_name, env_metrics = run_name_and_metrics
        log.debug(f"Computing {score_type} scores for run: {run_name}")

        if score_type == "ELO":
            scores = compute_elo_scores(
                env_metrics=env_metrics,
                possible_goals=possible_goals,
            )
        elif score_type == "BTL":
            scores = compute_btl_scores(
                env_metrics=env_metrics,
                possible_goals=possible_goals,
            )
        else:
            raise ValueError(f"Unknown score type: {score_type}")

        colour_shape_prefs = compute_shape_colour_preferences(
            scores,
            zero_mean=zero_mean_scores,
        )

        run_name = re.sub("maze_eval_", "", run_name)
        run_name = re.sub("_distill", "", run_name)

        return run_name, env_metrics, scores, colour_shape_prefs

    log.info(f"Computing scores for {len(run_env_metrics)} runs")
    run_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_run_scores, item): item[0] 
            for item in run_env_metrics.items()
        }
        
        with tqdm(total=len(run_env_metrics), desc="Computing scores") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    run_results.append(result)
                except Exception as e:
                    run_name = futures[future]
                    log.error(f"Error computing scores for {run_name}: {e}")
                pbar.update(1)

    run_colour_shape_prefs = {}
    
    # Sequential plotting to avoid matplotlib thread-safety issues
    if plot_per_run_scores:
        log.info(f"Generating plots for {len(run_results)} runs")
        for run_name, env_metrics, scores, colour_shape_prefs in tqdm(run_results, desc="Generating plots"):
            run_colour_shape_prefs[run_name] = colour_shape_prefs
            try:
                plot_run_env_metrics(
                    run_name=run_name,
                    env_metrics=env_metrics,
                    possible_goals=possible_goals,
                    show_title=show_title,
                )
                plot_scores(
                    run_name=run_name,
                    scores=scores,
                    possible_goals=possible_goals,
                    score_type=score_type,
                    use_gridworld_colours=True,
                    output_dir=f"local/plots/{score_type.lower()}_scores",
                    show_title=show_title,
                )
            except Exception as e:
                log.error(f"Error plotting results for {run_name}: {e}")
    else:
        # Still need to populate run_colour_shape_prefs even if not plotting
        for run_name, env_metrics, scores, colour_shape_prefs in run_results:
            run_colour_shape_prefs[run_name] = colour_shape_prefs

    # Plot feature saliency matrix for single-stage, no-distractor runs
    if plot_feature_saliency:
        single_stage_no_distractor_prefs = {
            run_name: prefs
            for run_name, prefs in run_colour_shape_prefs.items()
            if "then" not in run_name and "distractor" not in run_name
        }
        if single_stage_no_distractor_prefs:
            plot_feature_saliency_matrix(
                run_preferences=single_stage_no_distractor_prefs,
                filename="feature_saliency_single_stage.png",
                show_title=show_title,
            )

    def filter_runs(
        run_colour_shape_prefs: dict[str, tuple[dict[str, float], dict[str, float]]],
        required_keywords: list[str],
        excluded_keywords: list[str],
    ) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
        """Filter runs based on required and excluded keywords in run name."""
        return {
            run_name: prefs
            for run_name, prefs in run_colour_shape_prefs.items()
            if all(keyword in run_name for keyword in required_keywords)
            and all(keyword not in run_name for keyword in excluded_keywords)
        }

    def group_finetune_runs_by_target(
        finetune_prefs: dict[str, tuple[dict[str, float], dict[str, float]]]
    ) -> dict[str, dict[str, tuple[dict[str, float], dict[str, float]]]]:
        """
        Group finetune runs by the goal they were finetuned on.
        E.g., black_diamond_distill_then_red_cross goes in red_cross group.
        """
        grouped_prefs: dict[str, dict[str, tuple[dict[str, float], dict[str, float]]]] = {}

        for run_name, prefs in finetune_prefs.items():
            match = re.search(r"then_([a-z]+_[a-z]+)", run_name)
            if match:
                finetune_on_goal = match.group(1)
                if finetune_on_goal not in grouped_prefs:
                    grouped_prefs[finetune_on_goal] = {}
                grouped_prefs[finetune_on_goal][run_name] = prefs
            else:
                log.warning(f"Could not determine finetune target from run name: {run_name}")

        return grouped_prefs

    def plot_preferences_paginated(
        run_preferences: dict[str, tuple[dict[str, float], dict[str, float]]],
        base_filename: str,
        max_prefs: int | None = None,
    ) -> None:
        """
        Plot shape/colour preferences with optional pagination.

        If max_prefs is set, divides preferences into blocks and saves
        with numbered suffixes. Otherwise calls the plotting function directly.

        Args:
            run_preferences: Dictionary of run preferences to plot
            base_filename: Base filename for output
            max_prefs: Maximum preferences per plot (overrides top-level max_prefs_per_plot)
        """
        effective_max = max_prefs if max_prefs is not None else max_prefs_per_plot
        if effective_max is None or len(run_preferences) <= effective_max:
            plot_shape_colour_preferences(
                run_preferences=run_preferences,
                filename=base_filename,
                show_title=show_title,
            )
            return

        # Split filename into name and extension
        if '.' in base_filename:
            name_part, ext = base_filename.rsplit('.', 1)
        else:
            name_part = base_filename
            ext = 'png'

        # Divide into blocks
        run_names = list(run_preferences.keys())
        n_runs = len(run_names)
        n_blocks = (n_runs + effective_max - 1) // effective_max

        for block_idx in range(n_blocks):
            start_idx = block_idx * effective_max
            end_idx = min((block_idx + 1) * effective_max, n_runs)

            block_run_names = run_names[start_idx:end_idx]
            block_prefs = {name: run_preferences[name] for name in block_run_names}

            block_filename = f"{name_part}_{block_idx + 1}.{ext}"

            plot_shape_colour_preferences(
                run_preferences=block_prefs,
                filename=block_filename,
                show_title=show_title,
            )

    def process_run_category(
        save_prefix: str,
        *,
        required: list[str] | None = None,
        excluded: list[str] | None = None,
        group_finetune: bool = False,
        max_prefs_per_plot: int | None = None,
    ) -> None:
        """
        Process a single run category and generate preference plots.

        Args:
            save_prefix: Prefix for output filenames
            required: Keywords that must be in run name (default: [])
            excluded: Keywords that must not be in run name (default: [])
            group_finetune: Whether to group by finetune target
            max_prefs_per_plot: Maximum preferences per plot (splits into multiple files if exceeded)
        """
        required = required if required is not None else []
        excluded = excluded if excluded is not None else []

        filtered_prefs = filter_runs(run_colour_shape_prefs, required, excluded)

        if len(filtered_prefs) == 0:
            return

        if group_finetune:
            # Group by finetune target and plot each group
            grouped_prefs = group_finetune_runs_by_target(filtered_prefs)
            for finetune_goal, prefs in grouped_prefs.items():
                plot_preferences_paginated(
                    run_preferences=prefs,
                    base_filename=f"{save_prefix}_{finetune_goal}_shape_colour_preferences.png",
                    max_prefs=max_prefs_per_plot,
                )
        else:
            # Plot all runs together
            plot_preferences_paginated(
                run_preferences=filtered_prefs,
                base_filename=f"{save_prefix}_shape_colour_preferences.png",
                max_prefs=max_prefs_per_plot,
            )

    # Process each category
    if run_categories:
        log.info(f"Generating shape and colour preference plots")
        for save_prefix, category_config in run_categories.items():
            try:
                process_run_category(save_prefix, **category_config)
            except TypeError as e:
                raise TypeError(f"Invalid config for category '{save_prefix}': {e}") from e

    # Generate comparison group plots
    if comparison_groups:
        log.info(f"Generating {len(comparison_groups)} comparison group plots")
        for group_name, group_config in comparison_groups.items():
            # Extract patterns and optional features_to_show
            patterns = group_config.get('patterns', [])
            features_to_show = group_config.get('features_to_show', None)

            # Find runs matching patterns
            matched_runs: dict[str, tuple[dict[str, float], dict[str, float]]] = {
                run_name: prefs
                for run_name, prefs in run_colour_shape_prefs.items()
                if any(re.search(pattern, run_name) for pattern in patterns)
            }

            if len(matched_runs) < 2:
                log.warning(f"Comparison group '{group_name}' has fewer than 2 matching runs, skipping")
                continue

            try:
                plot_grouped_preference_comparison(
                    run_preferences=matched_runs,
                    group_name=group_name,
                    output_dir="local/plots/preference_comparisons",
                    show_title=show_title,
                    features_to_show=features_to_show,
                )
            except Exception as e:
                log.error(f"Error plotting comparison group '{group_name}': {e}")

    # Generate two-stage pipeline analysis plots
    if two_stage_plot_config:
        log.info("Generating two-stage pipeline analysis plots")

        # Filter to two-stage pipelines only
        two_stage_prefs = {
            run_name: prefs
            for run_name, prefs in run_colour_shape_prefs.items()
            if "_then_" in run_name and "_distractor" not in run_name
        }

        if len(two_stage_prefs) > 0:
            log.info(f"Found {len(two_stage_prefs)} two-stage pipeline runs")
            try:
                # Merge show_title into the config if not already specified
                config_with_title = {**two_stage_plot_config}
                if 'show_title' not in config_with_title:
                    config_with_title['show_title'] = show_title

                analyze_two_stage_pipelines(
                    run_preferences=two_stage_prefs,
                    plot_config=config_with_title,
                    output_dir="local/plots/two_stage_analysis",
                )
            except Exception as e:
                log.error(f"Error generating two-stage pipeline plots: {e}")
        else:
            log.warning("No two-stage pipeline runs found for analysis")

    # Generate one-stage vs two-stage comparison plot
    log.info("Generating one-stage vs two-stage comparison plot")
    try:
        plot_stage_comparison(
            run_preferences=run_colour_shape_prefs,
            output_dir="local/plots/stage_comparison",
            show_title=show_title,
        )
    except Exception as e:
        log.error(f"Error generating stage comparison plot: {e}")


def fit_developmental_model(
    get_runs_from_wandb_kwargs: Kwargs,
    model_type: ModelType,
    model_kwargs: dict | None = None,
    cache_dir: str = "local/cache",
    max_workers: int = 10,
    learning_rate: float = 0.003,
    num_epochs: int = 3,
    log_interval: int = 100,
    compute_baselines: bool = False,
    baselines_epoch_scaling: int = 1,
    holdout_env_validation: bool = False,
    single_stage_to_multi_stage_validation: bool = False,
    no_distractor_to_distractor_validation: bool = False,
    kfold_agent_validation: bool = False,
    env_holdout_fraction: float = 0.1,
    env_holdout_seed: int = 42,
    agent_kfold_splits: int = 4,
    agent_kfold_seed: int = 42,
    few_shot_kwargs: dict | None = None,
    few_shot_sweep_kwargs: dict | None = None,
    save_name: str | None = None,
    no_goal_mode: str = "unweighted_ignore_no_goal",
    include_no_goal_feature: bool = False,
    offline: bool = False,
    skip_main_training: bool = False,
) -> dict[str, Any]:
    """
    Fit a Boltzmann-rational model with Rescorla-Wagner feature learning using JAX.

    The model assumes agents have values over goal features (shape and colour),
    where the value for pursuing a goal is the sum of its feature values.
    Feature values are derived from multi-stage training pipelines based on feature saliency.
    
    Args:
        model_type: Which model formulation to use:
            - "rw": Rescorla-Wagner learning rule with saliency-weighted updates
            - "kl": KL-divergence gradient descent with saliency metric
            - "multi_kl": Multi-choice KL with distractors (requires numerical integration)
        model_kwargs: Optional kwargs for model construction (e.g., n_shards for sharded models)
        get_runs_from_wandb_kwargs: Kwargs for fetching runs from wandb
        cache_dir: Directory for caching run metrics
        max_workers: Number of parallel workers for data fetching
        learning_rate: Learning rate for Adam optimiser
        num_epochs: Number of optimisation epochs
        log_interval: How often to log progress (in batches)
        saliency_scale_regularisation: Weight for regularisation term that constrains
            average saliency to be near 1.0
        kfold_agent_validation: Whether to perform K-fold cross-validation across agents
        holdout_env_validation: Whether to hold out random environments for validation
        env_holdout_fraction: Fraction of environments to hold out per agent (for holdout_env)
        env_holdout_seed: Random seed for environment holdout
        agent_kfold_splits: Number of folds for K-fold cross-validation
        agent_kfold_seed: Random seed for K-fold splits
        few_shot_kwargs: If provided, runs few-shot validation with these kwargs:
            - n_train_agents: Number of agents to sample for training (default: 3)
            - n_trials: Number of random trials to run (default: 10)
            - seed: Random seed for reproducibility (default: 42)
        few_shot_sweep_kwargs: If provided, runs few-shot sweep validation with these kwargs:
            - agent_counts: List of training set sizes to try (default: powers of 2)
            - n_trials_per_size: Number of trials per training set size (default: 5)
            - seed: Random seed for reproducibility (default: 42)
        include_no_goal_feature: If True (default), includes a dedicated 'no_goal' feature
            as the last dimension. When computing no-goal values in three-way loss, uses
            [0,...,0,1]. If False, uses zeros for no-goal features (original behavior).

    Returns:
        Fitted model parameters as a dict with keys:
            - 'params': Fitted model parameters
            - 'model_type': Which model formulation was used
            - 'model_kwargs': Model construction kwargs used
            - 'kfold_agent_val_results': list of validation result dicts per fold (if kfold_agent)
            - 'env_holdout_val_result': validation result dict for held-out environments (if holdout_env)
    
    Raises:
        ValueError: If no valid runs, pipelines, or training data found
    """
    model_kwargs = model_kwargs or {}

    # Load and prepare data
    run_env_metrics, possible_goals = load_runs_from_cache(
        get_runs_from_wandb_kwargs=get_runs_from_wandb_kwargs,
        cache_dir=cache_dir,
        max_workers=max_workers,
        offline=offline,
    )
    
    filtered_run_metrics = {
        run_name: env_metrics
        for run_name, env_metrics in run_env_metrics.items()
        # if "distractor" not in run_name and "distill" in run_name and "and" not in run_name
        if "distill" in run_name and "and" not in run_name
    }
    
    log.info(f"Fitting {model_type} model on {len(filtered_run_metrics)} runs")
    if model_kwargs:
        log.info(f"  Model kwargs: {model_kwargs}")
    
    if len(filtered_run_metrics) == 0:
        raise ValueError("No valid runs found to fit model")
    
    log.info("Preparing training pipelines and features")
    agent_training_pipelines, feature_to_idx, all_features, n_features = prepare_training_data(
        run_env_metrics=filtered_run_metrics,
        possible_goals=possible_goals,
        include_no_goal_feature=include_no_goal_feature,
    )
    
    if len(agent_training_pipelines) == 0:
        raise ValueError("No valid training pipelines found")

    # Merge include_no_goal_feature into model_kwargs
    effective_model_kwargs = {**model_kwargs, 'include_no_goal_feature': include_no_goal_feature}

    # Create loss functions
    loss_fn, data_loss_fn, per_example_loss_fn = create_loss_functions(
        model_type=model_type,
        model_kwargs=effective_model_kwargs,
        no_goal_mode=no_goal_mode,
    )

    # Prepare training data
    log.info("Preparing training examples")
    training_examples = prepare_examples(
        run_metrics_dict=filtered_run_metrics,
        agent_training_pipelines=agent_training_pipelines,
        feature_to_idx=feature_to_idx,
        n_features=n_features,
        include_no_goal_feature=include_no_goal_feature,
    )

    if len(training_examples) == 0:
        raise ValueError("No valid training data found")

    log.info("Batching training examples")
    batch_data = examples_to_batch(training_examples, no_goal_mode=no_goal_mode)
    log.info(f"Prepared {len(training_examples)} training examples")
    
    # Compute baselines if requested
    per_goal_baseline_loss = None
    per_feature_baseline_loss = None
    random_choice_baseline_loss = None
    
    if compute_baselines:
        log.info("="*30)
        log.info("Computing baselines on model performance")
        log.info("="*30)

        # Per-agent per-goal baseline (theoretical best, no feature decomposition)
        _, per_goal_baseline_loss = fit_per_agent_per_goal_baseline(
            training_examples=training_examples,
            no_goal_mode=no_goal_mode,
            include_no_goal_feature=include_no_goal_feature,
            learning_rate=learning_rate,
            num_epochs=num_epochs * baselines_epoch_scaling,
        )
        log.info(f"Per-agent per-goal baseline:")
        log.info(f"  Loss: {per_goal_baseline_loss:.6f}")

        # Per-agent per-feature baseline (best with feature decomposition)
        _, per_feature_baseline_loss = fit_per_agent_per_feature_baseline(
            training_examples=training_examples,
            n_features=n_features,
            no_goal_mode=no_goal_mode,
            include_no_goal_feature=include_no_goal_feature,
            learning_rate=learning_rate,
            num_epochs=num_epochs * baselines_epoch_scaling,
        )
        log.info(f"Per-agent per-feature baseline:")
        log.info(f"  Loss: {per_feature_baseline_loss:.6f}")

        # Random choice baseline
        random_choice_baseline_loss = get_random_choice_baseline(
            training_examples=training_examples,
            no_goal_mode=no_goal_mode,
        )
        log.info(f"Random choice baseline:")
        log.info(f"  Loss: {random_choice_baseline_loss:.6f}")
    
    # Train final model on all data
    params = None
    final_loss = None
    agent_train_losses = {}
    stratified_losses = {}

    if skip_main_training:
        log.info("="*30)
        log.info("Skipping main model training (skip_main_training=True)")
        log.info("="*30)
    else:
        log.info("="*30)
        log.info(f"Training {model_type} model with {num_epochs} epochs on all data")
        log.info("="*30)

        params = train_preference_model(
            train_dataset=batch_data,
            n_features=n_features,
            loss_fn=loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            show_progress=True,
            log_interval=log_interval,
        )

        log_fitted_parameters(
            params=params,
            all_features=all_features,
            agent_training_pipelines=agent_training_pipelines,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            save_name=save_name,
        )

        # Compute all training losses in a single pass (overall, per-agent, and stratified)
        log.info("Computing training losses...")
        final_loss, agent_train_losses, stratified_losses = compute_all_training_losses(
            params=params,
            training_examples=training_examples,
            agent_training_pipelines=agent_training_pipelines,
            per_example_loss_fn=per_example_loss_fn,
            no_goal_mode=no_goal_mode,
        )

        sorted_losses = sorted(agent_train_losses.items(), key=lambda x: x[1])
        per_agent_loss_log_str = "Per-Agent Training Losses:\n"

        best_table = [[name, f"{loss:.6f}"] for name, loss in sorted_losses[:5]]
        per_agent_loss_log_str += f"{tabulate(best_table, headers=['Best Agents', 'Loss'], tablefmt='simple_outline')}\n"

        worst_table = [[name, f"{loss:.6f}"] for name, loss in sorted_losses[-5:]]
        per_agent_loss_log_str += f"{tabulate(worst_table, headers=['Worst Agents', 'Loss'], tablefmt='simple_outline')}\n"

        log.info(per_agent_loss_log_str)

        # Report stratified training losses
        stratified_table = [
            [category_name, f"{category_loss:.6f}", f"{n_examples}"]
            for category_name, (category_loss, n_examples) in stratified_losses.items()
        ]
        log.info(f"Stratified Training Loss by Run Type:\n{tabulate(stratified_table, headers=['Category', 'Loss', 'Examples'], tablefmt='simple_outline')}")
        log.info("="*30)
        log.info("Model Performance Summary")
        log.info("="*30)

        if compute_baselines:
            comparison_table = [
                [f"{model_type} model", f"{final_loss:.6f}"],
                ["Per-agent per-goal", f"{per_goal_baseline_loss:.6f}"],
                ["Per-agent per-feature", f"{per_feature_baseline_loss:.6f}"],
                ["Random choice", f"{random_choice_baseline_loss:.6f}"],
            ]
        else:
            comparison_table = [
                [f"{model_type} model", f"{final_loss:.6f}"],
            ]
        log.info(f"\n{tabulate(comparison_table, headers=['Model', 'Loss'], tablefmt='simple_outline')}")

    # Perform validation if requested
    env_holdout_val_result = None
    single_stage_to_multi_stage_val_result = None
    no_distractor_to_distractor_val_result = None
    kfold_agent_val_results = []
    
    if holdout_env_validation:
        # Create holdout splits
        env_holdout_sets = create_env_holdout_splits(
            filtered_run_metrics=filtered_run_metrics,
            agent_training_pipelines=agent_training_pipelines,
            env_holdout_fraction=env_holdout_fraction,
            env_holdout_seed=env_holdout_seed,
        )
        
        env_holdout_val_result = perform_env_holdout_validation(
            training_examples=training_examples,
            env_holdout_sets=env_holdout_sets,
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            no_goal_mode=no_goal_mode,
        )

    if single_stage_to_multi_stage_validation:
        single_stage_to_multi_stage_val_result = perform_single_stage_to_multi_stage_validation(
            all_examples=training_examples,
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            all_features=all_features,
            agent_training_pipelines=agent_training_pipelines,
            per_example_loss_fn=per_example_loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            save_name=f"{save_name}_s2c_val" if save_name else None,
            no_goal_mode=no_goal_mode,
        )

    if no_distractor_to_distractor_validation:
        no_distractor_to_distractor_val_result = perform_no_distractor_to_distractor_validation(
            all_examples=training_examples,
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            all_features=all_features,
            agent_training_pipelines=agent_training_pipelines,
            per_example_loss_fn=per_example_loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            save_name=f"{save_name}_nd2d_val" if save_name else None,
            no_goal_mode=no_goal_mode,
        )

    if kfold_agent_validation:
        kfold_agent_val_results = perform_kfold_agent_validation(
            training_examples=training_examples,
            agent_list=list(agent_training_pipelines.keys()),
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            agent_kfold_splits=agent_kfold_splits,
            agent_kfold_seed=agent_kfold_seed,
            no_goal_mode=no_goal_mode,
        )

    few_shot_val_result = None
    if few_shot_kwargs is not None:
        few_shot_val_result = perform_few_shot_validation(
            training_examples=training_examples,
            agent_list=list(agent_training_pipelines.keys()),
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            n_train_agents=few_shot_kwargs.get("n_train_agents", 3),
            n_trials=few_shot_kwargs.get("n_trials", 10),
            seed=few_shot_kwargs.get("seed", 42),
            no_goal_mode=no_goal_mode,
            agent_training_pipelines=agent_training_pipelines,
            all_features=all_features,
            include_no_goal_feature=include_no_goal_feature,
        )

    few_shot_sweep_results = None
    if few_shot_sweep_kwargs is not None:
        few_shot_sweep_results = perform_few_shot_sweep_validation(
            training_examples=training_examples,
            agent_list=list(agent_training_pipelines.keys()),
            n_features=n_features,
            loss_fn=loss_fn,
            data_loss_fn=data_loss_fn,
            model_type=model_type,
            model_kwargs=effective_model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            train_agent_counts=few_shot_sweep_kwargs.get("agent_counts"),
            n_trials_per_size=few_shot_sweep_kwargs.get("n_trials_per_size", 5),
            seed=few_shot_sweep_kwargs.get("seed", 42),
            no_goal_mode=no_goal_mode,
            agent_training_pipelines=agent_training_pipelines,
            all_features=all_features,
            include_no_goal_feature=include_no_goal_feature,
        )

    # Log to wandb if available
    if wandb.run is not None:
        wandb_log_dict = {
            "model_fit/model_type": model_type,
        }

        if final_loss is not None:
            wandb_log_dict["model_fit/final_loss"] = final_loss

        if compute_baselines and not skip_main_training:
            wandb_log_dict.update({
                "baselines/per_goal_baseline_loss": per_goal_baseline_loss,
                "baselines/per_feature_baseline_loss": per_feature_baseline_loss,
                "baselines/random_choice_baseline_loss": random_choice_baseline_loss,
                "baselines/feature_decomposition_gap": per_feature_baseline_loss - per_goal_baseline_loss,
                "model_fit/loss_gap_to_per_feature_baseline": final_loss - per_feature_baseline_loss,
            })

        # Log model_kwargs
        for key, value in effective_model_kwargs.items():
            wandb_log_dict[f"model_fit/model_kwargs/{key}"] = value

        # Log hyperparameters
        if params is not None:
            for param_name, val in params.items():
                wandb_log_dict[f"model_fit/params/{param_name}"] = val

        for category_name, (category_loss, _) in stratified_losses.items():
            wandb_log_dict[f"model_fit/train_loss_{category_name}"] = category_loss

        # Add per-agent loss statistics
        if agent_train_losses:
            agent_losses_array = jnp.array(list(agent_train_losses.values()))
            wandb_log_dict["model_fit/train_loss_per_agent_mean"] = float(jnp.mean(agent_losses_array))
            wandb_log_dict["model_fit/train_loss_per_agent_std"] = float(jnp.std(agent_losses_array))
            wandb_log_dict["model_fit/train_loss_per_agent_min"] = float(jnp.min(agent_losses_array))
            wandb_log_dict["model_fit/train_loss_per_agent_max"] = float(jnp.max(agent_losses_array))
        
        if env_holdout_val_result is not None:
            wandb_log_dict["model_fit/env_holdout_val_loss"] = env_holdout_val_result['val_loss']

        if single_stage_to_multi_stage_val_result is not None:
            wandb_log_dict["model_fit/single_stage_to_multi_stage_val_loss"] = single_stage_to_multi_stage_val_result['val_loss']

        if no_distractor_to_distractor_val_result is not None:
            wandb_log_dict["model_fit/no_distractor_to_distractor_val_loss"] = no_distractor_to_distractor_val_result['val_loss']
        
        # Add validation results
        if kfold_agent_val_results:
            val_losses = jnp.array([r['val_loss'] for r in kfold_agent_val_results])
            wandb_log_dict["model_fit/kfold_agent_val_loss_mean"] = float(jnp.mean(val_losses))
            wandb_log_dict["model_fit/kfold_agent_val_loss_std"] = float(jnp.std(val_losses))
            for result in kfold_agent_val_results:
                wandb_log_dict[f"model_fit/kfold_agent_val_loss/fold_{result['fold']}"] = result['val_loss']

        if few_shot_val_result is not None:
            wandb_log_dict["model_fit/few_shot_val_mean_full_loss"] = few_shot_val_result['mean_full_loss']
            wandb_log_dict["model_fit/few_shot_val_std_error_full_loss"] = few_shot_val_result['std_error_full_loss']
            wandb_log_dict["model_fit/few_shot_val_mean_holdout_loss"] = few_shot_val_result['mean_holdout_loss']
            wandb_log_dict["model_fit/few_shot_val_std_error_holdout_loss"] = few_shot_val_result['std_error_holdout_loss']
            wandb_log_dict["model_fit/few_shot_val_n_train_agents"] = few_shot_val_result['n_train_agents']
            wandb_log_dict["model_fit/few_shot_val_n_trials"] = few_shot_val_result['n_trials']
            if few_shot_kwargs:
                for key, value in few_shot_kwargs.items():
                    wandb_log_dict[f"model_fit/few_shot_kwargs/{key}"] = value

        if few_shot_sweep_results is not None:
            for n_agents, res in few_shot_sweep_results.items():
                wandb_log_dict[f"model_fit/few_shot_sweep/{n_agents}_agents_holdout_loss"] = res['mean_holdout_loss']
                wandb_log_dict[f"model_fit/few_shot_sweep/{n_agents}_agents_holdout_loss_std_error"] = res['std_error_holdout_loss']
                wandb_log_dict[f"model_fit/few_shot_sweep/{n_agents}_agents_train_loss"] = res['mean_train_loss']
            if few_shot_sweep_kwargs:
                for key, value in few_shot_sweep_kwargs.items():
                    if isinstance(value, list):
                        wandb_log_dict[f"model_fit/few_shot_sweep_kwargs/{key}"] = str(value)
                    else:
                        wandb_log_dict[f"model_fit/few_shot_sweep_kwargs/{key}"] = value

        wandb.log(wandb_log_dict)

    # Save summary data to file
    summary_key = save_name or model_type
    summary_data = {
        "model_type": model_type,
        "model_kwargs": effective_model_kwargs,
    }

    if params is not None:
        summary_data['params'] = {k: float(v) if v.ndim == 0 else v.tolist() for k, v in params.items()}
    if final_loss is not None:
        summary_data["final_loss"] = final_loss
    if stratified_losses:
        summary_data["stratified_losses"] = {
            category: {"loss": loss, "n_examples": n_examples}
            for category, (loss, n_examples) in stratified_losses.items()
        }
    
    if compute_baselines:
        summary_data["baselines"] = {
            "per_goal_baseline_loss": per_goal_baseline_loss,
            "per_feature_baseline_loss": per_feature_baseline_loss,
            "random_choice_baseline_loss": random_choice_baseline_loss,
            "feature_decomposition_gap": per_feature_baseline_loss - per_goal_baseline_loss,
        }
    
    if env_holdout_val_result is not None:
        summary_data["env_holdout_validation"] = {
            "val_loss": env_holdout_val_result["val_loss"],
            "n_train_examples": env_holdout_val_result["n_train_examples"],
            "n_val_examples": env_holdout_val_result["n_val_examples"],
        }
    
    if single_stage_to_multi_stage_val_result is not None:
        summary_data["single_stage_to_multi_stage_validation"] = {
            "train_loss": single_stage_to_multi_stage_val_result.get("train_loss"),
            "val_loss": single_stage_to_multi_stage_val_result["val_loss"],
            "n_train_examples": single_stage_to_multi_stage_val_result["n_train_examples"],
            "n_val_examples": single_stage_to_multi_stage_val_result["n_val_examples"],
            "stratified_losses": {
                category: {"loss": loss, "n_examples": n_examples}
                for category, (loss, n_examples) in single_stage_to_multi_stage_val_result.get("stratified_val_losses", {}).items()
            },
        }
    
    if no_distractor_to_distractor_val_result is not None:
        summary_data["no_distractor_to_distractor_validation"] = {
            "train_loss": no_distractor_to_distractor_val_result.get("train_loss"),
            "val_loss": no_distractor_to_distractor_val_result["val_loss"],
            "n_train_examples": no_distractor_to_distractor_val_result["n_train_examples"],
            "n_val_examples": no_distractor_to_distractor_val_result["n_val_examples"],
            "stratified_losses": {
                category: {"loss": loss, "n_examples": n_examples}
                for category, (loss, n_examples) in no_distractor_to_distractor_val_result.get("stratified_val_losses", {}).items()
            },
        }
    
    if kfold_agent_val_results:
        val_losses = [r["val_loss"] for r in kfold_agent_val_results]
        summary_data["kfold_agent_validation"] = {
            "mean_val_loss": float(jnp.mean(jnp.array(val_losses))),
            "std_val_loss": float(jnp.std(jnp.array(val_losses))),
            "std_error_val_loss": float(jnp.std(jnp.array(val_losses)) / jnp.sqrt(len(val_losses))),
            "per_fold_losses": {f"fold_{r['fold']}": r["val_loss"] for r in kfold_agent_val_results},
        }

    # Keep all scalar aggregate metrics (mean/std/std_error losses + the
    # post-hoc 3-way / 2-way KL/TV/Brier/dir-acc), dropping the large
    # per_trial_results list which is not JSON-serialised.
    def _scalar_only(d: dict) -> dict:
        # Keep scalar ints/floats only; drop non-finite floats (e.g. a NaN
        # directional-accuracy from zero directional pairs) so the saved
        # summaries stay valid, strictly-parseable JSON.
        return {
            k: v for k, v in d.items()
            if isinstance(v, int)
            or (isinstance(v, float) and math.isfinite(v))
        }

    if few_shot_val_result is not None:
        summary_data["few_shot_validation"] = _scalar_only(few_shot_val_result)

    if few_shot_sweep_results is not None:
        summary_data["few_shot_sweep_validation"] = {
            str(n_agents): _scalar_only(res)
            for n_agents, res in few_shot_sweep_results.items()
        }

    # Save to summaries file using transform for thread-safety
    summaries_path = "local/developmental_model_fit_summaries.json"
    
    def update_summaries(existing: dict | None) -> dict:
        if existing is None:
            existing = {}
        existing[summary_key] = summary_data
        return existing
    
    transform("json", update_summaries, summaries_path, default_data={})
    log.info(f"Saved model fit summary to {summaries_path} under key '{summary_key}'")

    return summary_data


def plot_latent_dimension_pareto(
    sweep_results: dict[int | str, dict],
    n_features: int,
    output_dir: str,
    baselines: dict[str, float] | None = None,
    plot_name: str = "latent_dimension_sweep",
) -> None:
    """
    Plot a Pareto curve of loss vs latent dimension.

    Args:
        sweep_results: Dictionary mapping latent dimension (or "full") to result dict with 'final_loss'
        n_features: Number of features (full rank)
        output_dir: Directory to save the plot
        baselines: Optional dictionary of baseline losses (e.g., {"per_feature": 0.044, "random_choice": 0.189})
        plot_name: Base name for the output PNG file (without extension)
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Extract data points
    dims = []
    losses = []
    for key, result in sweep_results.items():
        if key == "full":
            dim = n_features
        else:
            dim = int(key)
        dims.append(dim)
        losses.append(result["final_loss"])

    # Sort by dimension
    sorted_pairs = sorted(zip(dims, losses), key=lambda x: x[0])
    dims, losses = zip(*sorted_pairs)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(dims, losses, 'o-', linewidth=2, markersize=8, label="Model loss")

    # Mark the full-rank point
    full_rank_idx = dims.index(n_features)
    ax.scatter([n_features], [losses[full_rank_idx]], s=150, c='red',
               marker='*', zorder=5, label=f"Full rank (d={n_features})")

    # Add baseline reference lines
    if baselines:
        colors = {'per_feature': 'green', 'random_choice': 'orange', 'per_goal': 'purple'}
        for baseline_name, baseline_loss in baselines.items():
            color = colors.get(baseline_name, 'gray')
            ax.axhline(y=baseline_loss, linestyle='--', color=color, alpha=0.7,
                      label=f"{baseline_name.replace('_', ' ').title()}: {baseline_loss:.4f}")

    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("Loss (KL Divergence)", fontsize=12)
    ax.set_title("Latent Dimension vs Model Loss", fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if range is large
    if max(dims) / min(dims) > 10:
        ax.set_xscale('log', base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{plot_name}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved Pareto plot to {output_path}")


def sweep_latent_dimension(
    get_runs_from_wandb_kwargs: Kwargs,
    latent_dimensions: list[int | None] = [1, 2, 3, 4, 8, 16, 32, 64, None],
    model_kwargs_base: dict | None = None,
    num_epochs: int = 3,
    learning_rate: float = 0.003,
    seed: int = 42,
    cache_dir: str = "local/cache",
    max_workers: int = 10,
    output_dir: str = "local/plots/sweeps",
    sweep_name: str = "latent_dimension_sweep",
    compute_baselines: bool = True,
    no_goal_mode: str = "unweighted_ignore_no_goal",
    include_no_goal_feature: bool = False,
    offline: bool = False,
) -> dict[str, Any]:
    """
    Sweep over latent dimensions for MultiChoiceKLModel and generate a Pareto plot.

    Args:
        get_runs_from_wandb_kwargs: Kwargs for fetching runs from wandb
        latent_dimensions: List of latent dimensions to test. None means full rank (n_features).
        model_kwargs_base: Base model kwargs (merged with latent dimension settings)
        num_epochs: Number of training epochs per model
        learning_rate: Learning rate for Adam optimizer
        seed: Seed for random S matrix initialization
        cache_dir: Directory for caching run metrics
        max_workers: Number of parallel workers for data fetching
        output_dir: Directory to save plots and results
        sweep_name: Name used for output filenames (e.g., "latent_dimension_sweep")
        compute_baselines: Whether to compute per_feature and random_choice baselines
        include_no_goal_feature: If True (default), includes dedicated no_goal feature.

    Returns:
        Dictionary with sweep config, baselines, and results per latent dimension
    """
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    model_kwargs_base = model_kwargs_base or {}

    # Load data once
    log.info("Loading data from wandb cache...")
    run_env_metrics, possible_goals = load_runs_from_cache(
        get_runs_from_wandb_kwargs=get_runs_from_wandb_kwargs,
        cache_dir=cache_dir,
        max_workers=max_workers,
        offline=offline,
    )

    # Filter runs
    filtered_run_metrics = {
        run_name: env_metrics
        for run_name, env_metrics in run_env_metrics.items()
        if "distill" in run_name and "and" not in run_name
    }

    if len(filtered_run_metrics) == 0:
        raise ValueError("No valid runs found to fit model")

    log.info(f"Loaded {len(filtered_run_metrics)} runs")

    # Prepare training data once
    log.info("Preparing training data...")
    agent_training_pipelines, feature_to_idx, all_features, n_features = prepare_training_data(
        run_env_metrics=filtered_run_metrics,
        possible_goals=possible_goals,
        include_no_goal_feature=include_no_goal_feature,
    )

    if len(agent_training_pipelines) == 0:
        raise ValueError("No valid training pipelines found")

    log.info("Preparing training examples...")
    training_examples = prepare_examples(
        run_metrics_dict=filtered_run_metrics,
        agent_training_pipelines=agent_training_pipelines,
        feature_to_idx=feature_to_idx,
        n_features=n_features,
        include_no_goal_feature=include_no_goal_feature,
    )

    if len(training_examples) == 0:
        raise ValueError("No valid training data found")

    log.info("Batching training examples...")
    batch_data = examples_to_batch(training_examples, no_goal_mode=no_goal_mode)
    log.info(f"Prepared {len(training_examples)} training examples with {n_features} features")

    # Compute baselines if requested
    baselines = {}
    if compute_baselines:
        log.info("Computing baselines...")

        _, per_feature_baseline_loss = fit_per_agent_per_feature_baseline(
            training_examples=training_examples,
            n_features=n_features,
            no_goal_mode=no_goal_mode,
            include_no_goal_feature=include_no_goal_feature,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )
        baselines["per_feature"] = float(per_feature_baseline_loss)
        log.info(f"  Per-feature baseline: {per_feature_baseline_loss:.6f}")

        random_choice_baseline_loss = get_random_choice_baseline(
            training_examples=training_examples,
            no_goal_mode=no_goal_mode,
        )
        baselines["random_choice"] = float(random_choice_baseline_loss)
        log.info(f"  Random choice baseline: {random_choice_baseline_loss:.6f}")

    # Sweep over latent dimensions
    sweep_results: dict[int | str, dict] = {}
    model_type = "multi_kl"

    log.info("="*50)
    log.info("Starting latent dimension sweep")
    log.info("="*50)

    for latent_dim in tqdm(latent_dimensions, desc="Sweeping latent dimensions"):
        # Build model kwargs with include_no_goal_feature
        model_kwargs = {**model_kwargs_base, 'include_no_goal_feature': include_no_goal_feature}

        if latent_dim is None:
            # Full rank - don't set latent_dimension
            key = "full"
            dim_str = f"full (d={n_features})"
        else:
            # Merge with existing value_function settings (e.g., bilinear type/rank)
            base_vf = model_kwargs_base.get("value_function", {}) if model_kwargs_base else {}
            vf_type = base_vf.get("type", "linear")

            # Build value_function config, only including params relevant to the type
            vf_config = {
                "type": vf_type,
                "latent_dimension": latent_dim,
                "seed": base_vf.get("seed", seed),
                **{k: v for k, v in base_vf.items() if k not in ["type", "latent_dimension", "init", "seed"]},
            }

            # Only add 'init' for linear value functions (not bilinear, mlp, etc.)
            if vf_type == "linear":
                vf_config["init"] = base_vf.get("init", "random_gaussian")

            model_kwargs["value_function"] = vf_config
            key = str(latent_dim)
            dim_str = f"d={latent_dim}"

        log.info(f"Training model with latent dimension {dim_str}...")

        # Create loss functions
        loss_fn, data_loss_fn, _ = create_loss_functions(
            model_type=model_type,
            model_kwargs=model_kwargs,
            no_goal_mode=no_goal_mode,
        )

        # Train model
        params = train_preference_model(
            train_dataset=batch_data,
            n_features=n_features,
            loss_fn=loss_fn,
            model_type=model_type,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            show_progress=False,
        )

        # Compute final loss
        final_loss = compute_data_loss_batched(data_loss_fn, params, batch_data)

        sweep_results[key] = {
            "latent_dimension": latent_dim if latent_dim is not None else n_features,
            "final_loss": float(final_loss),
            "params": {k: float(v) if v.ndim == 0 else v.tolist() for k, v in params.items()},
        }

        log.info(f"  Latent dim {dim_str}: loss = {final_loss:.6f}")

    # Generate Pareto plot
    log.info("Generating Pareto plot...")
    plot_latent_dimension_pareto(
        sweep_results=sweep_results,
        n_features=n_features,
        output_dir=output_dir,
        baselines=baselines if compute_baselines else None,
        plot_name=sweep_name,
    )

    # Prepare results dict
    results_data = {
        "sweep_config": {
            "latent_dimensions": [d if d is not None else n_features for d in latent_dimensions],
            "n_features": n_features,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "baselines": baselines,
        "results": sweep_results,
    }

    # Save results to JSON
    results_path = os.path.join(output_dir, f"{sweep_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    log.info(f"Saved results to {results_path}")

    # Log summary
    log.info("="*50)
    log.info("Sweep Summary")
    log.info("="*50)

    summary_table = [
        [result["latent_dimension"], f"{result['final_loss']:.6f}"]
        for result in sorted(sweep_results.values(), key=lambda x: x["latent_dimension"])
    ]
    log.info(f"\n{tabulate(summary_table, headers=['Latent Dim', 'Loss'], tablefmt='simple_outline')}")

    return results_data


def validate_preference_models(
    get_runs_from_wandb_kwargs: Kwargs,
    model_keys: list[str] | None = None,
    summaries_path: str = "local/validation_sweep_results/merged_summaries.json",
    cache_dir: str = "local/cache",
    max_workers: int = 10,
    include_no_goal_feature: bool = False,
    no_goal_mode: str = "full_distribution",
    elo_holdout_fraction: float = 0.2,
    elo_holdout_repeats: int = 10,
    directional_threshold: float = 0.10,
    scatter_plot_model_key: str | None = None,
    scatter_plot_output_dir: str = "local/plots/elo_vs_model",
    scatter_show_title: bool = True,
    output_dir: str = "local/plots/validation",
    offline: bool = False,
    skip_experiments: list[int] | None = None,
) -> dict[str, Any]:
    """
    Post-hoc validation of Elo scores and fitted preference models.

    Runs three experiments:
    1. Elo holdout validation: fit Elo on train split, evaluate on held-out pairs
    2. Post-hoc metrics: compute TV, Brier, directional accuracy for fitted models
    3. Scatter plot: Elo scores vs model values per (agent, goal)

    Args:
        get_runs_from_wandb_kwargs: Kwargs for fetching runs from wandb
        model_keys: List of keys in summaries file to evaluate. If None, evaluates
            all models with full_distribution losses.
        summaries_path: Path to saved model summaries JSON
        cache_dir: Directory for caching run metrics
        max_workers: Number of parallel workers
        include_no_goal_feature: Whether models use dedicated no-goal feature
        no_goal_mode: Loss mode for metric computation
        elo_holdout_fraction: Fraction of pairs to hold out for Elo validation
        elo_holdout_repeats: Number of random splits to average over
        directional_threshold: Min |obs_0 - obs_1| for directional accuracy
        scatter_plot_model_key: Which model to use for scatter plot (default: first in model_keys)
        output_dir: Directory for output plots and results
        offline: If True, only use cached data

    Returns:
        Dict with 'elo_validation', 'model_metrics', and 'scatter_results'.
    """
    import json
    import os
    from src.preference_modelling.metrics import (
        load_saved_params,
        compute_post_hoc_metrics,
        compute_per_agent_per_goal_values,
        compute_per_agent_weights,
    )

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Load data (shared by all experiments)
    # =========================================================================
    run_env_metrics, possible_goals = load_runs_from_cache(
        get_runs_from_wandb_kwargs=get_runs_from_wandb_kwargs,
        cache_dir=cache_dir,
        max_workers=max_workers,
        offline=offline,
    )

    filtered_run_metrics = {
        run_name: env_metrics
        for run_name, env_metrics in run_env_metrics.items()
        if "distill" in run_name and "and" not in run_name
    }
    before_zero_filter = len(filtered_run_metrics)
    filtered_run_metrics = {
        run_name: env_metrics
        for run_name, env_metrics in filtered_run_metrics.items()
        if any(r is not None and r != (0.0, 0.0) for r in env_metrics.values())
    }
    log.info(f"Dropped {before_zero_filter - len(filtered_run_metrics)} agents with all-zero eval rates")

    log.info(f"Loaded {len(filtered_run_metrics)} agent runs")

    skip = set(skip_experiments or [])

    # =========================================================================
    # Experiment 1: Elo holdout validation
    # =========================================================================
    if 1 in skip:
        log.info("Skipping Experiment 1 (Elo Holdout Validation)")
        elo_validation = None
    else:
        log.info("=" * 50)
        log.info("Experiment 1: Elo Holdout Validation")
        log.info("=" * 50)

        elo_validation = validate_elo_scores(
            run_env_metrics=filtered_run_metrics,
            possible_goals=possible_goals,
            holdout_fraction=elo_holdout_fraction,
            n_repeats=elo_holdout_repeats,
            directional_threshold=directional_threshold,
        )

    # =========================================================================
    # Prepare training data for model evaluation
    # =========================================================================
    log.info("Preparing training data for model evaluation...")
    agent_training_pipelines, feature_to_idx, all_features, n_features = prepare_training_data(
        run_env_metrics=filtered_run_metrics,
        possible_goals=possible_goals,
        include_no_goal_feature=include_no_goal_feature,
    )

    training_examples = prepare_examples(
        run_metrics_dict=filtered_run_metrics,
        agent_training_pipelines=agent_training_pipelines,
        feature_to_idx=feature_to_idx,
        n_features=n_features,
        include_no_goal_feature=include_no_goal_feature,
    )
    log.info(f"Prepared {len(training_examples)} training examples")

    # =========================================================================
    # Experiment 2: Post-hoc metrics for fitted models
    # =========================================================================
    model_metrics: dict[str, dict] = {}
    model_agent_weights: dict[str, dict] = {}

    if 2 in skip:
        log.info("Skipping Experiment 2 (Post-hoc Model Metrics)")
    else:
        log.info("=" * 50)
        log.info("Experiment 2: Post-hoc Model Metrics")
        log.info("=" * 50)

    # Determine which models to evaluate
    if model_keys is None:
        # Default: evaluate all models in the summaries file
        with open(summaries_path) as f:
            all_summaries = json.load(f)
        model_keys = list(all_summaries.keys())
        # Filter to full_distribution models only (loss > 0.1 heuristic)
        model_keys = [
            k for k in model_keys
            if all_summaries[k].get('final_loss', 0) > 0.1
        ]
        log.info(f"Auto-selected {len(model_keys)} full_distribution models: {model_keys}")

    if 2 not in skip:
        for model_key in model_keys:
            log.info(f"\n--- Evaluating model: {model_key} ---")
            try:
                params, model_type, model_kwargs = load_saved_params(summaries_path, model_key)
                # Ensure include_no_goal_feature is set
                model_kwargs['include_no_goal_feature'] = include_no_goal_feature

                result = compute_post_hoc_metrics(
                    params=params,
                    model_type=model_type,
                    model_kwargs=model_kwargs,
                    training_examples=training_examples,
                    agent_training_pipelines=agent_training_pipelines,
                    all_features=all_features,
                    feature_to_idx=feature_to_idx,
                    n_features=n_features,
                    include_no_goal_feature=include_no_goal_feature,
                    directional_threshold=directional_threshold,
                )
                model_metrics[model_key] = result['metrics']
                model_agent_weights[model_key] = result['agent_weights']
            except Exception as e:
                log.error(f"Failed to evaluate model {model_key}: {e}")
                import traceback
                traceback.print_exc()

        # Summary table
        if model_metrics:
            for variant, variant_label in [('3way', 'Three-way'), ('2way', 'Two-way normalised')]:
                metric_table = [
                    [
                        key,
                        f"{m[f'{variant}_kl_mean']:.4f}",
                        f"{m[f'{variant}_tv_mean']:.4f}",
                        f"{m[f'{variant}_brier_mean']:.4f}",
                        f"{m[f'{variant}_directional_accuracy']:.2%}" if not np.isnan(m.get(f'{variant}_directional_accuracy', float('nan'))) else "N/A",
                    ]
                    for key, m in model_metrics.items()
                ]
                log.info(f"\n{variant_label} Model Metrics:\n{tabulate(metric_table, headers=['Model', 'KL', 'TV', 'Brier', 'Dir. Accuracy'], tablefmt='simple_outline')}")

    # =========================================================================
    # Experiment 3: Elo vs Model Values scatter plot
    # =========================================================================
    scatter_results: dict = {}
    if 3 in skip:
        log.info("Skipping Experiment 3 (Elo vs Model Values Scatter Plot)")
    else:
        log.info("=" * 50)
        log.info("Experiment 3: Elo vs Model Values Scatter Plot")
        log.info("=" * 50)

        scatter_key = scatter_plot_model_key or (model_keys[0] if model_keys else None)

        if scatter_key is None:
            log.warning("Cannot generate scatter plot: no model key available")
        else:
            # Load scatter model params once (needed for weights and values)
            params, model_type, model_kwargs = load_saved_params(summaries_path, scatter_key)
            model_kwargs['include_no_goal_feature'] = include_no_goal_feature

            # Get agent weights for scatter model: reuse Experiment 2 output if available,
            # otherwise compute directly (avoids the expensive per-example prediction step)
            if scatter_key in model_agent_weights:
                scatter_agent_weights = model_agent_weights[scatter_key]
            else:
                log.info(f"Computing per-agent weights for {scatter_key}...")
                scatter_agent_weights = compute_per_agent_weights(
                    params=params,
                    model_type=model_type,
                    model_kwargs=model_kwargs,
                    all_features=all_features,
                    agent_training_pipelines=agent_training_pipelines,
                )

            # Compute per-agent Elo scores
            log.info("Computing per-agent Elo scores...")
            agent_elo_scores: dict[str, dict[str, float]] = {}
            for run_name, env_metrics in filtered_run_metrics.items():
                if run_name in scatter_agent_weights:
                    elo = compute_elo_scores(
                        env_metrics=env_metrics,
                        possible_goals=possible_goals,
                    )
                    # Remove no_goal entry for scatter plot
                    elo.pop("no_goal", None)
                    agent_elo_scores[run_name] = elo

            # Compute per-agent per-goal model values
            log.info("Computing per-agent per-goal model values...")
            agent_goal_values = compute_per_agent_per_goal_values(
                params=params,
                model_type=model_type,
                model_kwargs=model_kwargs,
                agent_weights=scatter_agent_weights,
                possible_goals=possible_goals,
                feature_to_idx=feature_to_idx,
                n_features=n_features,
            )

            scatter_results = plot_elo_vs_model_values(
                agent_elo_scores=agent_elo_scores,
                agent_goal_values=agent_goal_values,
                possible_goals=possible_goals,
                output_dir=scatter_plot_output_dir,
                model_name=scatter_key,
                show_title=scatter_show_title,
            )

    # =========================================================================
    # Save all results
    # =========================================================================
    results = {
        'elo_validation': elo_validation,
        'model_metrics': {k: v for k, v in model_metrics.items()},
        'scatter_results': scatter_results,
    }

    results_path = os.path.join(output_dir, "validation_results.json")

    # Make JSON-serialisable
    def make_serialisable(obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serialisable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(make_serialisable(results), f, indent=2)

    log.info(f"Saved validation results to {results_path}")

    return results

