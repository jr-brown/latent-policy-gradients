"""
Two-stage pipeline analysis for RL agent preference formation.

This module provides functions to analyze how values form and persist in
two-stage training pipelines, where agents are trained on one goal and then
another goal sequentially.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# Import helpers from preference_analysis for consistency
from src.preference_analysis import (
    get_palette_color,
    compute_figsize_for_bars,
    compute_bar_width,
    style_feature_tick_labels,
)

log = logging.getLogger(__name__)

# Type aliases for clarity
GoalTuple = Tuple[str, str]  # (colour, shape)
PreferenceScores = Tuple[Dict[str, float], Dict[str, float]]  # (colour_scores, shape_scores)
RunPreferences = Dict[str, PreferenceScores]


# ============================================================================
# Helper Functions
# ============================================================================


def parse_two_stage_pipeline(run_name: str) -> Optional[Tuple[GoalTuple, GoalTuple]]:
    """
    Parse a two-stage pipeline run name into stage 1 and stage 2 goals.

    Expected format: "<colour1>_<shape1>_then_<colour2>_<shape2>"
    Note: Run names are already cleaned (no "maze_eval_" or "_distill" prefix/suffix)

    Args:
        run_name: The cleaned run name

    Returns:
        ((stage1_colour, stage1_shape), (stage2_colour, stage2_shape)) or None if invalid
    """
    pattern = r"^([a-z]+)_([a-z]+)_then_([a-z]+)_([a-z]+)$"
    match = re.match(pattern, run_name)

    if match:
        c1, s1, c2, s2 = match.groups()
        return ((c1, s1), (c2, s2))

    return None


def extract_features_from_goal(goal: GoalTuple) -> Set[str]:
    """
    Extract feature set from a goal tuple.

    Args:
        goal: (colour, shape) tuple

    Returns:
        Set of feature strings
    """
    return {goal[0], goal[1]}


def count_shared_features(goal1: GoalTuple, goal2: GoalTuple) -> int:
    """
    Count how many features are shared between two goals.

    Args:
        goal1: First goal (colour, shape)
        goal2: Second goal (colour, shape)

    Returns:
        Number of shared features (0, 1, or 2)
    """
    features1 = extract_features_from_goal(goal1)
    features2 = extract_features_from_goal(goal2)
    return len(features1 & features2)


def find_control_pipelines_for_feature(
    experimental_stage1: GoalTuple,
    stage2_goal: GoalTuple,
    feature_to_test: str,
    all_run_names: Set[str],
) -> list:
    """
    Find control pipelines for testing persistence of a specific feature.

    Strategy: Keep the OTHER stage1 feature constant, vary the feature being tested.

    Example:
      Experimental: (red, cross) -> (blue, diamond), testing "red"
      Controls: (black, cross) -> (blue, diamond), (grey, cross) -> (blue, diamond), ...
                (keep "cross", vary first feature)

    Args:
        experimental_stage1: The experimental pipeline's stage 1 goal (f_i, f_j)
        stage2_goal: The stage 2 goal (f_k, f_l) - same for all
        feature_to_test: The feature we're measuring persistence for (f_i)
        all_run_names: Set of all available run names

    Returns:
        List of control pipeline names
    """
    # Identify which feature is being kept constant
    f_i, f_j = experimental_stage1
    if feature_to_test == f_i:
        constant_feature = f_j
        variable_position = 0  # First element of stage1
    elif feature_to_test == f_j:
        constant_feature = f_i
        variable_position = 1  # Second element of stage1
    else:
        raise ValueError(f"Feature {feature_to_test} not in experimental stage1 {experimental_stage1}")

    controls = []

    # Find all pipelines with same stage2, one constant stage1 feature, and different variable feature
    for run_name in all_run_names:
        parsed = parse_two_stage_pipeline(run_name)
        if parsed is None:
            continue

        control_stage1, control_stage2 = parsed

        # Must have same stage 2
        if control_stage2 != stage2_goal:
            continue

        # Must be different from experimental
        if control_stage1 == experimental_stage1:
            continue

        # Check if constant feature is in the right position
        control_features = [control_stage1[0], control_stage1[1]]
        if variable_position == 0:
            # Keep second feature constant, vary first
            if control_features[1] != constant_feature:
                continue
            if control_features[0] == feature_to_test:
                continue  # Must not have the test feature
        else:
            # Keep first feature constant, vary second
            if control_features[0] != constant_feature:
                continue
            if control_features[1] == feature_to_test:
                continue  # Must not have the test feature

        controls.append(run_name)

    return controls


def find_control_pipelines_for_inhibition(
    experimental_stage1: GoalTuple,
    experimental_stage2: GoalTuple,
    shared_feature: str,
    all_run_names: Set[str]
) -> list:
    """
    Find control pipelines for value inhibition analysis.

    Strategy: Keep stage2 the same, keep non-shared stage1 feature constant,
    vary the shared feature to features NOT present in stage2.

    Example:
      Experimental: (red, cross) -> (blue, cross), where "cross" is shared
      Controls: (red, ring) -> (blue, cross), (red, plus) -> (blue, cross), ...
                (keep "red" and stage2 "(blue, cross)", vary "cross" in stage1 to other features)

    This isolates the effect of having the shared feature in stage1.

    Args:
        experimental_stage1: Stage 1 goal containing the shared feature (f_i, f_j)
        experimental_stage2: Stage 2 goal containing the shared feature (f_j, f_k)
        shared_feature: The feature that appears in both stages (f_j)
        all_run_names: Set of all available run names

    Returns:
        List of control pipeline names
    """
    # Identify the non-shared stage1 feature (the one to keep constant)
    f_i, f_j = experimental_stage1
    if shared_feature == f_i:
        constant_stage1_feature = f_j
        shared_position_stage1 = 0
    elif shared_feature == f_j:
        constant_stage1_feature = f_i
        shared_position_stage1 = 1
    else:
        raise ValueError(f"Shared feature {shared_feature} not in stage1 {experimental_stage1}")

    controls = []

    for run_name in all_run_names:
        parsed = parse_two_stage_pipeline(run_name)
        if parsed is None:
            continue

        control_stage1, control_stage2 = parsed

        # Must have same stage2
        if control_stage2 != experimental_stage2:
            continue

        # Must be different from experimental
        if control_stage1 == experimental_stage1:
            continue

        # Extract control stage1 features
        control_f_i, control_f_j = control_stage1

        # Must keep the non-shared stage1 feature constant
        if shared_position_stage1 == 0:
            # Experimental: (shared, constant) -> stage2
            # Control: (varying, constant) -> stage2
            if control_f_j != constant_stage1_feature:
                continue
            if control_f_i == shared_feature:
                continue  # Must not have the experimental shared feature
        else:
            # Experimental: (constant, shared) -> stage2
            # Control: (constant, varying) -> stage2
            if control_f_i != constant_stage1_feature:
                continue
            if control_f_j == shared_feature:
                continue  # Must not have the experimental shared feature

        controls.append(run_name)

    return controls


def find_reverse_pipeline(
    stage1: GoalTuple,
    stage2: GoalTuple,
    all_run_names: Set[str]
) -> Optional[str]:
    """
    Find the reverse pipeline that swaps stage order.

    Args:
        stage1: Original stage 1 goal
        stage2: Original stage 2 goal
        all_run_names: Set of all available run names

    Returns:
        Reverse run name or None if not found
    """
    reverse_name = f"{stage2[0]}_{stage2[1]}_then_{stage1[0]}_{stage1[1]}"
    return reverse_name if reverse_name in all_run_names else None


def get_feature_score(
    prefs: PreferenceScores,
    feature: str
) -> Optional[float]:
    """
    Get the preference score for a specific feature.

    Args:
        prefs: (colour_scores, shape_scores) tuple
        feature: Feature name to look up

    Returns:
        Score for the feature or None if not found
    """
    colour_scores, shape_scores = prefs

    # Try colour scores first, then shape scores
    if feature in colour_scores:
        return colour_scores[feature]
    elif feature in shape_scores:
        return shape_scores[feature]

    return None


def is_same_goal(goal1: GoalTuple, goal2: GoalTuple) -> bool:
    """
    Check if two goals have the same features (order-independent).

    Args:
        goal1: First goal
        goal2: Second goal

    Returns:
        True if goals have same features
    """
    return extract_features_from_goal(goal1) == extract_features_from_goal(goal2)


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_value_persistence(
    run_preferences: RunPreferences,
    output_dir: str = "local/plots/two_stage_analysis",
    filename: str = "value_persistence.png",
    show_title: bool = True,
) -> None:
    """
    Plot how values from stage 1 persist after stage 2 training.

    Strategy: For each feature f_i in stage1, keep the OTHER stage1 feature constant,
    and average over pipelines where f_i varies.

    Experimental pipeline: A[(f_i, f_j), (f_k, f_l)] - testing feature f_i
    Control: Average over B[(f_n, f_j), (f_k, f_l)] where f_n ≠ f_i (keep f_j constant)

    Also tests feature f_j using controls B[(f_i, f_n), (f_k, f_l)] where f_n ≠ f_j (keep f_i constant)

    Measures: Does having f_i in stage 1 increase its value compared to pipelines
    without f_i in stage 1, controlling for the presence of f_j?

    Shows 4 bars:
    1. Has feature in stage 1 (experimental) - no shared features
    2. Lacks feature in stage 1 (control avg) - no shared features
    3. Has feature in stage 1 (experimental) - one shared feature
    4. Lacks feature in stage 1 (control avg) - one shared feature

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores)
        output_dir: Directory to save plots
        filename: Output filename
        show_title: If True, display the title. Set to False for figures with captions.
    """
    log.info("Generating value persistence plot")

    # Data collection
    data = {
        'no_shared_pipeline': [],
        'no_shared_control': [],
        'one_shared_pipeline': [],
        'one_shared_control': [],
    }

    all_run_names = set(run_preferences.keys())
    skipped_no_control = 0
    skipped_invalid = 0
    processed = 0

    for run_name, prefs in run_preferences.items():
        # Parse pipeline
        parsed = parse_two_stage_pipeline(run_name)
        if parsed is None:
            skipped_invalid += 1
            log.warning(f"Invalid run name format for value persistence: {run_name}")
            continue

        stage1_goal, stage2_goal = parsed
        n_shared = count_shared_features(stage1_goal, stage2_goal)

        # Skip if both features shared (same goal twice)
        if n_shared == 2:
            continue

        # Get all stage1 features (both will be tested separately)
        stage1_features = extract_features_from_goal(stage1_goal)
        stage2_features = extract_features_from_goal(stage2_goal)
        non_shared_features = stage1_features - stage2_features

        # For EACH non-shared stage1 feature, find controls and compare
        for feature_to_test in non_shared_features:
            # Find control pipelines for this specific feature
            control_names = find_control_pipelines_for_feature(
                stage1_goal, stage2_goal, feature_to_test, all_run_names
            )

            if not control_names:
                skipped_no_control += 1
                log.debug(f"No controls found for {run_name}, testing feature '{feature_to_test}'")
                continue

            # Get experimental score for this feature
            pipeline_score = get_feature_score(prefs, feature_to_test)
            if pipeline_score is None:
                continue

            # Collect control scores for this feature
            control_scores = []
            for control_name in control_names:
                control_prefs = run_preferences[control_name]
                control_score = get_feature_score(control_prefs, feature_to_test)
                if control_score is not None:
                    control_scores.append(control_score)

            # Average control scores
            if control_scores:
                avg_control_score = np.mean(control_scores)
                log.debug(f"Feature '{feature_to_test}' in {run_name}: {len(control_scores)} controls found")

                if n_shared == 0:
                    data['no_shared_pipeline'].append(pipeline_score)
                    data['no_shared_control'].append(avg_control_score)
                else:  # n_shared == 1
                    data['one_shared_pipeline'].append(pipeline_score)
                    data['one_shared_control'].append(avg_control_score)

                processed += 1

    # Log summary
    log.info(f"Value persistence: processed {processed} data points, "
             f"skipped {skipped_no_control} (no control), "
             f"skipped {skipped_invalid} (invalid format)")

    # Check if we have any data
    if not any(data.values()):
        log.warning("No valid data for value persistence plot, skipping")
        return

    # Calculate averages (handle empty lists)
    def safe_mean(lst):
        return np.mean(lst) if lst else 0.0

    def safe_se(lst):
        return np.std(lst, ddof=1) / np.sqrt(len(lst)) if len(lst) > 1 else 0.0

    avg_no_shared_pipeline = safe_mean(data['no_shared_pipeline'])
    avg_no_shared_control = safe_mean(data['no_shared_control'])
    avg_one_shared_pipeline = safe_mean(data['one_shared_pipeline'])
    avg_one_shared_control = safe_mean(data['one_shared_control'])

    se_no_shared_pipeline = safe_se(data['no_shared_pipeline'])
    se_no_shared_control = safe_se(data['no_shared_control'])
    se_one_shared_pipeline = safe_se(data['one_shared_pipeline'])
    se_one_shared_control = safe_se(data['one_shared_control'])

    # Create plot with consistent bar width
    n_groups = 2  # no shared, one shared
    n_bars_per_group = 2  # experimental, control
    figsize = compute_figsize_for_bars(n_groups, n_bars_per_group, short=True)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_groups)
    width = compute_bar_width(n_bars_per_group, n_groups=n_groups)

    color_experimental = get_palette_color(0)
    color_control = get_palette_color(1)

    # Add grid and horizontal line BEFORE bars (zorder control)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # Bars
    ax.bar(x[0] - width/2, avg_no_shared_pipeline, width,
           label='Feature in Stage 1 only', color=color_experimental,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    ax.bar(x[0] + width/2, avg_no_shared_control, width,
           label='Feature not present', color=color_control,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)

    ax.bar(x[1] - width/2, avg_one_shared_pipeline, width,
           color=color_experimental,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    ax.bar(x[1] + width/2, avg_one_shared_control, width,
           color=color_control,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)

    # Add error bars for all 4 bars
    ax.errorbar([x[0] - width/2, x[0] + width/2, x[1] - width/2, x[1] + width/2],
                [avg_no_shared_pipeline, avg_no_shared_control, avg_one_shared_pipeline, avg_one_shared_control],
                yerr=[se_no_shared_pipeline, se_no_shared_control, se_one_shared_pipeline, se_one_shared_control],
                fmt='none', ecolor='black', capsize=4, capthick=1.5, zorder=3)

    # Styling (font sizes +4 points)
    ax.set_ylabel('Marginalised Elo for Feature', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['No shared features', 'One shared feature'], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(top=max(60, ax.get_ylim()[1]))  # Adjust y-limit to make sure enough room for legend
    ax.legend(fontsize=14, loc='lower left')

    if show_title:
        plt.title('Value Persistence: Effect of Stage 1 Feature Presence', fontsize=16, pad=20)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    log.info(f"Saved value persistence plot to {filepath}")


def plot_ordering_effects(
    run_preferences: RunPreferences,
    output_dir: str = "local/plots/two_stage_analysis",
    variant: str = "all",
    show_title: bool = True,
) -> None:
    """
    Plot how training order affects value formation.

    For each feature, shows average preference when trained in stage 1 vs stage 2.

    Pipeline pairs of A[(f_i, f_j), (f_k, f_l)] and B[(f_k, f_l), (f_i, f_j)]
    Plots Af_i vs Bf_i across all features

    Note: Only pipelines with a reverse counterpart are included (to ensure balanced
    comparisons), but scores are aggregated globally across all qualifying pipelines
    rather than strictly within pairs.

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores)
        output_dir: Directory to save plots
        variant: One of "all", "no_shared", "one_shared" to filter pipelines
        show_title: If True, display the title. Set to False for figures with captions.
    """
    log.info(f"Generating ordering effects plot (variant: {variant})")

    # Validate variant
    if variant not in ["all", "no_shared", "one_shared"]:
        raise ValueError(f"Invalid variant: {variant}. Must be 'all', 'no_shared', or 'one_shared'")

    # Data collection: feature -> {"stage1": [scores], "stage2": [scores]}
    feature_data = {}

    all_run_names = set(run_preferences.keys())
    skipped_no_reverse = 0
    skipped_filtered = 0
    skipped_invalid = 0
    processed_pipelines = 0

    for run_name, prefs in run_preferences.items():
        # Parse pipeline
        parsed = parse_two_stage_pipeline(run_name)
        if parsed is None:
            skipped_invalid += 1
            log.warning(f"Invalid run name format for ordering effects: {run_name}")
            continue

        stage1_goal, stage2_goal = parsed

        # Check if reverse exists (required for this analysis)
        reverse_name = find_reverse_pipeline(stage1_goal, stage2_goal, all_run_names)
        if reverse_name is None:
            skipped_no_reverse += 1
            log.debug(f"No reverse found for {run_name}")
            continue

        # Filter by variant
        n_shared = count_shared_features(stage1_goal, stage2_goal)
        if variant == "no_shared" and n_shared != 0:
            skipped_filtered += 1
            continue
        elif variant == "one_shared" and n_shared != 1:
            skipped_filtered += 1
            continue

        # Process this pipeline (we'll process both forward and reverse)
        processed_pipelines += 1

        # Collect data from this pipeline's stage 1 features
        stage1_features = extract_features_from_goal(stage1_goal)
        for feature in stage1_features:
            if feature not in feature_data:
                feature_data[feature] = {"stage1": [], "stage2": []}
            score = get_feature_score(prefs, feature)
            if score is not None:
                feature_data[feature]["stage1"].append(score)

        # Collect data from this pipeline's stage 2 features
        stage2_features = extract_features_from_goal(stage2_goal)
        for feature in stage2_features:
            if feature not in feature_data:
                feature_data[feature] = {"stage1": [], "stage2": []}
            score = get_feature_score(prefs, feature)
            if score is not None:
                feature_data[feature]["stage2"].append(score)

    # Log summary
    log.info(f"Ordering effects ({variant}): processed {processed_pipelines} pipelines ({processed_pipelines // 2} pairs), "
             f"skipped {skipped_no_reverse} (no reverse), "
             f"skipped {skipped_filtered} (filtered by variant), "
             f"skipped {skipped_invalid} (invalid format)")

    # Filter to features that have data
    features_with_data = {
        feat: data for feat, data in feature_data.items()
        if data["stage1"] or data["stage2"]
    }

    if not features_with_data:
        log.warning(f"No valid data for ordering effects plot ({variant}), skipping")
        return

    # Separate into colours and shapes (assuming standard naming)
    all_colours = {'black', 'red', 'blue', 'grey'}
    all_shapes = {'circle', 'cross', 'plus', 'diamond', 'ring'}

    colour_features = sorted([f for f in features_with_data.keys() if f in all_colours])
    shape_features = sorted([f for f in features_with_data.keys() if f in all_shapes])

    # Log excluded features
    all_features_in_data = set(features_with_data.keys())
    excluded = all_features_in_data - set(colour_features) - set(shape_features)
    if excluded:
        log.warning(f"Excluded features (not in standard colours/shapes): {excluded}")

    ordered_features = colour_features + shape_features

    if not ordered_features:
        log.warning(f"No standard features found for ordering effects plot ({variant}), skipping")
        return

    # Calculate averages
    avg_stage1 = []
    se_stage1 = []
    avg_stage2 = []
    se_stage2 = []

    for feat in ordered_features:
        s1_scores = features_with_data[feat]["stage1"]
        s2_scores = features_with_data[feat]["stage2"]

        avg_stage1.append(np.mean(s1_scores) if s1_scores else 0.0)
        se_stage1.append(np.std(s1_scores, ddof=1) / np.sqrt(len(s1_scores)) if len(s1_scores) > 1 else 0.0)

        avg_stage2.append(np.mean(s2_scores) if s2_scores else 0.0)
        se_stage2.append(np.std(s2_scores, ddof=1) / np.sqrt(len(s2_scores)) if len(s2_scores) > 1 else 0.0)

    # Create plot with consistent bar width
    n_groups = len(ordered_features)
    n_bars_per_group = 2  # stage1, stage2
    figsize = compute_figsize_for_bars(n_groups, n_bars_per_group)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_groups)
    width = compute_bar_width(n_bars_per_group, n_groups=n_groups)

    color_stage1 = get_palette_color(0)
    color_stage2 = get_palette_color(1)

    # Add grid and horizontal line BEFORE bars (zorder control)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # Bars
    ax.bar(x - width/2, avg_stage1, width, label='Feature Present in First Goal',
           color=color_stage1, edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    ax.bar(x + width/2, avg_stage2, width, label='Feature Present in Second Goal',
           color=color_stage2, edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)

    # Add error bars for stage 1 bars
    ax.errorbar(x - width/2, avg_stage1, yerr=se_stage1,
                fmt='none', ecolor='black', capsize=3, capthick=1, zorder=3)

    # Add error bars for stage 2 bars
    ax.errorbar(x + width/2, avg_stage2, yerr=se_stage2,
                fmt='none', ecolor='black', capsize=3, capthick=1, zorder=3)

    # Styling (font sizes +4 points)
    ax.set_ylabel('Marginalised Elo of Feature', fontsize=15)
    ax.set_xlabel('Feature', fontsize=15)
    ax.set_xticks(x)

    # Add separator between colours and shapes (dashed line)
    labels = ordered_features.copy()
    if colour_features and shape_features:
        separator_idx = len(colour_features)
        labels.insert(separator_idx, '|')
        # Adjust x positions for labels
        label_positions = list(x)
        label_positions.insert(separator_idx, separator_idx - 0.5)
        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels, fontsize=14)
        # Add vertical dashed line
        ax.axvline(x=separator_idx - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
    else:
        ax.set_xticklabels(labels, fontsize=14)

    # Style feature labels (colors in their color, shapes as symbols)
    style_feature_tick_labels(ax, 'x')

    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14, loc='upper left')

    # Numbers on bars removed per feedback (reduces visual clutter)

    # Title with variant info
    if show_title:
        variant_titles = {
            "all": "All Pipelines",
            "no_shared": "No Shared Features",
            "one_shared": "One Shared Feature"
        }
        title = f'Training Order Effects - {variant_titles[variant]}'
        plt.title(title, fontsize=16, pad=10)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"ordering_effects_{variant}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    log.info(f"Saved ordering effects plot to {filepath}")


def plot_value_inhibition(
    run_preferences: RunPreferences,
    output_dir: str = "local/plots/two_stage_analysis",
    filename: str = "value_inhibition.png",
    show_title: bool = True,
) -> None:
    """
    Plot how first-stage values inhibit second-stage learning.

    Strategy: Keep stage2 and non-shared stage1 feature constant, vary where the
    shared feature appears.

    Experimental pipeline: A[(f_i, f_j), (f_j, f_k)] where f_j is shared (in both stages)
    Control: Average over B[(f_i, f_n), (f_j, f_k)] where f_n ≠ f_j (f_j only in stage2)

    Measures:
    - Shared feature (f_j) strengthening: Is f_j valued more when in both stages vs only stage2?
    - Non-shared stage2 feature (f_k) inhibition: Is f_k valued less when another feature is shared?

    Shows 4 bars:
    1. Shared feature value - Experimental (shared in both stages)
    2. Shared feature value - Control (only in stage2)
    3. Non-shared stage2 feature value - Experimental (competing with shared feature)
    4. Non-shared stage2 feature value - Control (less competition)

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores)
        output_dir: Directory to save plots
        filename: Output filename
        show_title: If True, display the title. Set to False for figures with captions.
    """
    log.info("Generating value inhibition plot")

    # Data collection
    data = {
        'shared_pipeline': [],
        'shared_control': [],
        'non_shared_pipeline': [],
        'non_shared_control': [],
    }

    all_run_names = set(run_preferences.keys())
    skipped_no_control = 0
    skipped_wrong_sharing = 0
    processed = 0

    for run_name, prefs in run_preferences.items():
        # Parse pipeline
        parsed = parse_two_stage_pipeline(run_name)
        if parsed is None:
            log.warning(f"Invalid run name format for value inhibition: {run_name}")
            continue

        stage1_goal, stage2_goal = parsed
        n_shared = count_shared_features(stage1_goal, stage2_goal)

        # Only process pipelines with exactly 1 shared feature
        if n_shared != 1:
            skipped_wrong_sharing += 1
            continue

        # Identify shared feature
        shared_features = extract_features_from_goal(stage1_goal) & extract_features_from_goal(stage2_goal)
        assert len(shared_features) == 1
        shared_feature = list(shared_features)[0]

        # Find control pipelines
        control_names = find_control_pipelines_for_inhibition(
            stage1_goal, stage2_goal, shared_feature, all_run_names
        )
        if not control_names:
            skipped_no_control += 1
            log.debug(f"No controls found for {run_name}")
            continue

        # Collect scores for shared feature
        exp_shared_score = get_feature_score(prefs, shared_feature)
        if exp_shared_score is not None:
            control_shared_scores = [
                get_feature_score(run_preferences[cn], shared_feature)
                for cn in control_names
                if get_feature_score(run_preferences[cn], shared_feature) is not None
            ]
            if control_shared_scores:
                data['shared_pipeline'].append(exp_shared_score)
                data['shared_control'].append(np.mean(control_shared_scores))
                log.debug(f"Shared feature '{shared_feature}' in {run_name}: {len(control_shared_scores)} controls found")
                processed += 1

        # Collect scores for non-shared stage2 features
        stage2_features = extract_features_from_goal(stage2_goal)
        non_shared_stage2 = stage2_features - {shared_feature}

        for feature in non_shared_stage2:
            exp_score = get_feature_score(prefs, feature)
            if exp_score is not None:
                control_scores = [
                    get_feature_score(run_preferences[cn], feature)
                    for cn in control_names
                    if get_feature_score(run_preferences[cn], feature) is not None
                ]
                if control_scores:
                    data['non_shared_pipeline'].append(exp_score)
                    data['non_shared_control'].append(np.mean(control_scores))
                    log.debug(f"Non-shared feature '{feature}' in {run_name}: {len(control_scores)} controls found")

    # Log summary
    log.info(f"Value inhibition: processed {processed} pipelines with 1 shared feature, "
             f"skipped {skipped_no_control} (no control), "
             f"skipped {skipped_wrong_sharing} (wrong number of shared features)")

    # Check if we have any data
    if not any(data.values()):
        log.warning("No valid data for value inhibition plot, skipping")
        return

    # Calculate averages
    def safe_mean(lst):
        return np.mean(lst) if lst else 0.0

    def safe_se(lst):
        return np.std(lst, ddof=1) / np.sqrt(len(lst)) if len(lst) > 1 else 0.0

    avg_shared_pipeline = safe_mean(data['shared_pipeline'])
    avg_shared_control = safe_mean(data['shared_control'])
    avg_non_shared_pipeline = safe_mean(data['non_shared_pipeline'])
    avg_non_shared_control = safe_mean(data['non_shared_control'])

    se_shared_pipeline = safe_se(data['shared_pipeline'])
    se_shared_control = safe_se(data['shared_control'])
    se_non_shared_pipeline = safe_se(data['non_shared_pipeline'])
    se_non_shared_control = safe_se(data['non_shared_control'])

    # Create plot with consistent bar width
    n_groups = 2  # shared feature, non-shared feature
    n_bars_per_group = 2  # experimental, control
    figsize = compute_figsize_for_bars(n_groups, n_bars_per_group, short=True)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_groups)
    width = compute_bar_width(n_bars_per_group, n_groups=n_groups)

    color_experimental = get_palette_color(0)
    color_control = get_palette_color(1)

    # Add grid and horizontal line BEFORE bars (zorder control)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # Bars
    ax.bar(x[0] - width/2, avg_shared_pipeline, width,
           label='Shared Feature Present', color=color_experimental,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    ax.bar(x[0] + width/2, avg_shared_control, width,
           label='Shared Feature Substituted', color=color_control,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)

    ax.bar(x[1] - width/2, avg_non_shared_pipeline, width,
           color=color_experimental,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    ax.bar(x[1] + width/2, avg_non_shared_control, width,
           color=color_control,
           edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)

    # Add error bars for all 4 bars
    ax.errorbar([x[0] - width/2, x[0] + width/2, x[1] - width/2, x[1] + width/2],
                [avg_shared_pipeline, avg_shared_control, avg_non_shared_pipeline, avg_non_shared_control],
                yerr=[se_shared_pipeline, se_shared_control, se_non_shared_pipeline, se_non_shared_control],
                fmt='none', ecolor='black', capsize=4, capthick=1.5, zorder=3)

    ax.set_ylabel('Marginalised Elo of Feature', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Shared goal feature', 'Second goal\'s other feature'], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14, loc='lower left')

    if show_title:
        plt.title('Value Strengthening and Inhibition: Shared Feature Effects', fontsize=16, pad=20)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    log.info(f"Saved value inhibition plot to {filepath}")


# ============================================================================
# Orchestrator Function
# ============================================================================


def analyze_two_stage_pipelines(
    run_preferences: RunPreferences,
    plot_config: Optional[Dict] = None,
    output_dir: str = "local/plots/two_stage_analysis",
) -> None:
    """
    Orchestrate two-stage pipeline analysis based on config.

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores)
        plot_config: Configuration dict with keys:
            - 'value_persistence': bool (default True)
            - 'ordering_effects': bool (default True)
            - 'ordering_effects_variants': list of str (default ['all'])
            - 'value_inhibition': bool (default True)
            - 'show_title': bool (default True) - whether to display titles on plots
        output_dir: Directory to save plots
    """
    plot_config = plot_config or {}

    log.info(f"Starting two-stage pipeline analysis with {len(run_preferences)} runs")

    show_title = plot_config.get('show_title', True)

    # Value persistence plot
    if plot_config.get('value_persistence', True):
        try:
            plot_value_persistence(run_preferences, output_dir, show_title=show_title)
        except Exception as e:
            log.error(f"Error generating value persistence plot: {e}", exc_info=True)

    # Ordering effects plots
    if plot_config.get('ordering_effects', True):
        variants = plot_config.get('ordering_effects_variants', ['all'])
        for variant in variants:
            try:
                plot_ordering_effects(run_preferences, output_dir, variant, show_title=show_title)
            except Exception as e:
                log.error(f"Error generating ordering effects plot ({variant}): {e}", exc_info=True)

    # Value inhibition plot
    if plot_config.get('value_inhibition', True):
        try:
            plot_value_inhibition(run_preferences, output_dir, show_title=show_title)
        except Exception as e:
            log.error(f"Error generating value inhibition plot: {e}", exc_info=True)

    log.info("Two-stage pipeline analysis complete")
