import re
import numpy as np
import logging
import matplotlib.pyplot as plt

from typing import Tuple
from pathlib import Path
from matplotlib.patches import Polygon

from src.gridworld import description_to_shape_and_colour


log = logging.getLogger(__name__)

# Primary color palette for multi-run comparisons and two-stage analysis
# Uses golden ratio spacing for perceptually distinct colors
RUN_COMPARISON_PALETTE = "turbo"
PALETTE_OFFSET = 0.25
# PALETTE_OVERRIDE = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#1abc9c', '#9b59b6', '#f1c40f']
PALETTE_OVERRIDE = ['#4c92c3', '#ff993e', '#56b356', '#e67e22', '#1abc9c', '#9b59b6', '#f1c40f']
# PALETTE_OVERRIDE = None

# Note: The feature saliency diagonal plot uses hardcoded colors (#e74c3c for
# color features, #3498db for shape features)

# Golden ratio for perceptually distinct color selection
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# Bar chart layout constants for fixed-width figures
DEFAULT_BAR_FIG_WIDTH = 10.0     # Fixed figure width in inches
ASPECT_RATIO = 2/3              # Height = ASPECT_RATIO * width
SHORT_ASPECT_RATIO = 0.5  # For shorter plots

# Spacing parameters that scale with bar count
MIN_GROUP_WIDTH = 0.4           # For sparse plots (few bars)
MAX_GROUP_WIDTH = 0.8           # For dense plots (many bars)
SPARSE_BAR_THRESHOLD = 4        # Below this: maximum spacing
DENSE_BAR_THRESHOLD = 20        # Above this: minimum spacing


def compute_figsize_for_bars(
    n_groups: int,
    n_bars_per_group: int = 1,
    fig_width: float = DEFAULT_BAR_FIG_WIDTH,
    short: bool = False,
) -> Tuple[float, float]:
    """Compute figure size with fixed width and 2:3 aspect ratio."""
    fig_height = fig_width * (ASPECT_RATIO if not short else SHORT_ASPECT_RATIO)
    return (fig_width, fig_height)


def compute_bar_width(n_bars_per_group: int = 1, n_groups: int = 1) -> float:
    """
    Compute bar width with spacing that scales inversely with bar count.

    Fewer bars = more whitespace (wider spacing between groups).
    More bars = tighter packing.

    Args:
        n_bars_per_group: Number of bars at each x position
        n_groups: Number of bar group positions on x-axis

    Returns:
        Bar width in data coordinates for use with ax.bar()
    """
    total_bars = n_groups * n_bars_per_group

    # Compute group width based on total bars (smooth interpolation)
    if total_bars <= SPARSE_BAR_THRESHOLD:
        group_width = MIN_GROUP_WIDTH
    elif total_bars >= DENSE_BAR_THRESHOLD:
        group_width = MAX_GROUP_WIDTH
    else:
        # Smooth interpolation using cosine
        t = (total_bars - SPARSE_BAR_THRESHOLD) / (DENSE_BAR_THRESHOLD - SPARSE_BAR_THRESHOLD)
        group_width = MIN_GROUP_WIDTH + (MAX_GROUP_WIDTH - MIN_GROUP_WIDTH) * (0.5 * (1 - np.cos(np.pi * t)))

    return group_width / n_bars_per_group


def get_palette_color(index: int) -> Tuple[float, float, float, float] | str:
    """Get a color from the palette using golden ratio indexing."""
    if PALETTE_OVERRIDE is not None:
        return PALETTE_OVERRIDE[index % len(PALETTE_OVERRIDE)]
    else:
        palette = plt.get_cmap(RUN_COMPARISON_PALETTE)
        return palette((index + PALETTE_OFFSET) / GOLDEN_RATIO % 1)


# ============================================================
# Feature label styling for visual rendering of colors and shapes
# ============================================================

# Map color feature names to matplotlib colors
FEATURE_COLORS = {
    'black': 'black',
    'red': 'red',
    'blue': 'blue',
    'grey': 'grey',
    'green': 'green',
}

# Map shape feature names to unicode symbols
FEATURE_SHAPE_SYMBOLS = {
    'circle': '●',      # U+25CF filled circle
    'ring': '○',        # U+25CB hollow circle
    'diamond': '◆',     # U+25C6 filled diamond
    'hollow-diamond': '◇',  # U+25C7 hollow diamond
    'cross': '×',       # U+00D7 multiplication sign
    'plus': '+',        # plus sign
}


def get_styled_feature_label(feature: str) -> Tuple[str, str]:
    """
    Get styled label text and color for a feature.

    For color features (red, blue, etc.): returns the color name with that color.
    For shape features (cross, diamond, etc.): returns a unicode symbol in black.
    For other features: returns the original text in black.

    Args:
        feature: Feature name (e.g., 'red', 'cross', 'blue_diamond')

    Returns:
        Tuple of (display_text, color)
    """
    feature_lower = feature.lower()

    if feature_lower in FEATURE_COLORS:
        return feature_lower, FEATURE_COLORS[feature_lower]
    elif feature_lower in FEATURE_SHAPE_SYMBOLS:
        return FEATURE_SHAPE_SYMBOLS[feature_lower], 'black'
    else:
        return feature, 'black'


def parse_single_goal(goal_str: str) -> Tuple[str, str] | None:
    """
    Parse a single goal like 'red diamond' into (symbol, color).

    Returns None if the string doesn't match the expected pattern.
    """
    parts = goal_str.strip().split()
    if len(parts) == 2:
        color_part, shape_part = parts
        if color_part in FEATURE_COLORS and shape_part in FEATURE_SHAPE_SYMBOLS:
            return FEATURE_SHAPE_SYMBOLS[shape_part], FEATURE_COLORS[color_part]
    return None


def parse_goal_with_distractor(goal_str: str) -> Tuple[Tuple[str, str], Tuple[str, str]] | None:
    """
    Parse a goal with distractor like 'red cross with black ring distractor'.

    Returns ((goal_symbol, goal_color), (distractor_symbol, distractor_color))
    or None if the pattern doesn't match.
    """
    import re
    # Pattern: "color shape with color shape distractor"
    match = re.match(r'^(.+?) with (.+?) distractor$', goal_str.strip())
    if match:
        goal_part = match.group(1)
        distractor_part = match.group(2)
        goal_parsed = parse_single_goal(goal_part)
        distractor_parsed = parse_single_goal(distractor_part)
        if goal_parsed and distractor_parsed:
            return goal_parsed, distractor_parsed
    return None


def format_goal_with_distractor_segments(
    goal: Tuple[str, str],
    distractor: Tuple[str, str]
) -> list[Tuple[str, str]]:
    """
    Format a goal with distractor as styled segments: (goal, ¬distractor)

    Args:
        goal: (symbol, color) tuple for the goal
        distractor: (symbol, color) tuple for the distractor

    Returns:
        List of (text, color) segments for rendering
    """
    goal_symbol, goal_color = goal
    distractor_symbol, distractor_color = distractor
    return [
        ('(', 'black'),
        (goal_symbol, goal_color),
        (', ¬ ', 'black'),
        (distractor_symbol, distractor_color),
        (')', 'black'),
    ]


def get_styled_goal_label(goal: str) -> list[Tuple[str, str]]:
    """
    Get styled label segments for a goal name.

    Handles:
    - Single goals: 'red_diamond' -> [(◆, red)]
    - Goals with distractors: 'red_cross_with_black_ring_distractor' -> [(, black), (×, red), (, ¬, black), (○, black), (), black)]
    - Multi-stage goals: 'blue_diamond_then_red_cross' -> [(◆, blue), ( → , black), (×, red)]
    - Multi-stage with distractors in any stage

    Args:
        goal: Goal name (e.g., 'red_diamond', 'blue_diamond_then_red_cross',
              'red_cross_with_black_ring_distractor')

    Returns:
        List of (text, color) tuples representing segments to render.
    """
    # Normalize: replace underscores with spaces, lowercase
    goal_normalized = goal.lower().replace('_', ' ')

    # Check for multi-stage pattern "X then Y"
    if ' then ' in goal_normalized:
        stage_parts = goal_normalized.split(' then ')
        segments = []
        for i, stage in enumerate(stage_parts):
            if i > 0:
                segments.append((' → ', 'black'))

            # Check if this stage has a distractor
            distractor_parsed = parse_goal_with_distractor(stage)
            if distractor_parsed:
                goal_parsed, distractor = distractor_parsed
                segments.extend(format_goal_with_distractor_segments(goal_parsed, distractor))
            else:
                parsed = parse_single_goal(stage)
                if parsed:
                    segments.append(parsed)
                else:
                    # Fallback to plain text
                    segments.append((stage, 'black'))
        return segments

    # Check for distractor pattern (single stage)
    distractor_parsed = parse_goal_with_distractor(goal_normalized)
    if distractor_parsed:
        goal_parsed, distractor = distractor_parsed
        return format_goal_with_distractor_segments(goal_parsed, distractor)

    # Try to parse as single "color shape"
    parsed = parse_single_goal(goal_normalized)
    if parsed:
        return [parsed]

    # If not a color_shape combo, try as single feature
    text, color = get_styled_feature_label(goal)
    return [(text, color)]


def get_styled_goal_label_simple(goal: str) -> Tuple[str, str]:
    """
    Get a simple styled label for a goal (single color fallback).

    For multi-stage goals, returns concatenated symbols with arrow,
    using black as the color since matplotlib text can't mix colors.

    Args:
        goal: Goal name

    Returns:
        Tuple of (display_text, color)
    """
    segments = get_styled_goal_label(goal)
    if len(segments) == 1:
        return segments[0]
    else:
        # Multi-segment: concatenate text, use black
        text = ''.join(seg[0] for seg in segments)
        return text, 'black'


def style_feature_tick_labels(ax, axis: str = 'x') -> None:
    """
    Style tick labels for feature names with colors and shape symbols.

    Modifies tick labels in-place:
    - Color names (red, blue, etc.) are displayed in their color
    - Shape names (cross, diamond, etc.) are replaced with unicode symbols

    Args:
        ax: Matplotlib axes object
        axis: Which axis to style ('x' or 'y')
    """
    if axis == 'x':
        labels = ax.get_xticklabels()
    else:
        labels = ax.get_yticklabels()

    for label in labels:
        text = label.get_text()
        styled_text, color = get_styled_feature_label(text)
        label.set_text(styled_text)
        label.set_color(color)

    # Re-set the labels to apply text changes
    if axis == 'x':
        ax.set_xticklabels([l.get_text() for l in labels])
        for label, orig_label in zip(ax.get_xticklabels(), labels):
            label.set_color(orig_label.get_color())
    else:
        ax.set_yticklabels([l.get_text() for l in labels])
        for label, orig_label in zip(ax.get_yticklabels(), labels):
            label.set_color(orig_label.get_color())


def set_styled_feature_xticklabels(
    ax,
    features: list[str],
    rotation: int | float = 0,
    ha: str = 'center',
    fontsize: int = 12,
) -> None:
    """
    Set x-axis tick labels with styled colors and shape symbols.

    Args:
        ax: Matplotlib axes object
        features: List of feature names
        rotation: Label rotation angle
        ha: Horizontal alignment
        fontsize: Font size for labels
    """
    styled_labels = []
    colors = []
    for feature in features:
        styled_text, color = get_styled_feature_label(feature)
        styled_labels.append(styled_text)
        colors.append(color)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(styled_labels, rotation=rotation, ha=ha, fontsize=fontsize)

    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)


def set_styled_feature_yticklabels(
    ax,
    features: list[str],
    fontsize: int = 12,
) -> None:
    """
    Set y-axis tick labels with styled colors and shape symbols.

    Args:
        ax: Matplotlib axes object
        features: List of feature names
        fontsize: Font size for labels
    """
    styled_labels = []
    colors = []
    for feature in features:
        styled_text, color = get_styled_feature_label(feature)
        styled_labels.append(styled_text)
        colors.append(color)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(styled_labels, fontsize=fontsize)

    for label, color in zip(ax.get_yticklabels(), colors):
        label.set_color(color)


def set_multicolor_yticklabels(
    ax,
    labels_segments: list[list[Tuple[str, str]]],
    fontsize: int = 10,
    x_offset: float = -0.02,
) -> None:
    """
    Set y-axis tick labels with multiple colors per label.

    Each label is a list of (text, color) segments that are rendered
    adjacently to form a single multi-colored label.

    Args:
        ax: Matplotlib axes object
        labels_segments: List of labels, where each label is a list of
                        (text, color) tuples representing segments
        fontsize: Font size for labels
        x_offset: X position in axes coordinates (negative = left of axes)
    """
    from matplotlib.transforms import offset_copy

    n_labels = len(labels_segments)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels([''] * n_labels)  # Clear default labels

    # Get the renderer and figure for measuring text widths
    fig = ax.get_figure()

    for i, segments in enumerate(labels_segments):
        # Start position: to the left of the axes, at the y-tick position
        # Use blended transform: axes x-coords, data y-coords
        trans = ax.get_yaxis_transform()

        # Render segments right-to-left, ending at x_offset
        # First, calculate total width by creating temporary text objects
        total_text = ''.join(seg[0] for seg in segments)

        # Create a temporary text to measure where to start
        # We'll position at x_offset and use ha='right' for the full label
        temp_text = ax.text(x_offset, i, total_text, fontsize=fontsize,
                           transform=trans, ha='right', va='center')

        # Get the bounding box to find the left edge
        fig.canvas.draw_idle()
        try:
            bbox = temp_text.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(trans.inverted())
            start_x = bbox_data.x0
        except Exception:
            # Fallback if we can't get bbox
            start_x = x_offset - len(total_text) * 0.008

        temp_text.remove()

        # Now render each segment from left to right
        current_x = start_x
        for seg_text, seg_color in segments:
            text_obj = ax.text(current_x, i, seg_text, fontsize=fontsize,
                              transform=trans, ha='left', va='center',
                              color=seg_color)

            # Measure this segment's width and advance
            try:
                fig.canvas.draw_idle()
                seg_bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
                seg_bbox_data = seg_bbox.transformed(trans.inverted())
                current_x = seg_bbox_data.x1
            except Exception:
                # Fallback estimate
                current_x += len(seg_text) * 0.008


def plot_run_env_metrics(
    run_name: str,
    env_metrics: dict[str, tuple[float, float]],
    possible_goals: list[str],
    output_dir: str = "local/plots/pairwise_preferences",
    show_title: bool = True,
):
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log.debug(f"Saving plot to {output_path}")

    # Create a matrix plot for each run
    n_goals = len(possible_goals)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the grid
    for i in range(n_goals + 1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Fill in the cells
    for row_idx, row_goal in enumerate(possible_goals):
        for col_idx, col_goal in enumerate(possible_goals):
            test_env_string = f"{row_goal}_and_{col_goal}"
            
            if test_env_string in env_metrics:
                row_rate, col_rate = env_metrics[test_env_string]
                
                # Bottom-left triangle: row goal rate
                triangle = Polygon(
                    [(col_idx, row_idx), (col_idx, row_idx + 1), (col_idx + 1, row_idx + 1)],
                    facecolor=plt.cm.viridis(row_rate),
                    edgecolor='none'
                )
                ax.add_patch(triangle)
                
                # Top-right triangle: column goal rate
                triangle = Polygon(
                    [(col_idx, row_idx), (col_idx + 1, row_idx), (col_idx + 1, row_idx + 1)],
                    facecolor=plt.cm.viridis(col_rate),
                    edgecolor='none'
                )
                ax.add_patch(triangle)
    
    # Set up axes
    ax.set_xlim(0, n_goals)
    ax.set_ylim(0, n_goals)
    ax.set_aspect('equal')
    
    # Invert y-axis so first goal is at top
    ax.invert_yaxis()
    
    # Set labels
    ax.set_xticks(np.arange(n_goals) + 0.5)
    ax.set_yticks(np.arange(n_goals) + 0.5)
    ax.set_xticklabels(possible_goals, rotation=45, ha='left')
    ax.set_yticklabels(possible_goals)
    
    # Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_xlabel('Column Goal (top-right triangle)')
    ax.set_ylabel('Row Goal (bottom-left triangle)')

    if show_title:
        ax.set_title(f'Agent Goal Preferences: {run_name}')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Achievement Rate')
    
    # plt.tight_layout()
    
    # Save the figure
    filename = f"agent_preferences_{run_name}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log.debug(f"Saved plot to {filepath}")
    
    plt.close(fig)


def compute_btl_scores(
    env_metrics: dict[str, tuple[float, float]],
    possible_goals: list[str],
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-3,
    regularization: float = 0.0,
    normalise_to_no_goal: bool = True,
) -> dict[str, float]:
    """
    Compute Bradley-Terry-Luce scores for each goal based on achievement rates.
    
    This is more principled than ELO because:
    1. BTL is the maximum likelihood estimator for pairwise comparisons
    2. Naturally handles "no_goal" as a legitimate option
    3. Provides proper probabilistic interpretation
    
    Model: P(goal_i | {goal_i, goal_j, no_goal}) = exp(θᵢ) / Σ exp(θₖ)
    
    Args:
        env_metrics: Dictionary mapping test_env_string to (goal_0_rate, goal_1_rate)
        possible_goals: List of all possible goals
        max_iterations: Maximum iterations for MM algorithm
        convergence_threshold: Convergence tolerance
        regularization: L2 regularization strength (helps with sparse data)
        normalise_to_no_goal: Set no_goal strength to 0 (default: True)
    
    Returns:
        Dictionary mapping goal names (and "no_goal") to log-strength parameters
    """
    import numpy as np
    from collections import defaultdict
    
    # Initialize strengths (in log space for numerical stability)
    log_strengths = {goal: 0.0 for goal in possible_goals}
    log_strengths["no_goal"] = 0.0
    
    # Collect comparison data
    # Format: (winner, loser_set, weight) where loser_set can be a set of alternatives
    comparisons: list[tuple[str, set[str], float]] = []
    
    for test_env_string, (rate_0, rate_1) in env_metrics.items():
        goal_0, goal_1 = test_env_string.split("_and_")
        
        assert goal_0 in possible_goals, f"Unknown goal: {goal_0}"
        assert goal_1 in possible_goals, f"Unknown goal: {goal_1}"
        
        # Three outcomes:
        # 1. goal_0 achieved (rate_0)
        if rate_0 > 0:
            comparisons.append((goal_0, {goal_1, "no_goal"}, rate_0))
        
        # 2. goal_1 achieved (rate_1)
        if rate_1 > 0:
            comparisons.append((goal_1, {goal_0, "no_goal"}, rate_1))
        
        # 3. Neither goal achieved (1 - rate_0 - rate_1)
        neither_rate = 1.0 - rate_0 - rate_1
        if neither_rate > 0:
            comparisons.append(("no_goal", {goal_0, goal_1}, neither_rate))
    
    if not comparisons:
        log.warning("No valid comparisons found for BTL computation")
        return log_strengths
    
    log.debug(f"Computing BTL scores from {len(comparisons)} comparisons")
    
    # Use Minorization-Maximization (MM) algorithm (Hunter 2004)
    # This is guaranteed to converge to MLE and is numerically stable
    for iteration in range(max_iterations):
        # Compute wins and denominators for each option
        wins = defaultdict(float)
        denominators = defaultdict(float)
        
        for winner, losers, weight in comparisons:
            # Winner gets the weight as a "win"
            wins[winner] += weight
            
            # All options in the comparison contribute to denominators
            # Compute sum of strengths for normalization
            all_options = {winner} | losers
            strength_sum = sum(np.exp(log_strengths[opt]) for opt in all_options)
            
            for opt in all_options:
                # Add weight / (sum of all strengths) to denominator
                denominators[opt] += weight * np.exp(log_strengths[opt]) / strength_sum
        
        # MM update: new_strength = wins / denominator
        new_log_strengths = {}
        max_change = 0.0
        
        for goal in log_strengths:
            # Apply regularization (shrink towards 0)
            denom = denominators.get(goal, 0.0) + regularization
            
            if denom > 0:
                new_strength = wins.get(goal, 0.0) / denom
                new_log_strength = np.log(new_strength) if new_strength > 0 else -10.0  # Floor for numerical stability
            else:
                new_log_strength = -10.0  # Option never appeared
            
            max_change = max(max_change, abs(new_log_strength - log_strengths[goal]))
            new_log_strengths[goal] = new_log_strength
        
        log_strengths = new_log_strengths
        
        # Check convergence
        if iteration % 100 == 0 or max_change < convergence_threshold:
            log.debug(f"BTL iteration {iteration + 1}/{max_iterations}: max_change = {max_change:.6f}")
            
            if max_change < convergence_threshold:
                log.debug(f"BTL converged after {iteration + 1} iterations")
                break
    else:
        log.warning(f"BTL did not converge after {max_iterations} iterations (max_change = {max_change:.6f})")
    
    # Normalize to no_goal baseline if requested
    if normalise_to_no_goal:
        no_goal_strength = log_strengths["no_goal"]
        for goal in log_strengths:
            log_strengths[goal] -= no_goal_strength
    
    return log_strengths


def compute_elo_scores(
    env_metrics: dict[str, tuple[float, float]],
    possible_goals: list[str],
    k_factor: float = 32.0,
    initial_rating: float = 0.0,
    max_iterations: int = 1000,
    convergence_threshold: float = 0.1,
    num_convergence_checks: int = 10,
    convergence_check_offset: float = 10.0,
    learning_rate_decay: float = 0.99,
    min_k_factor: float = 1.0,
    normalise_to_no_goal: bool = True,
) -> dict[str, float]:
    """
    Compute ELO scores for each goal based on pairwise matchup rates.
    
    Args:
        env_metrics: Dictionary mapping test_env_string to (goal_0_rate, goal_1_rate)
        possible_goals: List of all possible goals
        k_factor: ELO K-factor (default: 32.0)
        initial_rating: Initial ELO rating for all goals (default: 0.0)
        max_iterations: Maximum number of iterations for convergence (default: 1000)
        convergence_threshold: Maximum rating change to consider converged (default: 0.1)
        convergence_check_interval: How often to check convergence (default: 100)
        use_batch_updates: Apply all updates at once to prevent oscillations (default: True)
        learning_rate_decay: Decay factor for k_factor per iteration (default: 0.95)
        min_k_factor: Minimum k_factor value (default: 1.0)
    
    Returns:
        Dictionary mapping goal names (and "no_goal" baseline) to ELO scores
    """
    # Initialize ELO ratings
    elo_ratings = {goal: initial_rating for goal in possible_goals}
    elo_ratings["no_goal"] = initial_rating
    
    # Collect all matchups
    matchups: list[tuple[str, str, float, float]] = []  # List of (goal_0, goal_1, weight, win_rate_0)
    no_goal_matchups: list[tuple[str, float, float]] = []  # List of (goal_0, weight, achievement_rate)
    
    for test_env_string, (rate_0, rate_1) in env_metrics.items():
        goal_0, goal_1 = test_env_string.split("_and_")

        assert goal_0 in possible_goals, f"Unknown goal: {goal_0}"
        assert goal_1 in possible_goals, f"Unknown goal: {goal_1}"
        
        # Add pairwise matchup (normalized win rate)
        total = rate_0 + rate_1
        assert 0 <= total <= 1, f"Total rate is not in (0,1] for matchup: {test_env_string}"

        if total > 0:
            win_rate_0 = rate_0 / total
            matchups.append((goal_0, goal_1, total, win_rate_0))
        
        # Add to no_goal baseline matchups
        # Model goal winrate as rate at which goal is achieved when other goal is not achieved
        if rate_1 < 1.0:
            weight = 1 - rate_1
            no_goal_matchups.append((goal_0, weight, rate_0 / weight))
        if rate_0 < 1.0:
            weight = 1 - rate_0
            no_goal_matchups.append((goal_1, weight, rate_1 / weight))
    
    if not matchups:
        log.warning("No valid matchups found for ELO computation")
        return elo_ratings
    
    log.debug(f"Computing ELO scores from {len(matchups)} pairwise matchups and {len(no_goal_matchups)} no_goal matchups")
    
    # Dynamic k_factor with decay
    current_k_factor = k_factor

    check_iters = np.astype(np.geomspace(
        1 + convergence_check_offset,
        max_iterations - 1 + convergence_check_offset,
        num_convergence_checks,
    ) - convergence_check_offset, np.int32).tolist()

    assert isinstance(check_iters, list)
    assert isinstance(check_iters[0], int)
    
    # Iterate until convergence
    for iteration in range(max_iterations):
        max_change = 0.0
        
        # Accumulate all updates before applying
        rating_deltas = {goal: 0.0 for goal in elo_ratings}
        matchup_counts = {goal: 0.0 for goal in elo_ratings}
        
        # Process all pairwise matchups
        for goal_0, goal_1, weight, win_rate_0 in matchups:
            rating_0 = elo_ratings[goal_0]
            rating_1 = elo_ratings[goal_1]
            
            # Expected score
            expected_0 = 1 / (1 + 10 ** ((rating_1 - rating_0) / 400))
            
            # Calculate change
            change_0 = weight * current_k_factor * (win_rate_0 - expected_0)
            rating_deltas[goal_0] += change_0
            rating_deltas[goal_1] -= change_0
            matchup_counts[goal_0] += weight
            matchup_counts[goal_1] += weight
        
        # Process no_goal baseline matchups
        for goal, weight, achievement_rate in no_goal_matchups:
            rating_goal = elo_ratings[goal]
            rating_no_goal = elo_ratings["no_goal"]
            
            # Expected score for goal vs no_goal
            expected_goal = 1 / (1 + 10 ** ((rating_no_goal - rating_goal) / 400))
            
            # Calculate change
            change_goal = weight * current_k_factor * (achievement_rate - expected_goal)
            rating_deltas[goal] += change_goal
            rating_deltas["no_goal"] -= change_goal
            matchup_counts[goal] += weight
            matchup_counts["no_goal"] += weight
        
        # Apply normalized updates
        for goal, delta in rating_deltas.items():
            count = matchup_counts[goal]
            if count > 0:
                normalized_delta = delta / np.sqrt(count)  # Normalize by sqrt of count to keep error of update constant wrt to counts
                elo_ratings[goal] += normalized_delta
                max_change = max(max_change, abs(normalized_delta))
        
        # Decay k_factor to reduce oscillations
        current_k_factor = max(min_k_factor, current_k_factor * learning_rate_decay)
        
        # Check convergence periodically
        if iteration in check_iters:
            log.debug(f"Iteration {iteration + 1}/{max_iterations}: max_change = {max_change:.4f}, k_factor = {current_k_factor:.4f}")
            
            if max_change < convergence_threshold:
                log.debug(f"Converged after {iteration + 1} iterations")
                break
    else:
        log.warning(f"Did not converge after {max_iterations} iterations (max_change = {max_change:.4f})")

    if normalise_to_no_goal:
        no_goal_rating = elo_ratings["no_goal"]
        for goal in elo_ratings:
            elo_ratings[goal] -= no_goal_rating
    
    return elo_ratings


def plot_scores(
    run_name: str,
    scores: dict[str, float],
    possible_goals: list[str],
    score_type: str,
    output_dir: str = "local/plots/scores",
    use_gridworld_colours: bool = False,
    show_title: bool = True,
):
    """
    Plot scores as a bar chart with baseline threshold.
    
    Args:
        run_name: Name of the run
        scores: Dictionary mapping goal names to scores
        possible_goals: List of possible goal names
        score_type: Type of scores being plotted (e.g., "ELO", "BTL")
        output_dir: Directory to save plots
        use_gridworld_colours: Whether to use gridworld colors for bars
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract baseline and goal scores
    baseline_score = scores.get("no_goal", 0.0)
    goal_scores = [(goal, scores.get(goal, 0.0)) for goal in possible_goals if goal in scores]
    
    # Sort by score descending
    goal_scores.sort(key=lambda x: x[1], reverse=True)
    
    if not goal_scores:
        log.warning(f"No goal scores to plot for run: {run_name}")
        return
    
    goals, score_values = zip(*goal_scores)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    def get_gridworld_colour_for_goal(goal: str):
        _, colour = description_to_shape_and_colour(goal)
        # Need to convert from (r, g, b) in [0, 255] to format appropriate for matplotlib
        colour = tuple(c / 255.0 for c in colour)
        # Soften colour
        colour = tuple((c+0.1)/1.1 for c in colour)
        return colour
    
    # Create color map for goals
    if use_gridworld_colours:
        colors = [get_gridworld_colour_for_goal(goal) for goal in goals]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(goals)))
    
    # Plot bars
    bars = ax.bar(range(len(goals)), score_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Plot baseline as dashed horizontal line
    ax.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, label='Seek no goal baseline')
    
    # Customize plot
    ax.set_xlabel('Goals', fontsize=12)
    ax.set_ylabel(f'{score_type} Score', fontsize=12)

    if show_title:
        ax.set_title(f'Goal {score_type} Scores: {run_name}', fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(goals)))
    ax.set_xticklabels(goals, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, score_values):
        height = bar.get_height()
        label = f"{score:.0f}" if abs(score) >= 10 else f"{score:.2g}"
        va = 'bottom' if height >= 0 else 'top'
        offset = 0 if height >= 0 else -0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset, label,
                ha='center', va=va, fontsize=9)
    
    # plt.tight_layout()
    
    # Save figure
    filename = f"{score_type.lower()}_scores_{run_name}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log.debug(f"Saved {score_type} plot to {filepath}")
    
    plt.close(fig)


def compute_shape_colour_preferences(
    scores: dict[str, float],
    zero_mean: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute average preference scores for each colour and shape separately.
    
    Args:
        scores: Dictionary mapping goal names (e.g., "red_cross") to scores
        zero_mean: If True, normalize scores so each group (colours, shapes) has mean 0
    
    Returns:
        Tuple of (colour_scores, shape_scores) where each maps to average score
    """
    from collections import defaultdict
    
    colour_scores_list = defaultdict(list)
    shape_scores_list = defaultdict(list)
    
    for goal_name, score in scores.items():
        if goal_name == "no_goal":
            continue

        try:
            colour_name, shape = goal_name.split('_')
        except ValueError:
            raise ValueError(f"Invalid object description format: {goal_name}")
        
        colour_scores_list[colour_name].append(score)
        shape_scores_list[shape].append(score)
    
    # Compute averages
    colour_scores = {
        colour: sum(score_list) / len(score_list)
        for colour, score_list in colour_scores_list.items()
    }
    shape_scores = {
        shape: sum(score_list) / len(score_list)
        for shape, score_list in shape_scores_list.items()
    }
    
    # Optionally normalize to zero mean
    if zero_mean:
        if colour_scores:
            colour_mean = sum(colour_scores.values()) / len(colour_scores)
            colour_scores = {k: v - colour_mean for k, v in colour_scores.items()}
        if shape_scores:
            shape_mean = sum(shape_scores.values()) / len(shape_scores)
            shape_scores = {k: v - shape_mean for k, v in shape_scores.items()}
    
    return colour_scores, shape_scores


def validate_elo_scores(
    run_env_metrics: dict[str, dict[str, tuple[float, float]]],
    possible_goals: list[str],
    n_folds: int = 4,
    seed: int = 42,
    directional_threshold: float = 0.10,
    elo_kwargs: dict | None = None,
) -> dict[str, dict]:
    """
    Validate Elo scores via K-fold cross-validation: for each agent, partition
    pairwise comparisons into K folds, fit Elo on K-1 folds, and predict on
    the held-out fold. Metrics are averaged across folds and then across agents.

    Metrics computed on held-out pairs:
    - KL divergence: KL(observed || predicted)
    - Total variation distance: 0.5 * sum(|obs - pred|)
    - Brier score: mean((pred - obs)^2) over the 2-way distribution
    - % directionally correct: fraction of held-out pairs where Elo
      correctly predicts which goal wins, excluding pairs where the
      observed rate difference is below directional_threshold

    Args:
        run_env_metrics: Dict mapping run names to env_metrics dicts
        possible_goals: List of all possible goal names
        n_folds: Number of cross-validation folds (default 4)
        seed: Random seed for reproducibility
        directional_threshold: Minimum absolute difference in observed
            normalised win rates to count a pair for directional accuracy
            (default 0.10 = 10 percentage points). Configurable.
        elo_kwargs: Optional kwargs passed to compute_elo_scores (e.g. k_factor)

    Returns:
        Dict with keys 'per_agent' mapping agent names to per-agent metric dicts,
        and 'summary' with mean/std/se across agents for each metric.
    """
    elo_kwargs = elo_kwargs or {}
    rng = np.random.default_rng(seed)

    agent_metrics: dict[str, list[dict[str, float]]] = {}

    for run_name, env_metrics in run_env_metrics.items():
        env_keys = list(env_metrics.keys())
        n_envs = len(env_keys)

        # Create fold assignments via shuffled indices
        shuffled = rng.permutation(n_envs)
        fold_assignments = np.zeros(n_envs, dtype=int)
        for fold_idx in range(n_folds):
            start = fold_idx * n_envs // n_folds
            end = (fold_idx + 1) * n_envs // n_folds
            fold_assignments[shuffled[start:end]] = fold_idx

        fold_metrics: list[dict[str, float]] = []

        for fold_idx in range(n_folds):
            test_indices = set(np.where(fold_assignments == fold_idx)[0].tolist())

            train_env = {
                k: v for i, (k, v) in enumerate(env_metrics.items())
                if i not in test_indices
            }
            test_env = {
                k: v for i, (k, v) in enumerate(env_metrics.items())
                if i in test_indices
            }

            # Fit Elo on training folds
            elo_scores = compute_elo_scores(
                env_metrics=train_env,
                possible_goals=possible_goals,
                **elo_kwargs,
            )

            # Evaluate on held-out fold
            kl_values = []
            tv_values = []
            brier_values = []
            directional_correct = []
            directional_total = 0

            for test_key, (rate_0, rate_1) in test_env.items():
                goal_0, goal_1 = test_key.split("_and_")
                total = rate_0 + rate_1
                if total <= 0:
                    continue

                obs_0 = rate_0 / total
                obs_1 = rate_1 / total

                elo_0 = elo_scores.get(goal_0, 0.0)
                elo_1 = elo_scores.get(goal_1, 0.0)
                pred_0 = 1.0 / (1.0 + 10.0 ** ((elo_1 - elo_0) / 400.0))
                pred_1 = 1.0 - pred_0

                eps = 1e-10
                kl = obs_0 * np.log((obs_0 + eps) / (pred_0 + eps)) + \
                     obs_1 * np.log((obs_1 + eps) / (pred_1 + eps))
                kl_values.append(float(kl))

                tv = 0.5 * (abs(obs_0 - pred_0) + abs(obs_1 - pred_1))
                tv_values.append(float(tv))

                brier = (pred_0 - obs_0) ** 2 + (pred_1 - obs_1) ** 2
                brier_values.append(float(brier))

                if abs(obs_0 - obs_1) >= directional_threshold:
                    directional_total += 1
                    obs_winner = 0 if obs_0 > obs_1 else 1
                    pred_winner = 0 if pred_0 > pred_1 else 1
                    directional_correct.append(int(obs_winner == pred_winner))

            metrics = {
                'kl': float(np.mean(kl_values)) if kl_values else float('nan'),
                'tv': float(np.mean(tv_values)) if tv_values else float('nan'),
                'brier': float(np.mean(brier_values)) if brier_values else float('nan'),
                'directional_accuracy': (
                    float(np.mean(directional_correct))
                    if directional_correct else float('nan')
                ),
                'n_test_pairs': len(test_env),
                'n_directional_pairs': directional_total,
            }
            fold_metrics.append(metrics)

        agent_metrics[run_name] = fold_metrics

    # Aggregate: per-agent means (across folds), then summary across agents
    metric_names = ['kl', 'tv', 'brier', 'directional_accuracy']
    per_agent_summary: dict[str, dict[str, float]] = {}

    for run_name, folds in agent_metrics.items():
        per_agent_summary[run_name] = {
            metric: float(np.nanmean([f[metric] for f in folds]))
            for metric in metric_names
        }

    # Cross-agent summary
    summary: dict[str, float] = {}
    for metric in metric_names:
        values = [v[metric] for v in per_agent_summary.values() if not np.isnan(v[metric])]
        if values:
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            summary[f'{metric}_se'] = float(np.std(values) / np.sqrt(len(values)))

    summary['n_agents'] = len(per_agent_summary)
    summary['n_folds'] = n_folds
    summary['directional_threshold'] = directional_threshold

    log.info(f"Elo {n_folds}-Fold CV Summary:")
    for metric in metric_names:
        mean_key = f'{metric}_mean'
        se_key = f'{metric}_se'
        if mean_key in summary:
            log.info(f"  {metric}: {summary[mean_key]:.4f} +/- {summary[se_key]:.4f}")

    return {
        'per_agent': per_agent_summary,
        'summary': summary,
    }


def plot_elo_vs_model_values(
    agent_elo_scores: dict[str, dict[str, float]],
    agent_goal_values: dict[str, dict[str, float]],
    possible_goals: list[str],
    output_dir: str = "local/plots/elo_vs_model",
    filename: str = "elo_vs_model_values.png",
    model_name: str = "model",
    show_title: bool = True,
) -> dict[str, float]:
    """
    Scatter plot comparing per-agent per-goal Elo scores with fitted model values.

    Each point is an (agent, goal) pair. Points are coloured and shaped by goal,
    using the goal's actual gridworld colour and a shape marker corresponding
    to its shape feature.

    Includes linear regression fit line and reports Spearman's rho.

    Args:
        agent_elo_scores: {agent_name: {goal_name: elo_score}}
        agent_goal_values: {agent_name: {goal_name: model_value}}
        possible_goals: List of all goal names
        output_dir: Output directory for the plot
        filename: Output filename
        model_name: Name of the model (for axis label)
        show_title: Whether to show plot title

    Returns:
        Dict with 'spearman_rho', 'spearman_p', 'r_squared', 'n_points'.
    """
    from scipy import stats as sp_stats

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Shape markers for gridworld shapes: (marker, hollow)
    # hollow=True means facecolor='none' so the shape appears as an outline
    shape_marker_specs: dict[str, tuple[str, bool]] = {
        'cross': ('x', False),
        'plus': ('P', False),
        'circle': ('o', False),
        'ring': ('o', True),             # hollow circle
        'diamond': ('D', False),
        'hollow-diamond': ('D', True),   # hollow diamond
    }

    def get_goal_colour(goal: str) -> tuple[float, ...]:
        """Get matplotlib-compatible colour from goal name."""
        _, colour = description_to_shape_and_colour(goal)
        colour = tuple(c / 255.0 for c in colour)
        # Soften very dark colours so they're visible
        colour = tuple(max(c, 0.15) for c in colour)
        return colour

    def get_goal_marker_spec(goal: str) -> tuple[str, bool]:
        """Get (marker, hollow) from goal shape."""
        _, shape = goal.split('_', 1)
        return shape_marker_specs.get(shape, ('o', False))

    # Collect all (elo, value) points grouped by goal
    common_agents = set(agent_elo_scores.keys()) & set(agent_goal_values.keys())

    all_elos = []
    all_values = []
    all_goals = []

    for agent in common_agents:
        elo_scores = agent_elo_scores[agent]
        model_values = agent_goal_values[agent]
        for goal in possible_goals:
            if goal in elo_scores and goal in model_values:
                all_elos.append(elo_scores[goal])
                all_values.append(model_values[goal])
                all_goals.append(goal)

    if not all_elos:
        log.warning("No overlapping (agent, goal) data for scatter plot")
        return {}

    elos_arr = np.array(all_elos)
    values_arr = np.array(all_values)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(elos_arr, values_arr)
    r_squared = r_value ** 2

    # Spearman correlation
    spearman_rho, spearman_p = sp_stats.spearmanr(elos_arr, values_arr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each goal as a separate series (grouped by colour for legend)
    plotted_goals = sorted(set(all_goals))

    for goal in plotted_goals:
        mask = [g == goal for g in all_goals]
        goal_elos = elos_arr[mask]
        goal_values = values_arr[mask]

        marker, hollow = get_goal_marker_spec(goal)
        colour = get_goal_colour(goal)

        marker_kwargs: dict = {}
        if marker == 'x':
            marker_kwargs['linewidths'] = 0.8
            marker_kwargs['s'] = 12
        else:
            marker_kwargs['s'] = 10
            marker_kwargs['edgecolors'] = 'grey'
            marker_kwargs['linewidths'] = 0.2

        if hollow:
            marker_kwargs['facecolors'] = 'none'
            marker_kwargs['edgecolors'] = [colour]
            marker_kwargs['linewidths'] = 0.6

        ax.scatter(
            goal_elos, goal_values,
            c=[colour] if not hollow else 'none',
            marker=marker,
            alpha=0.4,
            label=None,
            **marker_kwargs,
        )

    # Regression line
    x_range = np.linspace(elos_arr.min(), elos_arr.max(), 100)
    ax.plot(x_range, slope * x_range + intercept, 'k--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Elo Score', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)

    if show_title:
        ax.set_title(f'Elo vs {model_name} Values\n'
                     f'Spearman ρ={spearman_rho:.3f} (p={spearman_p:.2e}), '
                     f'R²={r_squared:.3f}',
                     fontsize=11)

    from matplotlib.lines import Line2D
    fit_handle = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                        label=f'Linear fit (R²={r_squared:.3f}), Spearman ρ={spearman_rho:.3f}')
    ax.legend(handles=[fit_handle], fontsize=9, loc='best')
    ax.grid(alpha=0.2)

    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved Elo vs model scatter plot to {filepath}")
    plt.close(fig)

    results = {
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'r_squared': float(r_squared),
        'slope': float(slope),
        'intercept': float(intercept),
        'n_points': len(all_elos),
    }

    log.info(f"Elo vs {model_name}: Spearman ρ={spearman_rho:.3f}, R²={r_squared:.3f}, "
             f"n={len(all_elos)} points")

    # Helper to make a scatter subplot (reused for raw, zero-mean, marginalised)
    def _make_scatter(
        ax_: plt.Axes,
        x: np.ndarray, y: np.ndarray, goals: list[str],
        x_label: str, y_label: str, title_extra: str,
        point_size: float = 10.0, alpha: float = 0.4,
        errorbar_data: dict | None = None,
    ) -> dict[str, float]:
        s_, i_, r_, _, _ = sp_stats.linregress(x, y)
        r2_ = r_ ** 2
        rho_, rho_p_ = sp_stats.spearmanr(x, y)

        unique_goals = sorted(set(goals))
        for goal in unique_goals:
            mask_g = np.array([g == goal for g in goals])
            marker, hollow = get_goal_marker_spec(goal)
            colour = get_goal_colour(goal)

            mkw: dict = {}
            if marker == 'x':
                mkw['linewidths'] = 0.8 if point_size <= 15 else 1.5
                mkw['s'] = point_size * 1.2
            else:
                mkw['s'] = point_size
                mkw['edgecolors'] = 'grey'
                mkw['linewidths'] = 0.2 if point_size <= 15 else 0.3
            if hollow:
                mkw['facecolors'] = 'none'
                mkw['edgecolors'] = [colour]
                mkw['linewidths'] = 0.6 if point_size <= 15 else 1.2

            if errorbar_data is not None:
                idx = [i for i, g in enumerate(goals) if g == goal]
                for j in idx:
                    ax_.errorbar(
                        x[j], y[j],
                        xerr=errorbar_data['xerr'][j], yerr=errorbar_data['yerr'][j],
                        fmt='none', ecolor='grey', elinewidth=0.8, capsize=2, alpha=0.5,
                    )

            ax_.scatter(
                x[mask_g], y[mask_g],
                c=[colour] if not hollow else 'none',
                marker=marker, alpha=alpha, label=None, zorder=3,
                **mkw,
            )

        xr = np.linspace(x.min(), x.max(), 100)
        ax_.plot(xr, s_ * xr + i_, 'k--', linewidth=1.5, alpha=0.7)
        ax_.set_xlabel(x_label, fontsize=12)
        ax_.set_ylabel(y_label, fontsize=12)
        if show_title:
            ax_.set_title(f'{title_extra}\nSpearman ρ={rho_:.3f} (p={rho_p_:.2e}), R²={r2_:.3f}',
                          fontsize=11)
        fit_h = Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                       label=f'Linear fit (R²={r2_:.3f}), Spearman ρ={rho_:.3f}')
        ax_.legend(handles=[fit_h], fontsize=9, loc='best')
        ax_.grid(alpha=0.2)
        return {'spearman_rho': float(rho_), 'spearman_p': float(rho_p_), 'r_squared': float(r2_)}

    # === Zero-mean normalised plot: subtract per-agent mean ===
    from collections import defaultdict

    # Group by agent to compute per-agent means
    agent_elo_list: dict[str, list[float]] = defaultdict(list)
    agent_val_list: dict[str, list[float]] = defaultdict(list)
    all_agents = []

    for agent in common_agents:
        for goal in possible_goals:
            if goal in agent_elo_scores[agent] and goal in agent_goal_values[agent]:
                all_agents.append(agent)
                agent_elo_list[agent].append(agent_elo_scores[agent][goal])
                agent_val_list[agent].append(agent_goal_values[agent][goal])

    agent_elo_means = {a: np.mean(v) for a, v in agent_elo_list.items()}
    agent_val_means = {a: np.mean(v) for a, v in agent_val_list.items()}

    # Build zero-meaned arrays (same ordering as all_elos/all_values/all_goals)
    norm_elos = np.zeros_like(elos_arr)
    norm_values = np.zeros_like(values_arr)
    idx = 0
    for agent in common_agents:
        for goal in possible_goals:
            if goal in agent_elo_scores[agent] and goal in agent_goal_values[agent]:
                norm_elos[idx] = elos_arr[idx] - agent_elo_means[agent]
                norm_values[idx] = values_arr[idx] - agent_val_means[agent]
                idx += 1

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    norm_stats = _make_scatter(
        ax3, norm_elos, norm_values, all_goals,
        'Elo Score (zero-mean per agent)', 'Predicted Value (zero-mean per agent)',
        f'Elo vs {model_name} Values (zero-mean normalised per agent)',
    )
    norm_filepath = output_path / filename.replace('.png', '_normalised.png')
    plt.savefig(norm_filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved zero-mean scatter plot to {norm_filepath}")
    plt.close(fig3)

    results['norm_spearman_rho'] = norm_stats['spearman_rho']
    results['norm_spearman_p'] = norm_stats['spearman_p']
    results['norm_r_squared'] = norm_stats['r_squared']

    # === Marginalised plot: average across agents per goal ===
    goal_elos_by_name: dict[str, list[float]] = defaultdict(list)
    goal_values_by_name: dict[str, list[float]] = defaultdict(list)

    for elo, value, goal in zip(all_elos, all_values, all_goals):
        goal_elos_by_name[goal].append(elo)
        goal_values_by_name[goal].append(value)

    marginal_goals = sorted(goal_elos_by_name.keys())
    mean_elos = np.array([np.mean(goal_elos_by_name[g]) for g in marginal_goals])
    mean_values = np.array([np.mean(goal_values_by_name[g]) for g in marginal_goals])
    se_elos = np.array([np.std(goal_elos_by_name[g]) / np.sqrt(len(goal_elos_by_name[g]))
                        for g in marginal_goals])
    se_values = np.array([np.std(goal_values_by_name[g]) / np.sqrt(len(goal_values_by_name[g]))
                          for g in marginal_goals])

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    eb_data = {'xerr': se_elos, 'yerr': se_values}
    marginal_stats = _make_scatter(
        ax4, mean_elos, mean_values, marginal_goals,
        'Mean Elo Score', 'Mean Predicted Value',
        f'Elo vs {model_name} Values (per goal, averaged over agents)',
        point_size=50.0, alpha=0.85, errorbar_data=eb_data,
    )
    marginal_filepath = output_path / filename.replace('.png', '_marginalised.png')
    plt.savefig(marginal_filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved marginalised scatter plot to {marginal_filepath}")
    plt.close(fig4)

    results['marginal_spearman_rho'] = marginal_stats['spearman_rho']
    results['marginal_spearman_p'] = marginal_stats['spearman_p']
    results['marginal_r_squared'] = marginal_stats['r_squared']
    results['n_goals'] = len(marginal_goals)

    return results


def plot_shape_colour_preferences(
    run_preferences: dict[str, tuple[dict[str, float], dict[str, float]]],
    output_dir: str = "local/plots/shape_colour_preferences",
    filename: str = "shape_colour_preferences.png",
    show_title: bool = True,
):
    """
    Plot a heatmap grid of shape and colour preferences across runs.
    
    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores) tuples
        output_dir: Directory to save the plot
        filename: Name of the output file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not run_preferences:
        log.warning("No run preferences to plot")
        return
    
    # Collect all unique colours and shapes across all runs
    all_colours = set()
    all_shapes = set()
    for colour_scores, shape_scores in run_preferences.values():
        all_colours.update(colour_scores.keys())
        all_shapes.update(shape_scores.keys())
    
    # Sort for consistent ordering
    colours = sorted(all_colours)
    shapes = sorted(all_shapes)
    attributes = colours + shapes
    
    # Extract run names and sort them
    run_names = sorted(run_preferences.keys())
    
    # Build the data matrix (runs x attributes)
    data_matrix = np.zeros((len(run_names), len(attributes)))
    
    for i, run_name in enumerate(run_names):
        colour_scores, shape_scores = run_preferences[run_name]
        
        # Fill in colour scores
        for j, colour in enumerate(colours):
            data_matrix[i, j] = colour_scores.get(colour, 0.0)
        
        # Fill in shape scores
        for j, shape in enumerate(shapes):
            data_matrix[i, len(colours) + j] = shape_scores.get(shape, 0.0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(10, len(attributes) * 0.8), max(6, len(run_names) * 0.5)))
    
    # Create heatmap with diverging colormap centered at 0
    vmax = np.abs(data_matrix).max()
    im = ax.imshow(data_matrix, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Set ticks and labels (colors in their color, shapes as symbols)
    set_styled_feature_xticklabels(ax, attributes)

    # Style y-axis run names as goals with multi-colored labels
    # e.g., "red_cross_with_black_ring_distractor" -> (×, ¬○) with proper colors
    styled_run_segments = [get_styled_goal_label(name) for name in run_names]
    set_multicolor_yticklabels(ax, styled_run_segments, fontsize=10)

    # Calculate y-axis label padding based on longest tick label
    max_label_chars = max(
        sum(len(text) for text, _ in segments)
        for segments in styled_run_segments
    ) if styled_run_segments else 0
    ylabel_padding = 15 + max_label_chars * 4  # Base padding + scaled by character count

    # Move x-axis to top
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')

    # Add separator line between colours and shapes
    if colours and shapes:
        ax.axvline(x=len(colours) - 0.5, color='black', linewidth=2)

    # Add text annotations
    for i in range(len(run_names)):
        for j in range(len(attributes)):
            value = data_matrix[i, j]
            #text_color = 'white' if value < (data_matrix.max() + data_matrix.min()) / 2 else 'black'
            text_color = 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    # Labels and title
    ax.set_xlabel('Feature (Colours | Shapes)', fontsize=12)
    ax.set_ylabel('Training Pipeline', fontsize=12, labelpad=ylabel_padding)

    if show_title:
        ax.set_title('Shape and Colour Preferences by Training Pipeline', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label(f"Score", fontsize=11)
    
    # Save figure
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved shape/colour preferences plot to {filepath}")
    

def plot_feature_saliency_matrix(
    run_preferences: dict[str, tuple[dict[str, float], dict[str, float]]],
    output_dir: str = "local/plots/feature_saliency",
    filename: str = "feature_saliency_matrix.png",
    show_title: bool = True,
):
    """
    Plot a feature saliency matrix showing how training on each feature affects preferences.

    Creates a matrix where:
    - Rows: Features that were present in training objectives (colours and shapes separately)
    - Columns: All preference features (colours and shapes)
    - Cell values: Average preference score for column feature across all agents
                   whose training included the row feature

    For example, if an agent was trained on "red_cross":
    - It contributes to rows: "red" and "cross"
    - Its preference scores populate the corresponding row entries

    Note: This function is designed for single-stage training runs only.

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores) tuples.
                        Run names should contain a single goal description like "red_cross".
        output_dir: Directory to save the plot
        filename: Name of the output file
        show_title: If True, display titles. Set to False for figures with captions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not run_preferences:
        log.warning("No run preferences to plot for feature saliency")
        return
    
    # Collect all unique colours and shapes from preferences
    all_colours: set[str] = set()
    all_shapes: set[str] = set()
    for colour_scores, shape_scores in run_preferences.values():
        all_colours.update(colour_scores.keys())
        all_shapes.update(shape_scores.keys())

    # Reorder colors and shapes to create nice diagonal alignment
    # Features only in eval (not training): green (colour), circle, hollow-diamond (shapes)
    # Move green to the FRONT of colours, and eval-only shapes to the END of shapes
    # This makes the diagonal blocks (training colors, training shapes) meet in the middle
    eval_only_colours = {'green'}
    eval_only_shapes = {'circle', 'hollow-diamond'}

    eval_colours = sorted([c for c in all_colours if c in eval_only_colours])
    training_colours = sorted([c for c in all_colours if c not in eval_only_colours])
    training_shapes = sorted([s for s in all_shapes if s not in eval_only_shapes])
    eval_shapes = sorted([s for s in all_shapes if s in eval_only_shapes])

    # Reorder: eval-only colours first, then training colours, then training shapes, then eval-only shapes
    colours = eval_colours + training_colours
    shapes = training_shapes + eval_shapes
    
    # Parse training features from run names and group preferences by training feature
    # Each feature (colour or shape) that appears in training gets its own row
    feature_preferences: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = {
        feature: [] for feature in colours + shapes
    }
    
    for run_name, (colour_scores, shape_scores) in run_preferences.items():
        # Extract goal description from run name (single goal: "colour_shape")
        match = re.match(r"^([a-z]+)_([a-z]+)", run_name)
        if not match:
            log.warning(f"Could not parse training goal from run name: {run_name}")
            continue
        
        colour, shape = match.group(1), match.group(2)
        
        # This run contributes to rows for both its colour and shape
        if colour in all_colours:
            feature_preferences[colour].append((colour_scores, shape_scores))
        if shape in all_shapes:
            feature_preferences[shape].append((colour_scores, shape_scores))
    
    # Filter to only features that have at least one run
    active_features = [f for f in colours + shapes if feature_preferences[f]]
    
    if not active_features:
        log.warning("No valid training features found in run names")
        return
    
    # Separate into training colours and training shapes for row ordering
    training_colours = [c for c in colours if feature_preferences[c]]
    training_shapes = [s for s in shapes if feature_preferences[s]]
    row_features = training_colours + training_shapes
    n_rows = len(row_features)
    
    # Columns are all preference features
    col_features = colours + shapes
    n_cols = len(col_features)
    
    # Build the saliency matrix (training_features x preference_features)
    saliency_matrix = np.zeros((n_rows, n_cols))
    saliency_se = np.zeros((n_rows, n_cols))  # SE matrix

    for i, train_feature in enumerate(row_features):
        prefs_list = feature_preferences[train_feature]

        # Average across all runs where this feature was in training
        for j, pref_feature in enumerate(col_features):
            if pref_feature in colours:
                values = [cs.get(pref_feature, 0.0) for cs, _ in prefs_list]
            else:
                values = [ss.get(pref_feature, 0.0) for _, ss in prefs_list]
            saliency_matrix[i, j] = np.mean(values) if values else 0.0

            # Compute SE
            if len(values) > 1:
                saliency_se[i, j] = np.std(values, ddof=1) / np.sqrt(len(values))
            else:
                saliency_se[i, j] = 0.0
    
    # Extract diagonal elements (training feature == preference feature)
    diagonal_features = []
    diagonal_values = []
    diagonal_errors = []
    for i, train_feature in enumerate(row_features):
        if train_feature in col_features:
            j = col_features.index(train_feature)
            diagonal_features.append(train_feature)
            diagonal_values.append(saliency_matrix[i, j])
            diagonal_errors.append(saliency_se[i, j])
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.8), max(8, n_rows * 0.6)))
    
    # Create heatmap with diverging colormap centered at 0
    vmax = np.abs(saliency_matrix).max()
    if vmax == 0:
        vmax = 1.0  # Avoid division issues
    im = ax.imshow(saliency_matrix, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Set ticks and labels with styled features (colors in their color, shapes as symbols)
    set_styled_feature_xticklabels(ax, col_features)
    set_styled_feature_yticklabels(ax, row_features)
    
    # Add separator lines between colours and shapes
    if colours and shapes:
        # Vertical separator between colour and shape columns
        ax.axvline(x=len(colours) - 0.5, color='black', linewidth=2)
    if training_colours and training_shapes:
        # Horizontal separator between colour and shape rows
        ax.axhline(y=len(training_colours) - 0.5, color='black', linewidth=2)
    
    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            value = saliency_matrix[i, j]
            text_color = 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                   color=text_color, fontsize=8)
    
    # Highlight diagonal elements (training feature matches preference feature)
    for i, train_feature in enumerate(row_features):
        if train_feature in col_features:
            j = col_features.index(train_feature)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                 fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    
    # Labels
    ax.set_xlabel('Preference Feature (Colours | Shapes)', fontsize=12)
    ax.set_ylabel('Training Feature (Colours | Shapes)', fontsize=12)

    if show_title:
        ax.set_title('Feature Saliency Matrix', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Average Preference Score", fontsize=11)
    
    # Save heatmap figure
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved feature saliency matrix to {filepath}")
    
    plt.close(fig)
    
    # Create diagonal bar chart

    # Separate colours and shapes for the bar chart
    diag_colours = [(f, v, e) for f, v, e in zip(diagonal_features, diagonal_values, diagonal_errors) if f in colours]
    diag_shapes = [(f, v, e) for f, v, e in zip(diagonal_features, diagonal_values, diagonal_errors) if f in shapes]

    # Combine with colours first, then shapes
    ordered_diag = diag_colours + diag_shapes
    diag_features_ordered = [f for f, _, _ in ordered_diag]
    diag_values_ordered = [v for _, v, _ in ordered_diag]
    diag_errors_ordered = [e for _, _, e in ordered_diag]
    
    n_diag = len(diag_features_ordered)

    # Use consistent bar width and figure size
    figsize_bar = compute_figsize_for_bars(n_diag, n_bars_per_group=1)
    fig_bar, ax_bar = plt.subplots(figsize=figsize_bar)
    bar_width = compute_bar_width(n_bars_per_group=1, n_groups=n_diag)

    # Color bars by whether they're colours or shapes (hardcoded for semantic distinction)
    COLOUR_BAR_COLOR = get_palette_color(0)
    SHAPE_BAR_COLOR = get_palette_color(1)
    bar_colors = [COLOUR_BAR_COLOR if f in colours else SHAPE_BAR_COLOR for f in diag_features_ordered]

    # Add horizontal line at y=0 BEFORE bars (zorder control)
    ax_bar.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # Add grid BEFORE bars (zorder control)
    ax_bar.grid(axis='y', alpha=0.3, zorder=1)

    bars = ax_bar.bar(
        range(n_diag),
        diag_values_ordered,
        width=bar_width,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        zorder=2,  # Bars go on top
    )

    # Add error bars for diagonal bars
    ax_bar.errorbar(range(n_diag), diag_values_ordered, yerr=diag_errors_ordered,
                    fmt='none', ecolor='black', capsize=4, capthick=1.5, zorder=4)

    # Add separator line between colours and shapes (dashed)
    if diag_colours and diag_shapes:
        ax_bar.axvline(x=len(diag_colours) - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)

    # Numbers above bars removed per feedback

    set_styled_feature_xticklabels(ax_bar, diag_features_ordered)
    ax_bar.set_xlabel('Feature', fontsize=12)
    ax_bar.set_ylabel('Marginalised Elo', fontsize=12)

    # Set y-ticks every 100 instead of default (every 50)
    y_min, y_max = ax_bar.get_ylim()
    y_tick_spacing = 100
    y_ticks = np.arange(
        np.floor(y_min / y_tick_spacing) * y_tick_spacing,
        np.ceil(y_max / y_tick_spacing) * y_tick_spacing + 1,
        y_tick_spacing
    )
    ax_bar.set_yticks(y_ticks)
    
    # Add legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOUR_BAR_COLOR, edgecolor='black', linewidth=0.5, label='Colours'),
        Patch(facecolor=SHAPE_BAR_COLOR, edgecolor='black', linewidth=0.5, label='Shapes'),
    ]
    ax_bar.legend(handles=legend_elements, loc='upper right', fontsize=12)

    if show_title:
        ax_bar.set_title('Feature Saliency (Diagonal Elements)', fontsize=14, fontweight='bold')

    # Save bar chart
    bar_filename = filename.replace('.png', '_diagonal.png')
    bar_filepath = output_path / bar_filename
    plt.savefig(bar_filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved diagonal saliency bar chart to {bar_filepath}")
    
    plt.close(fig_bar)


def plot_stage_comparison(
    run_preferences: dict[str, tuple[dict[str, float], dict[str, float]]],
    output_dir: str = "local/plots/stage_comparison",
    filename: str = "one_stage_vs_two_stage.png",
    show_title: bool = True,
):
    """
    Plot per-feature ELOs for agents trained on one stage vs two stages.

    Creates a bar chart with 4 bars for each feature:
    - One-stage training
    - Two-stage with 0 shared features
    - Two-stage with 1 shared feature
    - Two-stage with 2 shared features (same goal twice)

    Followed by summary bars showing the average across all features.

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores) tuples
        output_dir: Directory to save the plot
        filename: Output filename
        show_title: If True, display the title
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper function to parse two-stage pipeline and count shared features
    def parse_and_count_shared(run_name):
        """Parse two-stage run name and count shared features."""
        pattern = r"^([a-z]+)_([a-z]+)_then_([a-z]+)_([a-z]+)$"
        match = re.match(pattern, run_name)
        if not match:
            return None

        c1, s1, c2, s2 = match.groups()
        stage1_features = {c1, s1}
        stage2_features = {c2, s2}
        n_shared = len(stage1_features & stage2_features)
        return n_shared

    # Separate runs by stage and shared features
    one_stage_prefs = {}
    two_stage_0_shared = {}
    two_stage_1_shared = {}
    two_stage_2_shared = {}

    for run_name, prefs in run_preferences.items():
        if "distractor" in run_name:
            continue

        if "_then_" not in run_name:
            # One-stage training
            one_stage_prefs[run_name] = prefs
        else:
            # Two-stage training - categorize by shared features
            n_shared = parse_and_count_shared(run_name)
            if n_shared is None:
                log.warning(f"Could not parse two-stage run name: {run_name}")
                continue

            if n_shared == 0:
                two_stage_0_shared[run_name] = prefs
            elif n_shared == 1:
                two_stage_1_shared[run_name] = prefs
            elif n_shared == 2:
                two_stage_2_shared[run_name] = prefs

    if not one_stage_prefs:
        log.warning("No one-stage runs found for stage comparison")
        return

    # Collect all features
    all_colours = set()
    all_shapes = set()
    all_prefs = list(one_stage_prefs.values()) + list(two_stage_0_shared.values()) + \
                list(two_stage_1_shared.values()) + list(two_stage_2_shared.values())

    for colour_scores, shape_scores in all_prefs:
        all_colours.update(colour_scores.keys())
        all_shapes.update(shape_scores.keys())

    colours = sorted(all_colours)
    shapes = sorted(all_shapes)
    all_features = colours + shapes

    # Compute per-feature averages
    def compute_feature_avg(prefs_dict, feature, is_colour):
        scores = []
        for colour_scores, shape_scores in prefs_dict.values():
            if is_colour and feature in colour_scores:
                scores.append(colour_scores[feature])
            elif not is_colour and feature in shape_scores:
                scores.append(shape_scores[feature])

        if not scores:
            return np.nan, np.nan

        mean = np.mean(scores)
        se = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
        return mean, se

    one_stage_values = []
    one_stage_errors = []
    two_stage_0_values = []
    two_stage_0_errors = []
    two_stage_1_values = []
    two_stage_1_errors = []
    two_stage_2_values = []
    two_stage_2_errors = []

    for feature in all_features:
        is_colour = feature in colours
        mean, se = compute_feature_avg(one_stage_prefs, feature, is_colour)
        one_stage_values.append(mean)
        one_stage_errors.append(se)

        mean, se = compute_feature_avg(two_stage_0_shared, feature, is_colour)
        two_stage_0_values.append(mean)
        two_stage_0_errors.append(se)

        mean, se = compute_feature_avg(two_stage_1_shared, feature, is_colour)
        two_stage_1_values.append(mean)
        two_stage_1_errors.append(se)

        mean, se = compute_feature_avg(two_stage_2_shared, feature, is_colour)
        two_stage_2_values.append(mean)
        two_stage_2_errors.append(se)

    # Compute overall averages (ignoring NaN values)
    overall_one_stage = np.nanmean(one_stage_values)
    overall_two_stage_0 = np.nanmean(two_stage_0_values)
    overall_two_stage_1 = np.nanmean(two_stage_1_values)
    overall_two_stage_2 = np.nanmean(two_stage_2_values)

    # Compute overall SEs using error propagation: SE = sqrt(sum(SE_i^2)) / N
    def compute_overall_se(errors_list, values_list):
        valid_errors = [e for e, v in zip(errors_list, values_list) if not np.isnan(v) and not np.isnan(e)]
        n_valid = np.sum(~np.isnan(values_list))
        if n_valid > 0 and valid_errors:
            return np.sqrt(np.sum(np.array(valid_errors)**2)) / n_valid
        return 0.0

    overall_one_stage_se = compute_overall_se(one_stage_errors, one_stage_values)
    overall_two_stage_0_se = compute_overall_se(two_stage_0_errors, two_stage_0_values)
    overall_two_stage_1_se = compute_overall_se(two_stage_1_errors, two_stage_1_values)
    overall_two_stage_2_se = compute_overall_se(two_stage_2_errors, two_stage_2_values)

    # Replace NaN with 0 for plotting
    one_stage_values = [0 if np.isnan(v) else v for v in one_stage_values]
    one_stage_errors = [0 if np.isnan(e) else e for e in one_stage_errors]
    two_stage_0_values = [0 if np.isnan(v) else v for v in two_stage_0_values]
    two_stage_0_errors = [0 if np.isnan(e) else e for e in two_stage_0_errors]
    two_stage_1_values = [0 if np.isnan(v) else v for v in two_stage_1_values]
    two_stage_1_errors = [0 if np.isnan(e) else e for e in two_stage_1_errors]
    two_stage_2_values = [0 if np.isnan(v) else v for v in two_stage_2_values]
    two_stage_2_errors = [0 if np.isnan(e) else e for e in two_stage_2_errors]

    # Define colors
    color_one_stage = get_palette_color(0)
    color_2_shared = get_palette_color(1)
    color_1_shared = get_palette_color(2)
    color_0_shared = get_palette_color(3)

    # ========== Plot 1: Per-feature breakdown ==========
    n_features = len(all_features)
    n_bars_per_group = 4  # one-stage, 2 shared, 1 shared, 0 shared
    figsize_per_feature = compute_figsize_for_bars(n_features, n_bars_per_group, short=True)
    fig, ax = plt.subplots(figsize=figsize_per_feature)

    x = np.arange(n_features)
    width = compute_bar_width(n_bars_per_group, n_groups=n_features)

    # Add grid and horizontal line BEFORE bars
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # Plot per-feature bars (ordered: one-stage, 2 shared, 1 shared, 0 shared)
    bars1 = ax.bar(x - 1.5*width, one_stage_values, width,
                   label='One-Stage (2/2 unique)', color=color_one_stage,
                   edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    bars2 = ax.bar(x - 0.5*width, two_stage_2_values, width,
                   label='Two-Stage (2/4 unique)', color=color_2_shared,
                   edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    bars3 = ax.bar(x + 0.5*width, two_stage_1_values, width,
                   label='Two-Stage (3/4 unique)', color=color_1_shared,
                   edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)
    bars4 = ax.bar(x + 1.5*width, two_stage_0_values, width,
                   label='Two-Stage (4/4 unique)', color=color_0_shared,
                   edgecolor='black', linewidth=0.5, alpha=0.85, zorder=2)

    # Add error bars for all bar groups
    ax.errorbar(x - 1.5*width, one_stage_values, yerr=one_stage_errors,
                fmt='none', ecolor='black', capsize=3, capthick=1, zorder=3)
    ax.errorbar(x - 0.5*width, two_stage_2_values, yerr=two_stage_2_errors,
                fmt='none', ecolor='black', capsize=3, capthick=1, zorder=3)
    ax.errorbar(x + 0.5*width, two_stage_1_values, yerr=two_stage_1_errors,
                fmt='none', ecolor='black', capsize=3, capthick=1, zorder=3)
    ax.errorbar(x + 1.5*width, two_stage_0_values, yerr=two_stage_0_errors,
                fmt='none', ecolor='black', capsize=3, capthick=1, zorder=3)

    # Add separator line between colours and shapes
    if colours and shapes:
        ax.axvline(x=len(colours) - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)

    # Styling
    ax.set_ylabel('Marginalised Elo', fontsize=15)
    ax.set_xlabel('Feature', fontsize=15)
    set_styled_feature_xticklabels(ax, all_features, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12, loc='lower right')

    if show_title:
        ax.set_title('Feature Preferences by Training Stage (Per-Feature)', fontsize=16, pad=20)

    plt.tight_layout()

    # Save per-feature plot
    per_feature_filename = filename.replace('.png', '_per_feature.png')
    filepath = output_path / per_feature_filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved per-feature stage comparison plot to {filepath}")

    plt.close(fig)

    # ========== Plot 2: Average comparison ==========
    n_avg_groups = 4  # one-stage, 2 shared, 1 shared, 0 shared
    figsize_avg = compute_figsize_for_bars(n_avg_groups, n_bars_per_group=1, short=True)

    fig, ax = plt.subplots(figsize=figsize_avg)

    x_avg = np.arange(n_avg_groups)
    width_avg = compute_bar_width(n_bars_per_group=1, n_groups=4)

    # Use single color for all bars
    bar_color = get_palette_color(0)

    # Add grid and horizontal line BEFORE bars
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # Plot average bars with spacing
    bars1_avg = ax.bar(x_avg[0], overall_one_stage, width_avg,
                       color=bar_color,
                       edgecolor='black', linewidth=1.5, alpha=0.95, zorder=2)
    bars2_avg = ax.bar(x_avg[1], overall_two_stage_2, width_avg,
                       color=bar_color,
                       edgecolor='black', linewidth=1.5, alpha=0.95, zorder=2)
    bars3_avg = ax.bar(x_avg[2], overall_two_stage_1, width_avg,
                       color=bar_color,
                       edgecolor='black', linewidth=1.5, alpha=0.95, zorder=2)
    bars4_avg = ax.bar(x_avg[3], overall_two_stage_0, width_avg,
                       color=bar_color,
                       edgecolor='black', linewidth=1.5, alpha=0.95, zorder=2)

    # Add error bars for average bars
    ax.errorbar(x_avg, [overall_one_stage, overall_two_stage_2, overall_two_stage_1, overall_two_stage_0],
                yerr=[overall_one_stage_se, overall_two_stage_2_se, overall_two_stage_1_se, overall_two_stage_0_se],
                fmt='none', ecolor='black', capsize=4, capthick=1.5, zorder=3)

    # Styling
    ax.set_ylabel('Average Marginalised Elo', fontsize=15)
    ax.set_xticks(x_avg)
    ax.set_xticklabels([
        'One-Stage',
        'Two-Stage\n(2 Shared)',
        'Two-Stage\n(1 Shared)',
        'Two-Stage\n(0 Shared)'
    ], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if show_title:
        ax.set_title('Feature Preferences by Training Stage (Average)', fontsize=16, pad=20)

    plt.tight_layout()

    # Save average plot
    average_filename = filename.replace('.png', '_average.png')
    filepath_avg = output_path / average_filename
    plt.savefig(filepath_avg, dpi=300, bbox_inches='tight')
    log.info(f"Saved average stage comparison plot to {filepath_avg}")

    plt.close(fig)


def plot_grouped_preference_comparison(
    run_preferences: dict[str, tuple[dict[str, float], dict[str, float]]],
    group_name: str,
    output_dir: str = "local/plots/preference_comparisons",
    figsize: tuple[float, float] | None = None,
    show_arrows: bool = False,
    features_to_show: list[str] | None = None,
    show_title: bool = True,
):
    """
    Plot a grouped bar chart comparing shape and colour preferences across models.

    Each attribute (colour or shape) gets a cluster of bars, one per model,
    allowing direct comparison of how different models value each attribute.

    Args:
        run_preferences: Dictionary mapping run names to (colour_scores, shape_scores) tuples
        group_name: Name for this comparison group (used in title and filename)
        output_dir: Directory to save the plot
        figsize: Optional figure size (width, height). Auto-calculated if None.
        show_arrows: If True, draw arrows showing movement from each run to the next for each feature
        features_to_show: Optional list of specific features to display. If None, shows all features.
        show_title: If True, display the title. Set to False for figures with captions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not run_preferences or len(run_preferences) < 2:
        log.warning(f"Need at least 2 runs to compare, got {len(run_preferences)}")
        return
    
    # Collect all unique colours and shapes
    all_colours: set[str] = set()
    all_shapes: set[str] = set()
    for colour_scores, shape_scores in run_preferences.values():
        all_colours.update(colour_scores.keys())
        all_shapes.update(shape_scores.keys())
    
    colours = sorted(all_colours)
    shapes = sorted(all_shapes)
    attributes = colours + shapes

    # Filter to specified features if provided
    if features_to_show is not None:
        attributes = [f for f in attributes if f in features_to_show]
        colours = [c for c in colours if c in features_to_show]
        shapes = [s for s in shapes if s in features_to_show]

        if not attributes:
            log.warning(f"No matching features found from features_to_show: {features_to_show}")
            return

    n_attributes = len(attributes)

    run_names = sorted(run_preferences.keys())
    # Get styled label segments for run names (e.g., "red_diamond" -> [(◆, red)])
    # Each entry is a list of (text, color) tuples for multi-colored rendering
    styled_run_segments = [get_styled_goal_label(name) for name in run_names]
    n_runs = len(run_names)

    # Build data matrix (runs x attributes) and optionally zero-mean
    data = np.zeros((n_runs, n_attributes))
    for i, run_name in enumerate(run_names):
        colour_scores, shape_scores = run_preferences[run_name]
        for j, colour in enumerate(colours):
            data[i, j] = colour_scores.get(colour, 0.0)
        for j, shape in enumerate(shapes):
            data[i, len(colours) + j] = shape_scores.get(shape, 0.0)

    # Calculate figure size for consistent bar width across plots
    if figsize is None:
        figsize = compute_figsize_for_bars(n_attributes, n_runs)

    fig, ax = plt.subplots(figsize=figsize)

    # Bar positioning - consistent width per individual bar
    bar_width = compute_bar_width(n_runs, n_groups=n_attributes)
    x = np.arange(n_attributes)

    run_colors = [get_palette_color(i) for i in range(n_runs)]

    # Plot bars for each run (no label - we'll create custom legend)
    bars_list = []
    bar_positions = []  # Store bar positions for arrows
    for i, (run_name, color) in enumerate(zip(run_names, run_colors)):
        offset = (i - n_runs / 2 + 0.5) * bar_width
        positions = x + offset
        bar_positions.append(positions)
        bars = ax.bar(
            positions,
            data[i],
            bar_width,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.85,
        )
        bars_list.append(bars)

    # Add arrows showing movement between consecutive runs if requested
    if show_arrows and n_runs > 1:
        for attr_idx in range(n_attributes):
            for run_idx in range(n_runs - 1):
                x_start = bar_positions[run_idx][attr_idx]
                y_start = data[run_idx, attr_idx]
                x_end = bar_positions[run_idx + 1][attr_idx]
                y_end = data[run_idx + 1, attr_idx]

                # Draw arrow from top of current bar to top of next bar
                ax.annotate('',
                           xy=(x_end, y_end),
                           xytext=(x_start, y_start),
                           arrowprops=dict(
                               arrowstyle='->',
                               color='gray',
                               alpha=0.4,
                               linewidth=1.5,
                               connectionstyle='arc3,rad=0.1'
                           ))
    
    # Add separator line between colours and shapes (dashed for elegance)
    if colours and shapes:
        ax.axvline(x=len(colours) - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Customize axes (text +4 points)
    ax.set_xlabel('Feature', fontsize=16)
    ax.set_ylabel('Marginalised Elo', fontsize=16)

    if show_title:
        ax.set_title(f'Preference Comparison: {group_name}', fontsize=18, fontweight='bold')

    set_styled_feature_xticklabels(ax, attributes, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Create custom legend with multi-colored text for styled goal labels
    from matplotlib.offsetbox import HPacker, VPacker, TextArea, AnchoredOffsetbox, DrawingArea
    from matplotlib.patches import Rectangle

    legend_rows = []
    for i, (bar_color, segments) in enumerate(zip(run_colors, styled_run_segments)):
        # Create color swatch
        swatch = DrawingArea(15, 10, 0, 0)
        rect = Rectangle((0, 0), 15, 10, facecolor=bar_color, edgecolor='black', linewidth=0.5)
        swatch.add_artist(rect)

        # Create text segments with colors
        text_areas = []
        for text, color in segments:
            ta = TextArea(text, textprops=dict(color=color, fontsize=13))
            text_areas.append(ta)

        # Pack text segments horizontally
        text_box = HPacker(children=text_areas, align='center', pad=0, sep=0)

        # Pack swatch and text together
        row = HPacker(children=[swatch, text_box], align='center', pad=0, sep=5)
        legend_rows.append(row)

    # Add title
    title_area = TextArea('Training Pipeline', textprops=dict(fontsize=14))

    # Stack all rows vertically
    legend_content = VPacker(children=[title_area] + legend_rows, align='left', sep=5)

    # Anchor the legend box
    anchored_box = AnchoredOffsetbox(
        loc='upper right',
        child=legend_content,
        pad=0.5,
        frameon=True,
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
    )
    anchored_box.patch.set_boxstyle("round,pad=0.3")
    anchored_box.patch.set_facecolor('white')
    anchored_box.patch.set_edgecolor('gray')
    anchored_box.patch.set_alpha(0.9)
    ax.add_artist(anchored_box)

    ax.grid(axis='y', alpha=0.3)
    
    # Save figure
    filename = f"comparison_{group_name.replace(' ', '_')}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log.info(f"Saved preference comparison plot to {filepath}")
    
    plt.close(fig)
