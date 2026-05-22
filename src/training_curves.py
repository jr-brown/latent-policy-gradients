"""
Training curves plotting for RL agent training pipelines.

This module provides functions to visualize training curves (e.g., rollout/ep_rew_mean)
for multiple agent training pipelines, with support for two-stage training where
stage transitions are clearly marked.
"""

import logging
import re
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from research_scaffold.plotting import smooth
from src.preference_analysis import get_palette_color, get_styled_goal_label


log = logging.getLogger(__name__)


def parse_two_stage_pipeline_name(run_name: str) -> Optional[tuple[str, str]]:
    """
    Parse a two-stage pipeline run name into stage 1 and stage 2 goals.

    Expected format: "maze_ppo_<colour1>_<shape1>_distill_then_<colour2>_<shape2>"

    Args:
        run_name: The run name to parse

    Returns:
        (stage1_goal, stage2_goal) as strings like ("red_diamond", "blue_cross"),
        or None if the name doesn't match the two-stage pattern.
    """
    # Pattern for two-stage finetune runs
    pattern = r"^maze_ppo_([a-z]+_[a-z]+)_distill_then_([a-z]+_[a-z]+)$"
    match = re.match(pattern, run_name)

    if match:
        stage1_goal, stage2_goal = match.groups()
        return (stage1_goal, stage2_goal)

    return None


def fetch_training_curves_with_names(
    runs: list,
    metric_key: str = "rollout/ep_rew_mean",
    x_key: str = "_step",
    n_samples: int = 500,
    max_workers: int = 8,
) -> dict[str, np.ndarray]:
    """
    Fetch training curves from wandb runs, returning a dict mapping run.name to curves.

    Each curve is a (2, n) array where row 0 is x values (steps) and row 1 is y values (metric).

    Args:
        runs: List of wandb Run objects
        metric_key: The metric to fetch (default: "rollout/ep_rew_mean")
        x_key: The x-axis key (default: "_step")
        n_samples: Number of samples to fetch per run (default: 500)
        max_workers: Number of parallel workers for fetching (default: 8)

    Returns:
        Dictionary mapping run name to (2, n) numpy array
    """
    from research_scaffold.wandb_run_processing import get_run_metrics

    results = {}

    def fetch_single_run(run):
        """Fetch metrics for a single run."""
        try:
            # get_run_metrics returns a list with one element per run
            curves = get_run_metrics(
                [run],
                y_key=metric_key,
                x_key=x_key,
                n_samples=n_samples,
                max_workers=1,
            )
            if curves and len(curves) > 0 and curves[0] is not None:
                curve = curves[0]
                # Ensure it's a 2D array
                if isinstance(curve, np.ndarray) and curve.ndim == 2:
                    return run.name, curve
        except Exception as e:
            log.warning(f"Failed to fetch metrics for run {run.name}: {e}")
        return run.name, None

    # Fetch in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_run, run): run for run in runs}

        with tqdm(total=len(runs), desc="Fetching training curves") as pbar:
            for future in as_completed(futures):
                run_name, curve = future.result()
                if curve is not None:
                    results[run_name] = curve
                pbar.update(1)

    return results


def combine_stage_curves(
    stage1_curve: np.ndarray,
    stage2_curve: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Combine two training curves by appending stage 2 after stage 1.

    Stage 2's x values are offset by the final x value of stage 1.

    Args:
        stage1_curve: (2, n1) array for stage 1
        stage2_curve: (2, n2) array for stage 2

    Returns:
        Tuple of (combined_curve, transition_step) where:
        - combined_curve is a (2, n1+n2) array
        - transition_step is the x value where stage 2 begins
    """
    # Get the final step of stage 1
    transition_step = int(stage1_curve[0, -1])

    # Offset stage 2 x values
    stage2_x_offset = stage2_curve.copy()
    stage2_x_offset[0, :] = stage2_curve[0, :] + transition_step

    # Combine
    combined = np.concatenate([stage1_curve, stage2_x_offset], axis=1)

    return combined, transition_step


def plot_training_curves(
    pipeline_curves: dict[str, np.ndarray],
    stage_transitions: dict[str, int],
    output_path: str,
    figsize: tuple[float | int, float | int] | list[float | int] = (8, 5),
    show_grid: bool = True,
    transition_marker_style: str = "vline",
    smooth_scaling: Optional[float] = None,
    xlabel: str = "Training Steps (millions)",
    ylabel: str = "Episode Reward (Mean)",
    title: Optional[str] = None,
    show_title: bool = True,
    transition_label: str = "Stage 2",
) -> None:
    """
    Plot training curves for multiple pipelines with optional stage transition markers.

    Args:
        pipeline_curves: Dict mapping pipeline name to (2, n) curve array
        stage_transitions: Dict mapping two-stage pipeline names to transition step
        output_path: Path to save the plot
        figsize: Figure size as (width, height)
        show_grid: Whether to show grid lines
        transition_marker_style: One of "vline", "point", or "both"
        smooth_scaling: Optional smoothing scaling parameter (passed to research_scaffold.plotting.smooth)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Optional plot title
        show_title: Whether to display the title
        transition_label: Label for the stage transition marker
    """
    from matplotlib.offsetbox import HPacker, VPacker, TextArea, AnchoredOffsetbox, DrawingArea
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    if not pipeline_curves:
        log.warning("No curves to plot")
        return

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Sort pipeline names for consistent ordering
    sorted_names = sorted(pipeline_curves.keys())

    # Collect data for custom legend
    legend_data = []  # List of (color, styled_segments)

    # Plot each curve
    for i, pipeline_name in enumerate(sorted_names):
        curve = pipeline_curves[pipeline_name]
        color = get_palette_color(i)

        # Apply smoothing if requested
        if smooth_scaling is not None:
            smoothed = smooth(curve, shape_scaling=smooth_scaling, output_as_dict=True)
            x_vals = smoothed['time'] / 1e6  # Scale to millions
            y_vals = smoothed['mean']
        else:
            x_vals = curve[0, :] / 1e6  # Scale to millions
            y_vals = curve[1, :]

        # Get styled label segments for legend
        # Extract the goal part from the pipeline name (remove "maze_ppo_" prefix)
        goal_str = re.sub(r"^maze_ppo_", "", pipeline_name)
        goal_str = re.sub(r"_distill$", "", goal_str)
        goal_str = goal_str.replace("_distill_then_", "_then_")

        segments = get_styled_goal_label(goal_str)
        legend_data.append((color, segments))

        # Plot the curve
        ax.plot(x_vals, y_vals, color=color, linewidth=2.0, alpha=0.9)

        # Add point marker at stage transition if this is a two-stage pipeline
        if pipeline_name in stage_transitions:
            transition_step = stage_transitions[pipeline_name] / 1e6  # Scale to millions

            if transition_marker_style in ("point", "both"):
                # Find the y value at or near the transition
                idx = np.searchsorted(x_vals, transition_step)
                if 0 <= idx < len(y_vals):
                    ax.scatter(
                        [transition_step],
                        [y_vals[idx]],
                        color=color,
                        s=60,
                        zorder=5,
                        marker='o',
                        edgecolors='black',
                        linewidth=0.5,
                    )

    # Add stage extent annotations for two-stage pipelines
    if stage_transitions:
        # Get the x-axis limits for positioning
        x_min, x_max = ax.get_xlim()

        # Fixed stage transition at 3M steps
        transition_step = 3.0  # 3 million steps
        annotation_color = 'black'

        # Warn if any actual stage boundary differs significantly from 3M
        tolerance = 0.5  # 500k steps tolerance
        for pipeline_name, actual_transition in stage_transitions.items():
            actual_scaled = actual_transition / 1e6
            if abs(actual_scaled - transition_step) > tolerance:
                log.warning(
                    f"Stage transition for '{pipeline_name}' is at {actual_scaled:.2f}M steps, "
                    f"expected ~{transition_step:.1f}M steps"
                )

        first_two_stage = next(
            (name for name in sorted_names if name in stage_transitions),
            None
        )

        if first_two_stage:

            # Position for the stage annotations (in axes coordinates for y)
            annotation_y = 1.04

            # Stage 1 bracket: from start to transition
            ax.annotate(
                '',
                xy=(x_min, annotation_y),
                xytext=(transition_step, annotation_y),
                xycoords=('data', 'axes fraction'),
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(
                    arrowstyle='<->',
                    color=annotation_color,
                    lw=1.5,
                    shrinkA=0,
                    shrinkB=0,
                ),
                annotation_clip=False,
            )
            # Stage 1 label
            ax.text(
                (x_min + transition_step) / 2,
                annotation_y + 0.04,
                'Stage 1',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color=annotation_color,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

            # Stage 2 bracket: from transition to end
            ax.annotate(
                '',
                xy=(transition_step, annotation_y),
                xytext=(x_max, annotation_y),
                xycoords=('data', 'axes fraction'),
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(
                    arrowstyle='<->',
                    color=annotation_color,
                    lw=1.5,
                    shrinkA=0,
                    shrinkB=0,
                ),
                annotation_clip=False,
            )
            # Stage 2 label
            ax.text(
                (transition_step + x_max) / 2,
                annotation_y + 0.04,
                'Stage 2',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color=annotation_color,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if show_grid:
        ax.grid(True, alpha=0.3)

    if show_title and title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Create custom legend with multi-colored text for styled goal labels
    legend_rows = []
    for line_color, segments in legend_data:
        # Create line swatch
        swatch = DrawingArea(20, 10, 0, 0)
        line = Line2D([0, 20], [5, 5], color=line_color, linewidth=2.0)
        swatch.add_artist(line)

        # Create text segments with colors
        text_areas = []
        for text, text_color in segments:
            ta = TextArea(text, textprops=dict(color=text_color, fontsize=11))
            text_areas.append(ta)

        # Pack text segments horizontally
        text_box = HPacker(children=text_areas, align='center', pad=0, sep=0)

        # Pack swatch and text together
        row = HPacker(children=[swatch, text_box], align='center', pad=0, sep=5)
        legend_rows.append(row)

    # Stack all rows vertically
    legend_content = VPacker(children=legend_rows, align='left', sep=4)

    # Anchor the legend box
    anchored_box = AnchoredOffsetbox(
        loc='lower right',
        child=legend_content,
        pad=0.5,
        frameon=True,
        bbox_to_anchor=(1, 0),
        bbox_transform=ax.transAxes,
    )
    anchored_box.patch.set_boxstyle("round,pad=0.3")
    anchored_box.patch.set_facecolor('white')
    anchored_box.patch.set_edgecolor('gray')
    anchored_box.patch.set_alpha(0.9)
    ax.add_artist(anchored_box)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    log.info(f"Saved training curves plot to {output_path}")
