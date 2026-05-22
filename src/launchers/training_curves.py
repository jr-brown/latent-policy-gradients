"""
Launcher function for plotting training curves from wandb runs.
"""

import logging
import re
from typing import Optional

from research_scaffold.wandb_run_processing import get_runs_from_wandb

from src.util import Kwargs
from src.training_curves import (
    parse_two_stage_pipeline_name,
    fetch_training_curves_with_names,
    combine_stage_curves,
    plot_training_curves,
)


log = logging.getLogger(__name__)


def plot_pipeline_training_curves(
    pipelines: list[str],
    get_runs_from_wandb_kwargs: Kwargs,
    output_path: str = "local/plots/training_curves.png",
    metric_key: str = "rollout/ep_rew_mean",
    x_key: str = "global_step",
    n_samples: int = 500,
    max_workers: int = 8,
    plot_kwargs: Optional[Kwargs] = None,
) -> None:
    """
    Plot training curves for specified pipelines.

    For two-stage pipelines (e.g., "maze_ppo_red_diamond_distill_then_blue_cross"),
    this function automatically finds both stage 1 (base training) and stage 2
    (finetune) runs and combines their curves with a stage transition marker.

    Args:
        pipelines: List of pipeline names to plot. Can include:
            - Single-stage: "maze_ppo_red_cross"
            - Two-stage: "maze_ppo_red_diamond_distill_then_blue_cross"
        get_runs_from_wandb_kwargs: Kwargs for fetching runs from wandb, including:
            - wandb_path: "username/project_name"
            - wandb_filters: Dict of filters (e.g., {"displayName": {"$regex": "^maze_"}})
        output_path: Path to save the plot (default: "local/plots/training_curves.png")
        metric_key: The metric to plot (default: "rollout/ep_rew_mean")
        x_key: The x-axis key for training steps (default: "global_step")
        n_samples: Number of samples per run (default: 500)
        max_workers: Number of parallel workers for fetching (default: 8)
        plot_kwargs: Optional kwargs passed to plot_training_curves, including:
            - figsize: (width, height) tuple
            - show_grid: bool
            - transition_marker_style: "vline", "point", or "both"
            - smooth_scaling: float for smoothing (passed to research_scaffold.plotting.smooth)
            - xlabel, ylabel, title: str
            - show_title: bool
    """
    plot_kwargs = plot_kwargs or {}

    # Fetch all runs from wandb
    log.info("Fetching runs from wandb...")
    runs = get_runs_from_wandb(**get_runs_from_wandb_kwargs)
    log.info(f"Found {len(runs)} runs")

    # Build a mapping from run name to run object
    run_by_name = {run.name: run for run in runs}

    # Determine which runs we need to fetch
    runs_to_fetch = set()
    two_stage_pipelines = {}  # Maps pipeline name -> (stage1_run_name, stage2_run_name)

    for pipeline in pipelines:
        # Check if this is a two-stage pipeline
        parsed = parse_two_stage_pipeline_name(pipeline)

        if parsed is not None:
            stage1_goal, stage2_goal = parsed

            # Stage 1 run: the base training run (e.g., "maze_ppo_red_diamond")
            stage1_run_name = f"maze_ppo_{stage1_goal}"

            # Stage 2 run: the finetune run (same as pipeline name)
            stage2_run_name = pipeline

            # Check if both runs exist
            if stage1_run_name not in run_by_name:
                log.warning(f"Stage 1 run '{stage1_run_name}' not found for pipeline '{pipeline}'")
                continue
            if stage2_run_name not in run_by_name:
                log.warning(f"Stage 2 run '{stage2_run_name}' not found for pipeline '{pipeline}'")
                continue

            two_stage_pipelines[pipeline] = (stage1_run_name, stage2_run_name)
            runs_to_fetch.add(stage1_run_name)
            runs_to_fetch.add(stage2_run_name)
        else:
            # Single-stage pipeline
            if pipeline not in run_by_name:
                log.warning(f"Run '{pipeline}' not found")
                continue
            runs_to_fetch.add(pipeline)

    if not runs_to_fetch:
        log.error("No valid pipelines found to plot")
        return

    # Fetch training curves for all needed runs
    log.info(f"Fetching training curves for {len(runs_to_fetch)} runs...")
    runs_subset = [run_by_name[name] for name in runs_to_fetch]
    all_curves = fetch_training_curves_with_names(
        runs=runs_subset,
        metric_key=metric_key,
        x_key=x_key,
        n_samples=n_samples,
        max_workers=max_workers,
    )
    log.info(f"Successfully fetched curves for {len(all_curves)} runs")

    # Build the final curves dict and stage transitions dict
    pipeline_curves = {}
    stage_transitions = {}

    for pipeline in pipelines:
        if pipeline in two_stage_pipelines:
            stage1_run_name, stage2_run_name = two_stage_pipelines[pipeline]

            if stage1_run_name not in all_curves:
                log.warning(f"No curve data for stage 1 run '{stage1_run_name}'")
                continue
            if stage2_run_name not in all_curves:
                log.warning(f"No curve data for stage 2 run '{stage2_run_name}'")
                continue

            # Combine the curves
            combined, transition_step = combine_stage_curves(
                all_curves[stage1_run_name],
                all_curves[stage2_run_name],
            )

            pipeline_curves[pipeline] = combined
            stage_transitions[pipeline] = transition_step
        else:
            # Single-stage pipeline
            if pipeline not in all_curves:
                log.warning(f"No curve data for run '{pipeline}'")
                continue
            pipeline_curves[pipeline] = all_curves[pipeline]

    if not pipeline_curves:
        log.error("No curves to plot after processing")
        return

    log.info(f"Plotting {len(pipeline_curves)} pipeline curves...")

    # Plot the curves
    plot_training_curves(
        pipeline_curves=pipeline_curves,
        stage_transitions=stage_transitions,
        output_path=output_path,
        **plot_kwargs,
    )
