import json
import logging
import os

from tqdm import tqdm
from typing import Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from research_scaffold.wandb_run_processing import get_runs_from_wandb, get_run_metrics


log = logging.getLogger(__name__)


Kwargs = dict[str, Any]


def wait_for_user(txt: str="\nPress enter to continue... "):
    input(txt)



def test_env_to_goal_names(env_string: str) -> tuple[str, str]:
    goal_0, goal_1 = env_string.split("_and_")
    return goal_to_goal_str(goal_0, 0), goal_to_goal_str(goal_1, 1)


def goal_to_goal_str(goal: str, index: int) -> str:
    return f"goal_{index}_{goal}"


def wandb_run_singleton_extractor(xs):
    if xs.size > 0:
        _, (val,) = xs
        return val.item()
    else:
        return None


def load_runs_from_cache(
    get_runs_from_wandb_kwargs: Kwargs,
    cache_dir: str = "local/cache",
    max_workers: int = 10,
    offline: bool = False,
) -> tuple[
    dict[
        str,  # Name of run
        dict[
            str,  # Name of env
            tuple[float, float] | None  # Rate of achieving first vs second goal
        ]
    ],
    list[str],  # Possible goals
]:
    """
    Load run environment metrics from cache, fetching missing data from wandb as needed.

    Args:
        get_runs_from_wandb_kwargs: Keyword arguments to pass to `get_runs_from_wandb`.
        cache_dir: Directory to use for caching run environment metrics.
        max_workers: Maximum number of worker threads to use for fetching data in parallel.
        offline: If True, load entirely from cache without contacting wandb API.

    Returns:
        A tuple containing:
            - A dictionary mapping run names to dictionaries of environment names and their corresponding
              rates of achieving the first and second goals.
            - A list of possible goal strings.
    """

    possible_goals = [
        f"{c}_{s}"
        for c in ['black', 'red', 'blue', 'green']
        for s in ['cross', 'plus', 'circle', 'ring', 'diamond', 'hollow-diamond']
    ]

    if offline:
        cache_path = Path(cache_dir)
        cache_file = cache_path / "run_env_metrics_cache.json"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Offline mode requires cached data at {cache_file}, but file not found."
            )
        log.info(f"Offline mode: loading all data from {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            run_env_metrics = {
                run_name: {
                    env_name: tuple(rates) if rates is not None else None
                    for env_name, rates in env_data.items()
                }
                for run_name, env_data in cached_data.items()
            }
        log.info(f"Loaded metrics for {len(run_env_metrics)} runs from cache (offline)")
        return run_env_metrics, possible_goals

    # If wandb_path wasn't set in the config, fall back to the WANDB_PATH env
    # var (typically loaded from .env at entrypoint). Lets configs stay
    # account-agnostic for open-sourcing; per-run overrides still work by
    # including `wandb_path:` in the config.
    if "wandb_path" not in get_runs_from_wandb_kwargs:
        env_path = os.environ.get("WANDB_PATH")
        if env_path is None:
            raise ValueError(
                "wandb_path not set: add `wandb_path: <entity>/<project>` to "
                "`get_runs_from_wandb_kwargs` in your config, set the WANDB_PATH "
                "env var (e.g. via a project-local `.env` file), or use "
                "`offline: true` to skip wandb entirely."
            )
        get_runs_from_wandb_kwargs = {**get_runs_from_wandb_kwargs, "wandb_path": env_path}

    # To-do: Handle runs via IDs and only raise warning for duplicate names
    runs = get_runs_from_wandb(**get_runs_from_wandb_kwargs)
    run_names = [run.name for run in runs]

    if len(set(run_names)) < len(run_names):
        log.warning("Duplicate run names detected:")
        for run_name in set(run_names):
            count = run_names.count(run_name)
            if count > 1:
                log.warning(f"  {run_name}: {count} occurrences")
        raise ValueError("Run names must be unique")

    log.info(f"Processing {len(runs)} runs")

    test_env_strings = [
        f"{goal_0}_and_{goal_1}"
        for goal_0 in possible_goals
        for goal_1 in possible_goals
    ]

    # Load cache
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "run_env_metrics_cache.json"
    
    run_env_metrics = {}

    def process_rates_from_cache(rates: list[float] | None) -> tuple[float, float] | None:
        if rates is None:
            return None
        r1, r2 = rates
        return (r1, r2)
    
    if cache_file.exists():
        log.info(f"Loading cached metrics from {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            # Convert lists back to tuples
            run_env_metrics = {
                run_name: {
                    env_name: process_rates_from_cache(rates)
                    for env_name, rates in env_data.items()
                }
                for run_name, env_data in cached_data.items()
            }
        log.info(f"Loaded metrics for {len(run_env_metrics)} runs from cache")
    
    # Determine which metrics need to be fetched
    runs_to_fetch = {}
    envs_to_fetch_per_run = {}

    # Alert for runs that are in cache but not in current runs
    for cached_run_name in run_env_metrics.keys():
        if cached_run_name not in run_names:
            log.warning(f"Cached run '{cached_run_name}' not found in current runs - it may have been deleted or renamed")
    
    for run, run_name in zip(runs, run_names):
        if run_name not in run_env_metrics:
            # New run - fetch all environments
            runs_to_fetch[run_name] = run
            envs_to_fetch_per_run[run_name] = set(test_env_strings)
            run_env_metrics[run_name] = {}
        else:
            # Existing run - check for missing environments
            missing_or_none_envs = {
                env for env in test_env_strings
                if env not in run_env_metrics[run_name] or run_env_metrics[run_name][env] is None
            }
            if missing_or_none_envs:
                runs_to_fetch[run_name] = run
                envs_to_fetch_per_run[run_name] = missing_or_none_envs

    if runs_to_fetch:
        log.info(f"Fetching metrics for {len(runs_to_fetch)} runs with missing data")
        
        # Build list of all fetch tasks
        fetch_tasks = []
        for test_env_string in test_env_strings:
            runs_needing_env = [
                (run_name, run) 
                for run_name, run in runs_to_fetch.items() 
                if test_env_string in envs_to_fetch_per_run[run_name]
            ]
            
            if runs_needing_env:
                fetch_tasks.append((test_env_string, runs_needing_env))
        
        log.info(f"Total fetch tasks: {len(fetch_tasks)}")
        
        def fetch_env_metrics(task):
            test_env_string, runs_needing_env = task
            goal_0_str, goal_1_str = test_env_to_goal_names(test_env_string)
            
            runs_subset = [run for _, run in runs_needing_env]
            run_names_subset = [run_name for run_name, _ in runs_needing_env]
            
            goal_0_rates = get_run_metrics(
                runs_subset,
                f"eval_{test_env_string}/touches/{goal_0_str}/mean",
                processing_fn=wandb_run_singleton_extractor,
            )
            goal_1_rates = get_run_metrics(
                runs_subset,
                f"eval_{test_env_string}/touches/{goal_1_str}/mean",
                processing_fn=wandb_run_singleton_extractor,
            )
            
            results = {}
            for run_name, goal_0_rate, goal_1_rate in zip(run_names_subset, goal_0_rates, goal_1_rates):
                results[run_name] = (test_env_string, (goal_0_rate, goal_1_rate))
            
            return results
        
        # Parallel fetch with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_env_metrics, task): task for task in fetch_tasks}
            
            with tqdm(total=len(fetch_tasks), desc="Fetching environment metrics") as pbar:
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        for run_name, (env_string, rates) in results.items():
                            run_env_metrics[run_name][env_string] = rates
                    except Exception as e:
                        task = futures[future]
                        log.error(f"Error fetching metrics for {task[0]}: {e}")
                    pbar.update(1)

        def process_rates_into_cache(rates: tuple[float | None, float | None] | None) -> list[float] | None:
            # Overall None - wasn't run, individually None - didn't achieve that goal at all so set to 0.0
            if rates is None:
                return None
            r1, r2 = rates
            return [r1 if r1 is not None else 0.0, r2 if r2 is not None else 0.0]
        
        # Save updated cache
        log.info(f"Saving updated cache to {cache_file}")
        cache_data = {
            run_name: {
                env_name: process_rates_into_cache(rates)
                for env_name, rates in env_data.items()
            }
            for run_name, env_data in run_env_metrics.items()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        log.info("Cache updated successfully")
    else:
        log.info("All metrics found in cache - no fetching needed")

    return run_env_metrics, possible_goals

