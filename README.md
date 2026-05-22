# Latent Policy Gradients & RL Generalisation

Reinforcement learning experiments on generalisation: training,
distilling, and analysing agents on configurable maze tasks, with computational
models of agent preferences.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then set WANDB_PATH=<entity>/<project>
```

Set `offline: true` in any analysis config to skip wandb and use the local
cache only.

## Run

```bash
# Train a PPO agent on the maze
uv run python main.py -c testing/configs/ppo_maze.yaml

# Multi-teacher distillation
uv run python main.py -c testing/configs/multi_distill_maze.yaml

# Fit a preference model + few-shot sweep
uv run python main.py -c analysis/models/lbr_few_shot_sweep_dense.yaml
```

Meta configs (sweeps over multiple runs):

```bash
uv run python main.py -m <meta_config.yaml>
```

## Layout

- `src/` — env, models, distillation, preference modelling.
- `analysis/` — analysis configs (model fitting, sweeps).
- `testing/configs/` — quick development configs.
- `local/` — outputs, caches, plots (gitignored).

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).
