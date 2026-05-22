from beartype.claw import beartype_this_package
beartype_this_package()

from gymnasium.envs.registration import register

register(
    id='Maze-v0',
    entry_point='src.maze_env:MazeEnv',
    max_episode_steps=200,
)
