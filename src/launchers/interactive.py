from src.interactive.cartpole import InteractiveCartPole
from src.interactive.maze import InteractiveMaze


def play_cartpole(**kwargs):
    env = InteractiveCartPole(**kwargs)
    env.run()


def play_maze(**kwargs):
    env = InteractiveMaze(**kwargs)
    env.run()

