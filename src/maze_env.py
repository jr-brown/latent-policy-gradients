import pygame
import logging
import numpy as np
import gymnasium as gym

from typing import Tuple, Optional, Dict, Any, Literal
from jaxtyping import Int, UInt
from gymnasium import spaces

from src.gridworld import GridWorld, GridDirection, initialize_maze


log = logging.getLogger(__name__)


Array = np.ndarray
GenericInt = int | np.int64 | np.int32 | Int[Array, ""]


FINISH_MAZE_REWARD: float = 1.0
STEP_REWARD: float = -0.1


MazeObservation = Dict[str, Int[Array, "..."]] | UInt[Array, "3 width height"] | Int[Array, "3 height width"]
ObservationMode = Literal["positions", "image"]


def sample_distractors(
    distractors: list[str],
    probability: float,
) -> list[str]:
    """Sample distractors based on specified probability."""
    sampled: list[str] = []
    for distractor in distractors:
        if np.random.rand() < probability:
            sampled.append(distractor)
    return sampled


class MazeEnv(gym.Env):
    """
    A simple gridworld environment with a maze and goal.
    
    The agent starts at a random position and must reach the goal.
    Walls are black, free space is white, agent is blue, goal color/shape configurable.
    
    Observations are dicts with keys: 'maze', 'agent_pos', 'goal_pos'
    """
    metadata: Dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 12}
    
    def __init__(
        self, 
        size: int = 8, 
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
        observation_mode: ObservationMode = "positions",
        obstacle_density: float = 0.2,
        agent: str = "black_circle",
        goals: list[str] | None = None,
        distractors: list[str] | None = None,
        distractor_probability: float = 1.0,
        goal_reward: float = FINISH_MAZE_REWARD,
        target_image_size: int | None = 128,
        cell_size: int | None = None,
        shape_line_thickness: int | None = None,
    ) -> None:
        super().__init__()

        if cell_size is not None and target_image_size is not None:
            effective_image_size = cell_size * size
            log.warning(f"Both cell_size {cell_size} and target_image_size {target_image_size} specified; using cell_size. Effective image size will be {effective_image_size}.")

        elif cell_size is None and target_image_size is not None:
            cell_size = target_image_size // size
            effective_image_size = cell_size * size
            log.debug(f"Computed cell_size {cell_size} from target_image_size {target_image_size} and size {size}.")

            if effective_image_size != target_image_size:
                log.warning(f"target_image_size {target_image_size} is not divisible by size {size}, resulting image_size {effective_image_size} instead.")

        elif cell_size is None and target_image_size is None:
            raise ValueError("Must specify at least one of cell_size or target_image_size.")

        assert cell_size is not None

        if shape_line_thickness is None:
            shape_line_thickness = max(1, cell_size // 8)
            log.debug(f"Computed shape_line_thickness {shape_line_thickness} from cell_size {cell_size}.")
        
        self.size: int = size
        self.obstacle_density: float = obstacle_density
        self.goals: list[str] = goals or ["grey_ring"]
        self.distractors: list[str] = distractors or []
        self.distractor_probability: float = distractor_probability
        self.agent: str = agent
        self.cell_size: int = cell_size
        self.shape_line_thickness: int = shape_line_thickness
        self.goal_reward: float = goal_reward

        self.maze: GridWorld = initialize_maze(
            size=self.size,
            obstacle_density=self.obstacle_density,
            goals=self.goals,
            distractors=sample_distractors(self.distractors, self.distractor_probability),
            agent=self.agent,
            goal_reward=self.goal_reward,
            cell_pixel_length=self.cell_size,
            shape_line_thickness=self.shape_line_thickness,
        )

        self.render_mode = render_mode
        self.observation_mode: ObservationMode = observation_mode
        self.has_warned_about_position_obs: bool = False
        
        # Pygame rendering
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # Define action and observation space
        # Actions: 0=up, 1=right, 2=down, 3=left (See GridDirection)
        self.action_space: spaces.Discrete = spaces.Discrete(4)
        
        # Observation is a dict with maze, agent position, and goal position
        # agent_pos and goal_pos are one-hot encoded grids
        if self.observation_mode == "positions":
            self.observation_space = spaces.Box(
                low=0, high=1, 
                shape=(3, size, size), 
                dtype=np.int32
            )
        elif self.observation_mode == "image":
            img_size = self.size * self.cell_size
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(3, img_size, img_size),  # CxWxH, though NatureCNN is CxHxW
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Invalid observation_mode: {observation_mode}")
    
    def _get_obs(self) -> MazeObservation:
        """Get current observation."""

        if self.observation_mode == "positions":
            agent_pos_onehot = np.zeros((self.size, self.size), dtype=np.int32)
            agent_pos_onehot[*self.maze.objects["agent"].pos] = 1
            
            if not self.has_warned_about_position_obs:
                log.warning(f"Position observations will only give the location of goals and not distractors (for now).")
                self.has_warned_about_position_obs = True

            # Get all goals
            goal_names = [name for name in self.maze.objects if name.startswith("goal_")]

            goal_positions = np.zeros((self.size, self.size), dtype=np.int32)

            for goal_name in goal_names:
                goal_positions[*self.maze.objects[goal_name].pos] = 1

            # Return stack of observations
            return np.stack([
                self.maze.obstacles.copy().astype(np.int32),
                agent_pos_onehot,
                goal_positions,
            ], axis=0)

        elif self.observation_mode == "image":
            return self._render_frame()

        else:
            raise ValueError("Invalid observation space configuration.")
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[MazeObservation, Dict[str, Any]]:
        """Reset environment state (mutates self.maze, self.goal_pos, self.agent_pos)."""
        super().reset(seed=seed, options=options)

        self.maze: GridWorld = initialize_maze(
            size=self.size,
            obstacle_density=self.obstacle_density,
            goals=self.goals,
            distractors=sample_distractors(self.distractors, self.distractor_probability),
            agent=self.agent,
            cell_pixel_length=self.cell_size,
            shape_line_thickness=self.shape_line_thickness,
            goal_reward=self.goal_reward,
        )

        observation: MazeObservation = self._get_obs()
        info: Dict[str, Any] = {}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(
        self, 
        action: GenericInt,
    ) -> Tuple[MazeObservation, float, bool, bool, Dict[str, Any]]:

        # Update agent position
        reward, terminated, touched_objects = self.maze.move("agent", GridDirection(action))

        if not terminated:
            reward += STEP_REWARD
        
        observation: MazeObservation = self._get_obs()
        info: Dict[str, Any] = {
            "touched_objects": touched_objects,
        }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info
    
    def render(self) -> Optional[UInt[Array, "3 width height"]]:
        """Public render interface (doesn't mutate state)."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> UInt[Array, "3 width height"]:
        """Render current state (mutates self.window and self.clock on first call)."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size * self.cell_size, self.size * self.cell_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = self.maze.render()
        
        if self.render_mode == "human":
            assert self.window is not None
            assert self.clock is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(2, 1, 0)
        ).astype(np.uint8)
    
    def close(self) -> None:
        """Clean up resources (mutates pygame state)."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # Example usage with random policy
    env: MazeEnv = MazeEnv(render_mode="human")
    
    observation, info = env.reset()
    
    for _ in range(1000):
        action: int = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
