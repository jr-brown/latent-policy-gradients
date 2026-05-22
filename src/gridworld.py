import pygame
import logging
import numpy as np

from enum import Enum
from typing import Tuple, Literal, Set, get_args
from itertools import product
from jaxtyping import Int, Bool
from collections import deque
from dataclasses import dataclass


log = logging.getLogger(__name__)


Array = np.ndarray
GenericInt = int | np.int64 | np.int32
IntPair = Tuple[GenericInt, GenericInt]
ObjectShape = Literal['circle', 'plus', 'cross', 'ring', 'square', 'triangle', 'diamond', 'hollow-diamond']
GridPosition = Int[Array, "2"]  # [row, col]
Colour = Tuple[int, int, int]  # RGB
RandomColour = Tuple[Colour, Colour]  # RGB box bounds

class GridDirection(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def direction_to_vector(direction: GridDirection) -> Int[Array, "2"]:
    """Convert a GridDirection to a movement vector."""
    mapping = {
        GridDirection.UP: np.array([-1, 0], dtype=np.int32),
        GridDirection.RIGHT: np.array([0, 1], dtype=np.int32),
        GridDirection.DOWN: np.array([1, 0], dtype=np.int32),
        GridDirection.LEFT: np.array([0, -1], dtype=np.int32),
    }
    return mapping[direction]


def description_to_shape_and_colour(
    description: str
) -> Tuple[ObjectShape, Colour]:
    """Map object description to shape and colour."""
    try:
        colour, shape = description.split('_')
    except ValueError:
        raise ValueError(f"Invalid object description format: {description}")

    colour = colour.lower()
    shape = shape.lower()

    colour_map: dict[str, Colour | RandomColour] = {
        'black': (0, 0, 0),
        'grey': (128, 128, 128),
        'white': (255, 255, 255),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'yellow': (255, 255, 0),
        'teal': (0, 128, 128),
        'purple': (128, 0, 128),
        'olive': (128, 128, 0),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'random': ((0, 0, 0), (255, 255, 255)),
        'random-green': ((0, 100, 0), (0, 255, 0)),
    }

    try:
        colour_rgb = colour_map[colour]
    except KeyError:
        raise ValueError(f"Unknown colour: {colour}, from description: {description}")

    if shape not in get_args(ObjectShape):
        raise ValueError(f"Unknown shape: {shape}, from description: {description}")

    # If colour is random, sample a random colour within the specified bounds
    if isinstance(colour_rgb, tuple) and isinstance(colour_rgb[0], tuple):
        assert len(colour_rgb) == 2, "Random colour must have two bounds"
        assert len(colour_rgb[0]) == 3 and len(colour_rgb[1]) == 3, "Colour bounds must be RGB tuples"
        colour_rgb = tuple(
            np.random.randint(low, high + 1) for low, high in zip(colour_rgb[0], colour_rgb[1])
        )

    assert isinstance(colour_rgb, tuple) and len(colour_rgb) == 3, "Final colour must be an RGB tuple"

    return shape, colour_rgb


@dataclass
class GridWorldObject:
    shape: ObjectShape
    pos: GridPosition
    color: Colour
    blocking: bool = False
    reward: float = 0.0
    terminating: bool = False
    is_agent: bool = False
    
    def draw(self, canvas: pygame.Surface, cell_pixel_length, shape_line_thickness) -> None:
        """Draw the object on the canvas."""
        # Convert grid position to pixel coordinates (center of cell)
        center_x: int = int(self.pos[1] * cell_pixel_length + cell_pixel_length // 2)
        center_y: int = int(self.pos[0] * cell_pixel_length + cell_pixel_length // 2)
        size: int = cell_pixel_length // 3
        
        if self.shape == 'circle':
            pygame.draw.circle(
                canvas,
                self.color,
                (center_x, center_y),
                size
            )
        
        elif self.shape == 'plus':
            # Draw horizontal line
            pygame.draw.line(
                canvas,
                self.color,
                (center_x - size, center_y),
                (center_x + size, center_y),
                shape_line_thickness
            )
            # Draw vertical line
            pygame.draw.line(
                canvas,
                self.color,
                (center_x, center_y - size),
                (center_x, center_y + size),
                shape_line_thickness
            )
        elif self.shape == 'cross':
            # Draw diagonal line \
            pygame.draw.line(
                canvas,
                self.color,
                (center_x - size, center_y - size),
                (center_x + size, center_y + size),
                shape_line_thickness
            )
            # Draw diagonal line /
            pygame.draw.line(
                canvas,
                self.color,
                (center_x - size, center_y + size),
                (center_x + size, center_y - size),
                shape_line_thickness
            )
        
        elif self.shape == 'ring':
            pygame.draw.circle(
                canvas,
                self.color,
                (center_x, center_y),
                size,
                shape_line_thickness  # Width parameter makes it a ring
            )
        
        elif self.shape == 'square':
            square_size: int = size * 2
            pygame.draw.rect(
                canvas,
                self.color,
                pygame.Rect(
                    center_x - size,
                    center_y - size,
                    square_size,
                    square_size
                )
            )
        
        elif self.shape == 'triangle':
            # Equilateral triangle pointing up
            height: int = int(size * 1.732)  # sqrt(3) * size
            points: list[Tuple[int, int]] = [
                (center_x, center_y - height // 2),  # Top
                (center_x - size, center_y + height // 2),  # Bottom left
                (center_x + size, center_y + height // 2),  # Bottom right
            ]
            pygame.draw.polygon(canvas, self.color, points)
        
        elif self.shape == 'diamond':
            # Diamond (rotated square)
            points: list[Tuple[int, int]] = [
                (center_x, center_y - size),  # Top
                (center_x + size, center_y),  # Right
                (center_x, center_y + size),  # Bottom
                (center_x - size, center_y),  # Left
            ]
            pygame.draw.polygon(canvas, self.color, points)

        elif self.shape == 'hollow-diamond':
            # Hollow diamond (rotated square outline)
            points: list[Tuple[int, int]] = [
                (center_x, center_y - size),  # Top
                (center_x + size, center_y),  # Right
                (center_x, center_y + size),  # Bottom
                (center_x - size, center_y),  # Left
            ]
            pygame.draw.polygon(
                canvas,
                self.color,
                points,
                shape_line_thickness  # Width parameter makes it hollow
            )

        else:
            raise ValueError(f"Unknown shape: {self.shape}")

@dataclass
class GridWorld:
    size: int
    cell_pixel_length: int
    shape_line_thickness: int
    obstacles: Bool[Array, "size size"]
    objects: dict[str, GridWorldObject]

    def render(self) -> pygame.Surface:
        pixel_size: int = self.cell_pixel_length * self.size
        canvas: pygame.Surface = pygame.Surface((pixel_size, pixel_size))
        canvas.fill((255, 255, 255))  # White background
        
        # Draw maze walls (black)
        for i, j in product(range(self.size), range(self.size)):
            if self.obstacles[i, j]:
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),  # Black
                    pygame.Rect(
                        j * self.cell_pixel_length,
                        i * self.cell_pixel_length,
                        self.cell_pixel_length,
                        self.cell_pixel_length,
                    ),
                )
        
        # Draw objects
        for gridworld_object in self.objects.values():
            gridworld_object.draw(canvas, self.cell_pixel_length, self.shape_line_thickness)
        
        return canvas

    def move(
        self,
        object_name: str,
        movement: GridDirection | GridPosition,
    ) -> Tuple[float, bool, list[str]]:

        obj: GridWorldObject = self.objects[object_name]
        current_pos: GridPosition = obj.pos.copy()
        
        # Determine target position
        if isinstance(movement, GridDirection):
            target_pos = current_pos + direction_to_vector(movement)
        else:
            target_pos = movement
        
        # Check bounds
        if (0 <= target_pos).all() and (target_pos < self.size).all():
            # Check for walls
            if not self.obstacles[*target_pos]:
                # Check for blocking objects
                for other_obj in self.objects.values():
                    if (other_obj.blocking and 
                        np.array_equal(other_obj.pos, target_pos)):
                        break
                else:
                    # Move the object
                    obj.pos = target_pos
        
        # After moving, check for rewards and termination
        reward: float = 0.0
        terminated: bool = False
        touched_objects: list[str] = []
        
        if obj.is_agent:
            for obj_name, other_obj in self.objects.items():
                if np.array_equal(other_obj.pos, obj.pos) and obj_name != object_name:
                    touched_objects.append(obj_name)
                    reward += other_obj.reward
                    if other_obj.terminating:
                        terminated = True
        
        return reward, terminated, touched_objects


def generate_random_obstacles(
    size: int,
    obstacle_density: float,
) -> Bool[Array, "size size"]:
    """Create a maze with random internal obstacles (no border walls)."""
    obstacles: Bool[Array, "size size"] = np.zeros((size, size), np.bool)
    
    # Add random internal obstacles based on density
    total_cells = size * size
    num_obstacles = int(total_cells * obstacle_density)
    
    # Randomly place obstacles
    for _ in range(num_obstacles):
        row: int = np.random.randint(0, size)
        col: int = np.random.randint(0, size)
        obstacles[row, col] = True
    
    return obstacles


def get_free_position(
    empty_cells: Bool[Array, "size size"],
) -> GridPosition:
    """Find a valid goal position."""
    free_positions: Int[Array, "n 2"] = np.argwhere(empty_cells)
    
    # Random goal position
    if len(free_positions) > 0:
        return free_positions[np.random.randint(len(free_positions))].astype(np.int32)
    else:
        raise ValueError("No free positions available.")


def verify_connectivity(obstacles: Bool[Array, "size size"]) -> bool:
    """
    Verify that all free cells are connected (i.e., there's a path 
    from any free cell to any other free cell using BFS).
    """
    size: int = obstacles.shape[0]
    free_cells: Int[Array, "n 2"] = np.argwhere(~obstacles)

    if len(free_cells) == 0:
        return False
    
    # Start BFS from first free cell
    start: IntPair = tuple(free_cells[0])  # type: ignore
    visited: Set[IntPair] = set()
    queue: deque[IntPair] = deque([start])
    visited.add(start)
    
    while queue:
        row, col = queue.popleft()
        
        # Check all 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row: GenericInt = row + dr
            new_col: GenericInt = col + dc
            
            if (0 <= new_row < size and 
                0 <= new_col < size and
                not obstacles[new_row, new_col] and
                (new_row, new_col) not in visited):
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))
    
    # All free cells should be reachable
    return len(visited) == len(free_cells)


def initialize_maze(
    size: int,
    obstacle_density: float,
    goals: list[str],
    distractors: list[str],
    agent: str,
    cell_pixel_length: int=50,
    shape_line_thickness: int=5,
    max_generation_attempts: int=10,
    goal_reward: float=1.0,
) -> GridWorld:

    # First generate a connected maze
    for _ in range(max_generation_attempts):
        obstacles = generate_random_obstacles(size, obstacle_density)
        if verify_connectivity(obstacles):
            break
    else:
        log.warning("Could not generate valid maze, using simple layout")
        obstacles = np.zeros((size, size), np.bool)
    
    # Generate goal and distractor positions, keeping track of empty cells
    empty_cells: Bool[Array, "size size"] = ~obstacles
    objects: dict[str, GridWorldObject] = {}

    for i, goal_obj_desc in enumerate(goals):
        shape, color = description_to_shape_and_colour(goal_obj_desc)
        pos = get_free_position(empty_cells)
        objects[f"goal_{i}_{goal_obj_desc}"] = GridWorldObject(
            shape=shape,
            pos=pos,
            color=color,
            blocking=False,
            reward=goal_reward,
            terminating=True,
            is_agent=False,
        )
        empty_cells[*pos] = False

    for i, distractor_obj_desc in enumerate(distractors):
        shape, color = description_to_shape_and_colour(distractor_obj_desc)
        pos = get_free_position(empty_cells)
        objects[f"distractor_{i}_{distractor_obj_desc}"] = GridWorldObject(
            shape=shape,
            pos=pos,
            color=color,
            blocking=False,
            reward=0.0,
            terminating=False,
            is_agent=False,
        )
        empty_cells[*pos] = False

    # Place the agent
    shape, color = description_to_shape_and_colour(agent)
    pos = get_free_position(empty_cells)
    objects["agent"] = GridWorldObject(
        shape=shape,
        pos=pos,
        color=color,
        blocking=False,
        reward=0.0,
        terminating=False,
        is_agent=True,
    )

    return GridWorld(
        size=size,
        cell_pixel_length=cell_pixel_length,
        shape_line_thickness=shape_line_thickness,
        obstacles=obstacles,
        objects=objects,
    )

