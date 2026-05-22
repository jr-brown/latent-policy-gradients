#!/usr/bin/env python3
"""
Generate a figure showing all shapes and colours used in the environment.

This script renders each shape as a 16x16 pixel image and displays circles
for each colour, with visual separation between feature categories.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

# Initialize pygame (required for rendering)
pygame.init()

# Constants matching the environment
CELL_PIXEL_LENGTH = 16
SHAPE_LINE_THICKNESS = 2

# Colours (RGB tuples)
COLOURS = {
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'grey': (128, 128, 128),
    'green': (0, 255, 0),
}

# All shapes used in the environment
SHAPES = ['cross', 'plus', 'diamond', 'ring', 'circle', 'hollow-diamond']

# Feature categories
TRAINING_EVAL_COLOURS = ['black', 'red', 'blue']  # Used in both training and evaluation
AGENT_ONLY_COLOURS = ['grey']  # Used only for rendering the agent
EVAL_ONLY_COLOURS = ['green']  # Used only in evaluation

TRAINING_EVAL_SHAPES = ['cross', 'plus', 'diamond', 'ring']
EVAL_ONLY_SHAPES = ['circle', 'hollow-diamond']

# Background colour for cells
BACKGROUND_COLOUR = (255, 255, 255)  # White


def render_shape(shape: str, colour: tuple[int, int, int]) -> np.ndarray:
    """Render a shape as a 16x16 pixel image and return as numpy array."""
    # Create a pygame surface
    surface = pygame.Surface((CELL_PIXEL_LENGTH, CELL_PIXEL_LENGTH))
    surface.fill(BACKGROUND_COLOUR)

    # Calculate center and size
    center_x = CELL_PIXEL_LENGTH // 2
    center_y = CELL_PIXEL_LENGTH // 2
    size = CELL_PIXEL_LENGTH // 3

    if shape == 'circle':
        pygame.draw.circle(surface, colour, (center_x, center_y), size)

    elif shape == 'plus':
        pygame.draw.line(surface, colour,
                        (center_x - size, center_y),
                        (center_x + size, center_y),
                        SHAPE_LINE_THICKNESS)
        pygame.draw.line(surface, colour,
                        (center_x, center_y - size),
                        (center_x, center_y + size),
                        SHAPE_LINE_THICKNESS)

    elif shape == 'cross':
        pygame.draw.line(surface, colour,
                        (center_x - size, center_y - size),
                        (center_x + size, center_y + size),
                        SHAPE_LINE_THICKNESS)
        pygame.draw.line(surface, colour,
                        (center_x - size, center_y + size),
                        (center_x + size, center_y - size),
                        SHAPE_LINE_THICKNESS)

    elif shape == 'ring':
        pygame.draw.circle(surface, colour, (center_x, center_y), size, SHAPE_LINE_THICKNESS)

    elif shape == 'diamond':
        points = [
            (center_x, center_y - size),
            (center_x + size, center_y),
            (center_x, center_y + size),
            (center_x - size, center_y),
        ]
        pygame.draw.polygon(surface, colour, points)

    elif shape == 'hollow-diamond':
        points = [
            (center_x, center_y - size),
            (center_x + size, center_y),
            (center_x, center_y + size),
            (center_x - size, center_y),
        ]
        pygame.draw.polygon(surface, colour, points, SHAPE_LINE_THICKNESS)

    # Convert pygame surface to numpy array
    array = pygame.surfarray.array3d(surface)
    # Pygame uses (width, height, channels), matplotlib expects (height, width, channels)
    array = np.transpose(array, (1, 0, 2))

    return array


def generate_features_figure(output_path: str = "local/plots/features_figure.png"):
    """Generate a figure showing all shapes and colours."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure with two rows: colours on top, shapes on bottom
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.5))

    # --- Row 1: Colours (3 categories) ---
    ax_colours = axes[0]
    ax_colours.set_xlim(-0.5, 7.5)
    ax_colours.set_ylim(-0.5, 1.5)
    ax_colours.set_aspect('equal')
    ax_colours.axis('off')

    colour_y = 0.5
    circle_radius = 0.35

    # Positions with spacing between sections
    pos = 0
    section_gap = 1.0  # Gap between sections
    eval_sep_x = 5.0  # Position of eval-only separator (aligned across rows)

    # Training & Eval colours
    training_eval_start = pos
    for i, colour_name in enumerate(TRAINING_EVAL_COLOURS):
        x = pos + i
        rgb = tuple(c / 255 for c in COLOURS[colour_name])
        circle = plt.Circle((x, colour_y), circle_radius, color=rgb, ec='black', linewidth=1)
        ax_colours.add_patch(circle)
        ax_colours.text(x, colour_y - 0.6, colour_name, ha='center', va='top', fontsize=11)
    training_eval_end = pos + len(TRAINING_EVAL_COLOURS) - 1
    pos += len(TRAINING_EVAL_COLOURS)

    # Separator line
    sep_x = pos - 0.5 + section_gap / 2
    ax_colours.axvline(x=sep_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    pos += section_gap

    # Agent-only colours
    agent_only_start = pos
    for i, colour_name in enumerate(AGENT_ONLY_COLOURS):
        x = pos + i
        rgb = tuple(c / 255 for c in COLOURS[colour_name])
        circle = plt.Circle((x, colour_y), circle_radius, color=rgb, ec='black', linewidth=1)
        ax_colours.add_patch(circle)
        ax_colours.text(x, colour_y - 0.6, colour_name, ha='center', va='top', fontsize=11)
    agent_only_end = pos + len(AGENT_ONLY_COLOURS) - 1
    pos += len(AGENT_ONLY_COLOURS)

    # Separator line
    sep_x = pos - 0.5 + section_gap / 2
    ax_colours.axvline(x=sep_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    pos += section_gap

    # Eval-only colours
    eval_only_start = pos
    for i, colour_name in enumerate(EVAL_ONLY_COLOURS):
        x = pos + i
        rgb = tuple(c / 255 for c in COLOURS[colour_name])
        circle = plt.Circle((x, colour_y), circle_radius, color=rgb, ec='black', linewidth=1)
        ax_colours.add_patch(circle)
        ax_colours.text(x, colour_y - 0.6, colour_name, ha='center', va='top', fontsize=11)
    eval_only_end = pos + len(EVAL_ONLY_COLOURS) - 1


    # Add category labels at top
    ax_colours.text((training_eval_start + training_eval_end) / 2, 1.2, 'Training & Eval',
                   ha='center', va='center', fontsize=10, style='italic', color='gray')
    ax_colours.text((agent_only_start + agent_only_end) / 2, 1.2, 'Agent-only',
                   ha='center', va='center', fontsize=10, style='italic', color='gray')
    ax_colours.text((eval_only_start + eval_only_end) / 2, 1.2, 'Eval-only',
                   ha='center', va='center', fontsize=10, style='italic', color='gray')

    # --- Row 2: Shapes (2 categories) ---
    ax_shapes = axes[1]
    ax_shapes.set_xlim(-0.5, 7.5)
    ax_shapes.set_ylim(-0.5, 1.5)
    ax_shapes.set_aspect('equal')
    ax_shapes.axis('off')

    shape_y = 0.5
    render_colour = (0, 0, 0)  # Render shapes in black

    pos = 0

    # Training & Eval shapes
    training_shapes_start = pos
    for i, shape_name in enumerate(TRAINING_EVAL_SHAPES):
        x = pos + i
        img = render_shape(shape_name, render_colour)
        imagebox = OffsetImage(img, zoom=2.5)
        ab = AnnotationBbox(imagebox, (x, shape_y), frameon=False)
        ax_shapes.add_artist(ab)
        ax_shapes.text(x, shape_y - 0.6, shape_name, ha='center', va='top', fontsize=11)
    training_shapes_end = pos + len(TRAINING_EVAL_SHAPES) - 1

    # Separator line - aligned with eval separator in colours row
    ax_shapes.axvline(x=eval_sep_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Eval-only shapes (positioned after the aligned separator)
    pos = eval_sep_x + 0.5  # Start after separator
    eval_shapes_start = pos
    for i, shape_name in enumerate(EVAL_ONLY_SHAPES):
        x = pos + i * 1.5  # Extra spacing between eval-only shapes
        img = render_shape(shape_name, render_colour)
        imagebox = OffsetImage(img, zoom=2.5)
        ab = AnnotationBbox(imagebox, (x, shape_y), frameon=False)
        ax_shapes.add_artist(ab)
        ax_shapes.text(x, shape_y - 0.6, shape_name, ha='center', va='top', fontsize=11)
    eval_shapes_end = pos + (len(EVAL_ONLY_SHAPES) - 1) * 1.5


    # Add category labels at top
    ax_shapes.text((training_shapes_start + training_shapes_end) / 2, 1.2, 'Training & Eval',
                   ha='center', va='center', fontsize=10, style='italic', color='gray')
    ax_shapes.text((eval_shapes_start + eval_shapes_end) / 2, 1.2, 'Eval-only',
                   ha='center', va='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved features figure to {output_path}")

    plt.close(fig)
    pygame.quit()


if __name__ == "__main__":
    generate_features_figure()
