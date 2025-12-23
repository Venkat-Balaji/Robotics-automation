# grid_input.py
"""Fixed 10x10 grid (user-specified). Start & goal are entered by the user."""

import numpy as np
import math
from typing import Tuple

Position = Tuple[int, int]

def in_bounds(pos: Position, rows: int, cols: int) -> bool:
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols

def get_position(prompt: str, grid: np.ndarray, pos_type: str) -> Position:
    """Generic function to get and validate position input."""
    rows, cols = grid.shape
    while True:
        try:
            r, c = map(int, input(prompt).strip().split())
            if not in_bounds((r, c), rows, cols):
                print(f"{pos_type} is out of bounds. Try again.")
                continue
            if grid[r, c] == 1:
                print(f"{pos_type} is on an obstacle. Choose a free cell.")
                continue
            return r, c
        except (ValueError, IndexError):
            print(f"Invalid format. Example: {'0 9' if pos_type == 'Goal' else '0 0'}")

def user_input_grid() -> Tuple[np.ndarray, Position, Position]:
    """Returns the user-chosen fixed 10x10 grid, and reads start & goal from the user."""
    grid = np.array([[0,0,0,0,1,0,0,0,0,0], [0,1,1,0,1,0,1,1,1,0], [0,0,1,0,0,0,0,0,1,0],
                     [0,0,1,1,1,1,1,0,1,0], [0,0,0,0,0,0,1,0,1,0], [1,1,1,1,1,0,1,0,1,0],
                     [0,0,0,0,1,0,1,0,1,0], [0,1,1,0,1,0,1,0,1,0], [0,0,0,0,0,0,0,0,1,0],
                     [0,1,1,1,1,1,1,0,0,0]], dtype=int)
    print("Using fixed 10x10 grid:\n", grid)
    sr, sc = get_position("Enter START position (row col): ", grid, "Start")
    gr, gc = get_position("Enter GOAL position (row col): ", grid, "Goal")
    print(f"Start = ({sr}, {sc})\nGoal  = ({gr}, {gc})")
    return grid, (sr, sc), (gr, gc)

def inflate_obstacles(grid: np.ndarray, radius: int, metric: str = "euclidean") -> np.ndarray:
    """Inflate obstacles by `radius` cells. Returns new grid."""
    if radius <= 0:
        return grid.copy()
    rows, cols = grid.shape
    inflated = grid.copy()
    obstacles = np.argwhere(grid == 1)
    
    offsets = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
               if (metric == "euclidean" and math.hypot(dr, dc) <= radius + 1e-9) or 
                  (metric == "manhattan" and abs(dr) + abs(dc) <= radius)]
    
    for r, c in obstacles:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                inflated[nr, nc] = 1
    return inflated
