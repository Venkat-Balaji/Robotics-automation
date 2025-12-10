# grid_input.py
"""
Fixed 10x10 grid (user-specified). Start & goal are entered by the user.
"""

import numpy as np
import math
from typing import Tuple

Position = Tuple[int, int]


def in_bounds(pos: Position, rows: int, cols: int) -> bool:
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols


def user_input_grid():
    """
    Returns the user-chosen fixed 10x10 grid, and reads start & goal from the user.
    """

    grid = np.array([
        [0,0,0,0,1,0,0,0,0,0],
        [0,1,1,0,1,0,1,1,1,0],
        [0,0,1,0,0,0,0,0,1,0],
        [0,0,1,1,1,1,1,0,1,0],
        [0,0,0,0,0,0,1,0,1,0],
        [1,1,1,1,1,0,1,0,1,0],
        [0,0,0,0,1,0,1,0,1,0],
        [0,1,1,0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0,0,0],
    ], dtype=int)

    rows, cols = grid.shape

    print("Using fixed 10x10 grid:")
    print(grid)

    # --- START INPUT ---
    while True:
        try:
            sr_sc = input("Enter START position (row col): ").strip().split()
            if len(sr_sc) != 2:
                raise ValueError
            sr, sc = int(sr_sc[0]), int(sr_sc[1])
            if not in_bounds((sr, sc), rows, cols):
                print("Start is out of bounds. Try again.")
                continue
            if grid[sr, sc] == 1:
                print("Start is on an obstacle. Choose a free cell.")
                continue
            break
        except ValueError:
            print("Invalid format. Example: 0 9")
        except Exception:
            print("Invalid input. Example: 0 9")

    # --- GOAL INPUT ---
    while True:
        try:
            gr_gc = input("Enter GOAL position (row col): ").strip().split()
            if len(gr_gc) != 2:
                raise ValueError
            gr, gc = int(gr_gc[0]), int(gr_gc[1])
            if not in_bounds((gr, gc), rows, cols):
                print("Goal is out of bounds. Try again.")
                continue
            if grid[gr, gc] == 1:
                print("Goal is on an obstacle. Choose a free cell.")
                continue
            break
        except ValueError:
            print("Invalid format. Example: 9 9")
        except Exception:
            print("Invalid input. Example: 9 9")

    print(f"Start = ({sr}, {sc})")
    print(f"Goal  = ({gr}, {gc})")
    return grid, (sr, sc), (gr, gc)


def inflate_obstacles(grid: np.ndarray, radius: int, metric: str = "euclidean") -> np.ndarray:
    """
    Inflate obstacles by `radius` cells. Returns new grid.
    radius == 0 -> returns original grid (copy).
    metric: 'euclidean' or 'manhattan'
    """
    rows, cols = grid.shape
    inflated = grid.copy()
    if radius <= 0:
        return inflated

    obstacles = np.argwhere(grid == 1)
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if metric == "euclidean":
                if math.hypot(dr, dc) <= radius + 1e-9:
                    offsets.append((dr, dc))
            else:
                if abs(dr) + abs(dc) <= radius:
                    offsets.append((dr, dc))

    for (r, c) in obstacles:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                inflated[nr, nc] = 1

    return inflated
