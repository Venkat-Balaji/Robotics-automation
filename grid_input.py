"""
grid_input.py
Handles:
- Grid input from user
- Start & goal input
- Obstacle inflation for robot radius
"""

import numpy as np
import math
from typing import Tuple

Position = Tuple[int, int]


def in_bounds(pos: Position, rows: int, cols: int) -> bool:
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols


def user_input_grid():
    print("=== GRID SETUP ===")
    while True:
        try:
            rows = int(input("Enter number of rows (>1): "))
            cols = int(input("Enter number of columns (>1): "))
            if rows < 2 or cols < 2:
                print("Rows and columns must be >= 2.")
                continue
            break
        except:
            print("Invalid number. Try again.")

    grid = np.zeros((rows, cols), dtype=int)
    print("\nEnter grid row by row (0 = free, 1 = obstacle).")

    for r in range(rows):
        while True:
            row_str = input(f"Row {r} (space-separated {cols} values): ")
            try:
                vals = [int(x) for x in row_str.strip().split()]
                if len(vals) != cols:
                    print("Incorrect number of columns.")
                    continue
                if any(v not in [0, 1] for v in vals):
                    print("Only 0 or 1 allowed.")
                    continue
                grid[r] = vals
                break
            except:
                print("Invalid input.")

    print("\n=== START & GOAL SETUP ===")

    while True:
        try:
            sr, sc = [int(x) for x in input("Enter start (row col): ").split()]
            if not in_bounds((sr, sc), rows, cols) or grid[sr, sc] == 1:
                print("Invalid or blocked start cell.")
                continue
            break
        except:
            print("Invalid format.")

    while True:
        try:
            gr, gc = [int(x) for x in input("Enter goal (row col): ").split()]
            if not in_bounds((gr, gc), rows, cols) or grid[gr, gc] == 1:
                print("Invalid or blocked goal cell.")
                continue
            break
        except:
            print("Invalid format.")

    return grid, (sr, sc), (gr, gc)


def inflate_obstacles(grid: np.ndarray, radius: int, metric="euclidean"):
    rows, cols = grid.shape
    inflated = grid.copy()
    obstacles = np.argwhere(grid == 1)

    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if metric == "euclidean":
                if math.hypot(dr, dc) <= radius:
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
