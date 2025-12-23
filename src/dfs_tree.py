# dfs_tree.py
"""Tree-search DFS (iterative) with path-based cycle detection."""

import numpy as np
from typing import Tuple, List, Optional, Generator

Position = Tuple[int, int]

def neighbors(pos: Position, rows: int, cols: int, order: str = "URDL") -> Generator:
    r, c = pos
    mapping = {"U": (r - 1, c), "R": (r, c + 1), "D": (r + 1, c), "L": (r, c - 1)}
    for d in order:
        nr, nc = mapping[d]
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def dfs_tree_iterative(grid: np.ndarray, start: Position, goal: Position, order: str = "URDL",
                       max_depth: Optional[int] = None, max_expansions: Optional[int] = None) -> dict:
    """Iterative tree-search DFS with path-cycle checking."""
    rows, cols = grid.shape
    max_depth = max_depth or rows * cols
    max_expansions = max_expansions or rows * cols * 20

    stack = [(start, 0, [start])]
    explored_order, expansions = [], 0

    while stack:
        pos, depth, path = stack.pop()
        explored_order.append(pos)
        expansions += 1

        if expansions > max_expansions:
            return {"found": False, "path": None, "explored": explored_order, "expansions": expansions, "status": "max_expansions_reached"}
        if pos == goal:
            return {"found": True, "path": path, "explored": explored_order, "expansions": expansions, "status": "found"}
        if depth >= max_depth:
            continue

        neighs = [nb for nb in neighbors(pos, rows, cols, order=order) if grid[nb] == 0 and nb not in path]
        for nb in reversed(neighs):
            stack.append((nb, depth + 1, path + [nb]))

    return {"found": False, "path": None, "explored": explored_order, "expansions": expansions, "status": "exhausted"}
