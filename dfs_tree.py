# dfs_tree.py
"""
Tree-search DFS (iterative) with path-based cycle detection.

Behaviour:
- Tree-search (keeps full path on stack)
- Prevents cycles inside the current path by skipping any neighbor already present in that path
- Returns a dictionary with keys: found, path, explored, expansions, status
"""

import numpy as np
from typing import Tuple, List, Optional

Position = Tuple[int, int]


def neighbors(pos: Position, rows: int, cols: int, order: str = "URDL"):
    r, c = pos
    mapping = {"U": (r - 1, c), "R": (r, c + 1), "D": (r + 1, c), "L": (r, c - 1)}
    for d in order:
        yield mapping[d]


def dfs_tree_iterative(
    grid: np.ndarray,
    start: Position,
    goal: Position,
    order: str = "URDL",
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
):
    """
    Iterative tree-search DFS with path-cycle checking.

    - grid: 2D numpy array (0 free, 1 obstacle)
    - start, goal: (row, col)
    - order: neighbor expansion order string like "URDL"
    - max_depth: optional depth limit (defaults to rows*cols)
    - max_expansions: optional expansion cap (defaults to rows*cols*20)
    """

    rows, cols = grid.shape
    if max_depth is None:
        max_depth = rows * cols
    if max_expansions is None:
        max_expansions = rows * cols * 20

    # stack entries: (pos, depth, path_list)
    stack: List[Tuple[Position, int, List[Position]]] = [(start, 0, [start])]
    explored_order: List[Position] = []
    expansions = 0

    while stack:
        pos, depth, path = stack.pop()
        explored_order.append(pos)
        expansions += 1

        # safety check
        if expansions > max_expansions:
            return {
                "found": False,
                "path": None,
                "explored": explored_order,
                "expansions": expansions,
                "status": "max_expansions_reached",
            }

        # goal test
        if pos == goal:
            return {
                "found": True,
                "path": path,
                "explored": explored_order,
                "expansions": expansions,
                "status": "found",
            }

        # depth guard
        if depth >= max_depth:
            continue

        neighs = []
        for nb in neighbors(pos, rows, cols, order=order):
            r, c = nb
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[nb] == 1:
                continue
            neighs.append(nb)

        # push in reverse so order is respected; skip if neighbor already in current path (prevents cycles)
        for nb in reversed(neighs):
            if nb in path:
                continue
            stack.append((nb, depth + 1, path + [nb]))

    return {
        "found": False,
        "path": None,
        "explored": explored_order,
        "expansions": expansions,
        "status": "exhausted",
    }
