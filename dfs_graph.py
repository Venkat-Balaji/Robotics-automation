# dfs_graph.py
"""
Graph-search DFS (iterative) with measurement helper.
"""

import time
import tracemalloc
from typing import Tuple, Dict, List
import numpy as np

Position = Tuple[int, int]


def neighbors(pos: Position, rows: int, cols: int, order: str = "URDL"):
    r, c = pos
    mapping = {"U": (r - 1, c), "R": (r, c + 1), "D": (r + 1, c), "L": (r, c - 1)}
    for d in order:
        yield mapping[d]


def reconstruct_path(parent: Dict[Position, Position], start: Position, goal: Position):
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


def dfs_graph_iterative(grid: np.ndarray, start: Position, goal: Position, order: str = "URDL"):
    rows, cols = grid.shape
    stack = [start]
    visited = set([start])
    parent: Dict[Position, Position] = {}
    explored_order: List[Position] = []
    expansions = 0

    while stack:
        node = stack.pop()
        explored_order.append(node)
        expansions += 1

        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return {
                "found": True,
                "path": path,
                "explored": explored_order,
                "visited": visited,
                "expansions": expansions,
                "status": "found",
            }

        for nb in neighbors(node, rows, cols, order=order):
            r, c = nb
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[nb] == 1:
                continue
            if nb in visited:
                continue

            visited.add(nb)
            parent[nb] = node
            stack.append(nb)

    return {
        "found": False,
        "path": None,
        "explored": explored_order,
        "visited": visited,
        "expansions": expansions,
        "status": "exhausted",
    }


def measure(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"result": res, "time_s": t1 - t0, "peak_memory_kb": peak / 1024.0}
