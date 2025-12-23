# dfs_graph.py
"""Graph-search DFS (iterative) with measurement helper."""

import time, tracemalloc
from typing import Tuple, Dict, List, Generator
import numpy as np

Position = Tuple[int, int]

def neighbors(pos: Position, rows: int, cols: int, order: str = "URDL") -> Generator:
    r, c = pos
    mapping = {"U": (r - 1, c), "R": (r, c + 1), "D": (r + 1, c), "L": (r, c - 1)}
    for d in order:
        nr, nc = mapping[d]
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def reconstruct_path(parent: Dict[Position, Position], start: Position, goal: Position) -> List[Position]:
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    return path[::-1]

def dfs_graph_iterative(grid: np.ndarray, start: Position, goal: Position, order: str = "URDL") -> Dict:
    rows, cols = grid.shape
    stack, visited, parent, explored_order, expansions = [start], {start}, {}, [], 0

    while stack:
        node = stack.pop()
        explored_order.append(node)
        expansions += 1
        
        if node == goal:
            return {"found": True, "path": reconstruct_path(parent, start, goal), "explored": explored_order,
                    "visited": visited, "expansions": expansions, "status": "found"}

        for nb in neighbors(node, rows, cols, order=order):
            if grid[nb] == 0 and nb not in visited:
                visited.add(nb)
                parent[nb] = node
                stack.append(nb)

    return {"found": False, "path": None, "explored": explored_order, "visited": visited, 
            "expansions": expansions, "status": "exhausted"}

def measure(func, *args, **kwargs) -> Dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"result": res, "time_s": t1 - t0, "peak_memory_kb": peak / 1024.0}
