# dfs_graph.py
import numpy as np
import time, tracemalloc
from typing import Dict, Tuple, List

Position = Tuple[int, int]

def neighbors(pos: Position, order='URDL'):
    r, c = pos
    mapping = {
        'U': (r - 1, c),
        'R': (r, c + 1),
        'D': (r + 1, c),
        'L': (r, c - 1)
    }
    for ch in order:
        yield mapping[ch]

def reconstruct_path(parent, start, goal):
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    return list(reversed(path))

def dfs_graph_iterative(grid, start, goal, order='URDL'):
    rows, cols = grid.shape
    stack = [start]
    visited = set([start])
    parent = {}
    explored = []
    expansions = 0

    while stack:
        node = stack.pop()
        explored.append(node)
        expansions += 1

        if node == goal:
            return {
                "found": True,
                "path": reconstruct_path(parent, start, goal),
                "explored": explored,
                "visited": visited,
                "expansions": expansions,
                "status": "found"
            }

        for nb in neighbors(node, order):
            r, c = nb
            if not (0 <= r < rows and 0 <= c < cols): continue
            if grid[r][c] == 1: continue
            if nb in visited: continue

            visited.add(nb)
            parent[nb] = node
            stack.append(nb)

    return {
        "found": False,
        "path": None,
        "explored": explored,
        "visited": visited,
        "expansions": expansions,
        "status": "exhausted"
    }

def measure(func, *args):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "result": result,
        "time_s": t1 - t0,
        "peak_memory_kb": peak / 1024
    }
