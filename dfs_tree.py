# dfs_tree.py
import numpy as np
from typing import Tuple, List, Optional

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

def dfs_tree_iterative(grid, start, goal, order='URDL', max_depth=None):
    rows, cols = grid.shape
    if max_depth is None:
        max_depth = rows * cols

    stack = [(start, 0, [start])]
    explored = []
    expansions = 0

    while stack:
        pos, depth, path = stack.pop()
        explored.append(pos)
        expansions += 1

        if pos == goal:
            return {
                "found": True,
                "path": path,
                "explored": explored,
                "expansions": expansions,
                "status": "found"
            }

        if depth >= max_depth:
            continue

        for nb in reversed(list(neighbors(pos, order))):
            r, c = nb
            if not (0 <= r < rows and 0 <= c < cols): continue
            if grid[r][c] == 1: continue
            stack.append((nb, depth + 1, path + [nb]))

    return {
        "found": False,
        "path": None,
        "explored": explored,
        "expansions": expansions,
        "status": "exhausted"
    }
