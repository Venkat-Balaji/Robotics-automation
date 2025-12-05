"""
robot_sim.py
Continuous robot movement simulator (smooth animation)
"""

import numpy as np
import math
from typing import List, Tuple

Position = Tuple[int, int]


class Robot:
    def __init__(self, start: Position, grid: np.ndarray, radius=0, interp_step=0.2):
        self.grid = grid
        self.pose = (float(start[0]), float(start[1]))
        self.cell = start
        self.step = interp_step
        self.radius = radius

    def is_collision(self, cell):
        r, c = cell
        rows, cols = self.grid.shape
        if not (0 <= r < rows and 0 <= c < cols):
            return True
        return self.grid[cell] == 1

    def follow_path(self, path: List[Position]):
        for target in path:
            if self.is_collision(target):
                raise RuntimeError("Collision detected.")

            sy, sx = self.pose
            ty, tx = float(target[0]), float(target[1])
            dist = math.hypot(ty - sy, tx - sx)
            steps = max(1, int(dist / self.step))

            for i in range(steps):
                t = (i + 1) / steps
                ny = sy + (ty - sy) * t
                nx = sx + (tx - sx) * t
                self.pose = (ny, nx)
                yield self.pose

            self.cell = target
            self.pose = (ty, tx)
            yield self.pose
