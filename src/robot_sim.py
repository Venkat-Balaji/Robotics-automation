# robot_sim.py
"""Robot simulator: smooth interpolation between discrete cells. NO saving of MP4 or PNG."""

import math
from typing import List, Tuple, Generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Position = Tuple[int, int]

class Robot:
    def __init__(self, start_cell: Position, grid: np.ndarray, radius_cells: int = 0, interp_step: float = 0.15):
        self.cell = start_cell
        self.pose = (float(start_cell[0]), float(start_cell[1]))
        self.grid = grid
        self.radius = radius_cells
        self.step = interp_step

    def is_collision_cell(self, cell: Position) -> bool:
        r, c = cell
        rows, cols = self.grid.shape
        return not (0 <= r < rows and 0 <= c < cols) or self.grid[r, c] == 1

    def follow_path_steps(self, path: List[Position]) -> Generator:
        """Generator: yields continuous poses (row, col) as robot moves along path. Raises RuntimeError on collision."""
        for cell in path:
            if self.is_collision_cell(cell):
                raise RuntimeError(f"Collision at target cell {cell} before entering.")
            sy, sx = self.pose
            ty, tx = float(cell[0]), float(cell[1])
            dist = math.hypot(ty - sy, tx - sx)
            if dist == 0:
                self.pose = (ty, tx)
                self.cell = cell
                yield self.pose
                continue
            steps = max(1, int(dist / self.step))
            for s in range(1, steps + 1):
                t = s / steps
                ny, nx = sy + (ty - sy) * t, sx + (tx - sx) * t
                self.pose = (ny, nx)
                if self.is_collision_cell((int(round(ny)), int(round(nx)))):
                    raise RuntimeError(f"Collision while moving near {int(round(ny)), int(round(nx))}")
                yield self.pose
            self.cell, self.pose = cell, (ty, tx)
            yield self.pose

def animate_path_live(grid: np.ndarray, path: List[Position], start: Position, goal: Position):
    """Only displays animation LIVE. NO saving to MP4 or PNG."""
    fig, ax = plt.subplots(figsize=(6, 6))
    img = np.ones((grid.shape[0], grid.shape[1], 3), dtype=float)
    img[grid == 1] = (0.7, 0.7, 0.7)
    ax.imshow(img, origin="upper", interpolation="none")
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)

    if path:
        pr, pc = zip(*path)
        ax.plot(pc, pr, "-k", linewidth=3)
        ax.scatter(pc, pr, s=50, facecolors="yellow", edgecolors="k")

    ax.scatter([start[1]], [start[0]], c="red", s=140, marker="s")
    ax.scatter([goal[1]], [goal[0]], c="green", s=140, marker="s")

    robot = Robot(start, grid)
    poses = [robot.pose]
    try:
        for p in robot.follow_path_steps(path):
            poses.append(p)
    except RuntimeError as e:
        print("Simulation aborted:", e)

    robot_patch = plt.Circle((start[1], start[0]), 0.3, color="blue", zorder=10)
    ax.add_patch(robot_patch)
    animation.FuncAnimation(fig, lambda i: (robot_patch.set_center((poses[i][1], poses[i][0])), (robot_patch,))[1],
                          frames=len(poses), interval=40, blit=False, repeat=False)
    plt.show()
