# robot_sim.py
"""
Robot simulator: smooth interpolation between discrete cells and optional animation saving.
"""

import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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
        if not (0 <= r < rows and 0 <= c < cols):
            return True
        return self.grid[r, c] == 1

    def follow_path_steps(self, path: List[Position]):
        """
        Generator: yields continuous poses (row, col) as robot moves along path.
        Raises RuntimeError on collision.
        """
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
                ny = sy + (ty - sy) * t
                nx = sx + (tx - sx) * t
                self.pose = (ny, nx)
                nearest_cell = (int(round(ny)), int(round(nx)))
                if self.is_collision_cell(nearest_cell):
                    raise RuntimeError(f"Collision while moving near {nearest_cell}")
                yield self.pose
            # finalize
            self.cell = cell
            self.pose = (ty, tx)
            yield self.pose


def animate_and_save(grid: np.ndarray, path: List[Position], start: Position, goal: Position, out_dir: str, filename_mp4: str = "robot_follow.mp4", save_mp4: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    img = np.ones((grid.shape[0], grid.shape[1], 3), dtype=float)
    img[grid == 1] = (0.7, 0.7, 0.7)
    ax.imshow(img, origin="upper", interpolation="none")
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    if path:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
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

    patches = []

    def update(i):
        # Remove previous robot patches we added
        for p in patches[:]:
            try:
                p.remove()
            except Exception:
                pass
            patches.remove(p)
        pose = poses[i]
        circ = plt.Circle((pose[1], pose[0]), 0.3 + 0.12 * robot.radius, color="blue", alpha=0.9, zorder=10)
        ax.add_patch(circ)
        patches.append(circ)
        return patches

    ani = animation.FuncAnimation(fig, update, frames=len(poses), interval=40, blit=False, repeat=False)
    outpath = os.path.join(out_dir, filename_mp4)
    if save_mp4:
        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=25, metadata=dict(artist="dfs_tool"), bitrate=1800)
            ani.save(outpath, writer=writer)
            print(f"Saved animation to {outpath}")
        except Exception as e:
            # fallback to saving frames
            print("Could not save MP4 (ffmpeg missing?). Saving frames to folder.")
            for i, pose in enumerate(poses):
                for p in patches[:]:
                    try:
                        p.remove()
                    except:
                        pass
                patches.clear()
                circ = plt.Circle((pose[1], pose[0]), 0.3 + 0.12 * robot.radius, color="blue", alpha=0.9)
                ax.add_patch(circ)
                fig.savefig(os.path.join(out_dir, f"frame_{i:04d}.png"))
            print(f"Saved {len(poses)} frames to {out_dir}")
    else:
        for i, pose in enumerate(poses):
            for p in patches[:]:
                try:
                    p.remove()
                except:
                    pass
            patches.clear()
            circ = plt.Circle((pose[1], pose[0]), 0.3 + 0.12 * robot.radius, color="blue", alpha=0.9)
            ax.add_patch(circ)
            fig.savefig(os.path.join(out_dir, f"frame_{i:04d}.png"))
        print(f"Saved {len(poses)} frames to {out_dir}")
    plt.close(fig)
