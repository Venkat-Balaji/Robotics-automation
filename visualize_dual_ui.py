# visualize_dual_ui.py
"""
Dual UI: Tree-search (left) and Graph-search (right)
- Shows exploration overlays (magenta boxes)
- Robot moves ONLY on the final path (smooth interpolation)
- Draws a BLACK LINE along the final path
- No yellow dots
- No separate final-path window
"""

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Button
from typing import List, Tuple
import numpy as np
import math

Position = Tuple[int, int]

# -------------------------------------------------------
# Smooth interpolation for robot movement
# -------------------------------------------------------
def interpolate_path(path: List[Position], step: float = 0.15):
    poses = []
    if not path:
        return poses

    cy, cx = float(path[0][0]), float(path[0][1])
    poses.append((cy, cx))

    for cell in path[1:]:
        ty, tx = float(cell[0]), float(cell[1])
        dist = math.hypot(ty - cy, tx - cx)
        if dist == 0:
            poses.append((ty, tx))
            cy, cx = ty, tx
            continue

        n = max(1, int(dist / step))
        for i in range(1, n + 1):
            t = i / n
            ny = cy + (ty - cy) * t
            nx = cx + (tx - cx) * t
            poses.append((ny, nx))

        cy, cx = ty, tx

    return poses


# -------------------------------------------------------
# Dual Animator Class
# -------------------------------------------------------
class DualGridAnimator:
    def __init__(
        self,
        grid: np.ndarray,
        start: Position,
        goal: Position,
        tree_explored: List[Position],
        tree_path: List[Position],
        graph_explored: List[Position],
        graph_path: List[Position],
        delay: float = 0.18,
    ):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.delay = delay

        # Exploration overlays only
        self.tree_explored = tree_explored or []
        self.graph_explored = graph_explored or []

        # Final robot paths
        self.tree_path = tree_path or []
        self.graph_path = graph_path or []

        # Interpolated robot motion
        self.tree_motion = interpolate_path(self.tree_path)
        self.graph_motion = interpolate_path(self.graph_path)

        # Playback indices
        self.tree_idx = 0
        self.graph_idx = 0
        self.tree_playing = False
        self.graph_playing = False

        # UI setup
        self.fig = plt.figure(figsize=(12, 6))
        self.ax_tree = self.fig.add_subplot(1, 2, 1)
        self.ax_graph = self.fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(bottom=0.18)

        # draw everything
        self._draw_base()
        self._draw_explored()
        self._draw_path_lines()  # <-- ADDED: draws black final robot paths

        # robot markers
        self.tree_robot = patches.Circle((start[1], start[0]), 0.32, color="blue", zorder=12)
        self.graph_robot = patches.Circle((start[1], start[0]), 0.32, color="blue", zorder=12)

        self.ax_tree.add_patch(self.tree_robot)
        self.ax_graph.add_patch(self.graph_robot)

        # Status text
        self.tree_text = self.ax_tree.text(0.02, 0.98, "", transform=self.ax_tree.transAxes, va="top")
        self.graph_text = self.ax_graph.text(0.02, 0.98, "", transform=self.ax_graph.transAxes, va="top")

        self._update_texts()

        # UI buttons
        ax_t = plt.axes([0.18, 0.06, 0.14, 0.07])
        ax_g = plt.axes([0.68, 0.06, 0.14, 0.07])
        ax_r = plt.axes([0.43, 0.03, 0.14, 0.07])

        self.btn_t_play = Button(ax_t, "Tree Play")
        self.btn_g_play = Button(ax_g, "Graph Play")
        self.btn_reset = Button(ax_r, "Reset Both")

        self.btn_t_play.on_clicked(self._toggle_tree)
        self.btn_g_play.on_clicked(self._toggle_graph)
        self.btn_reset.on_clicked(self._reset_both)

        # Start animator loop
        self._run_loop()

    # ---------------------------------------------------
    # Draw base map + start/goal
    # ---------------------------------------------------
    def _draw_base(self):
        rows, cols = self.grid.shape
        img = np.ones((rows, cols, 3))
        img[self.grid == 1] = (0.7, 0.7, 0.7)

        for ax, title in [(self.ax_tree, "Tree Search"), (self.ax_graph, "Graph Search")]:
            ax.imshow(img, origin="upper", interpolation="none")
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(rows - 0.5, -0.5)
            ax.grid(color="lightgray", linewidth=0.6)
            ax.set_title(title)

            ax.scatter([self.start[1]], [self.start[0]], c="red", s=100, marker="s", edgecolors="k", zorder=9)
            ax.scatter([self.goal[1]], [self.goal[0]], c="green", s=100, marker="s", edgecolors="k", zorder=9)

    # ---------------------------------------------------
    # Explored overlays (visual only)
    # ---------------------------------------------------
    def _draw_explored(self):
        if self.tree_explored:
            ys = [p[0] for p in self.tree_explored]
            xs = [p[1] for p in self.tree_explored]
            self.ax_tree.scatter(xs, ys, s=60, facecolors="none", edgecolors="magenta", linewidth=1.3)

        if self.graph_explored:
            ys = [p[0] for p in self.graph_explored]
            xs = [p[1] for p in self.graph_explored]
            self.ax_graph.scatter(xs, ys, s=60, facecolors="none", edgecolors="magenta", linewidth=1.3)

    # ---------------------------------------------------
    # Draw FINAL PATH as black line (requested)
    # ---------------------------------------------------
    def _draw_path_lines(self):
        if self.tree_path:
            pr = [p[0] for p in self.tree_path]
            pc = [p[1] for p in self.tree_path]
            self.ax_tree.plot(pc, pr, "-k", linewidth=3, zorder=8)

        if self.graph_path:
            pr = [p[0] for p in self.graph_path]
            pc = [p[1] for p in self.graph_path]
            self.ax_graph.plot(pc, pr, "-k", linewidth=3, zorder=8)

    # ---------------------------------------------------
    # Controls
    # ---------------------------------------------------
    def _toggle_tree(self, _):
        if not self.tree_motion:
            return
        self.tree_playing = not self.tree_playing
        self.btn_t_play.label.set_text("Pause" if self.tree_playing else "Tree Play")

    def _toggle_graph(self, _):
        if not self.graph_motion:
            return
        self.graph_playing = not self.graph_playing
        self.btn_g_play.label.set_text("Pause" if self.graph_playing else "Graph Play")

    def _reset_both(self, _=None):
        self.tree_playing = False
        self.graph_playing = False
        self.tree_idx = 0
        self.graph_idx = 0

        self.tree_robot.center = (self.start[1], self.start[0])
        self.graph_robot.center = (self.start[1], self.start[0])

        self.btn_t_play.label.set_text("Tree Play")
        self.btn_g_play.label.set_text("Graph Play")

        self._update_texts()
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------
    # Updates
    # ---------------------------------------------------
    def _update_texts(self):
        tpos = self.tree_motion[self.tree_idx] if self.tree_motion else self.start
        gpos = self.graph_motion[self.graph_idx] if self.graph_motion else self.start

        self.tree_text.set_text(f"Tree robot: {tuple(round(v,2) for v in tpos)}")
        self.graph_text.set_text(f"Graph robot: {tuple(round(v,2) for v in gpos)}")

    # ---------------------------------------------------
    # Main playback loop
    # ---------------------------------------------------
    def _run_loop(self):
        try:
            while plt.fignum_exists(self.fig.number):

                moved = False

                # TREE motion
                if self.tree_playing and self.tree_idx < len(self.tree_motion) - 1:
                    self.tree_idx += 1
                    py, px = self.tree_motion[self.tree_idx]
                    self.tree_robot.center = (px, py)
                    moved = True

                # GRAPH motion
                if self.graph_playing and self.graph_idx < len(self.graph_motion) - 1:
                    self.graph_idx += 1
                    py, px = self.graph_motion[self.graph_idx]
                    self.graph_robot.center = (px, py)
                    moved = True

                if moved:
                    self._update_texts()
                    self.fig.canvas.draw_idle()

                plt.pause(self.delay)

        except Exception as e:
            print("Animator stopped:", e)
