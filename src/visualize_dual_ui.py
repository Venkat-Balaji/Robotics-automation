# visualize_dual_ui.py
"""Dual UI: Tree-search (left) and Graph-search (right) with exploration overlays and smooth path animation."""

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Button
from typing import List, Tuple
import numpy as np
import math

Position = Tuple[int, int]

def interpolate_path(path: List[Position], step: float = 0.15) -> List[Position]:
    """Interpolate path with smooth motion."""
    poses = []
    if not path:
        return poses
    cy, cx = float(path[0][0]), float(path[0][1])
    poses.append((cy, cx))
    for cell in path[1:]:
        ty, tx = float(cell[0]), float(cell[1])
        dist = math.hypot(ty - cy, tx - cx)
        if dist > 0:
            n = max(1, int(dist / step))
            poses.extend([(cy + (ty - cy) * i / n, cx + (tx - cx) * i / n) for i in range(1, n + 1)])
        cy, cx = ty, tx
    return poses

class DualGridAnimator:
    def __init__(self, grid: np.ndarray, start: Position, goal: Position, tree_explored: List[Position],
                 tree_path: List[Position], graph_explored: List[Position], graph_path: List[Position], delay: float = 0.18):
        self.grid, self.start, self.goal, self.delay = grid, start, goal, delay
        self.tree_explored, self.graph_explored = tree_explored or [], graph_explored or []
        self.tree_path, self.graph_path = tree_path or [], graph_path or []
        self.tree_motion, self.graph_motion = interpolate_path(self.tree_path), interpolate_path(self.graph_path)
        self.tree_idx = self.graph_idx = 0
        self.tree_playing = self.graph_playing = False

        self.fig = plt.figure(figsize=(12, 6))
        self.ax_tree, self.ax_graph = self.fig.add_subplot(1, 2, 1), self.fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(bottom=0.18)

        self._draw_base()
        self._draw_explored()
        self._draw_path_lines()
        self.tree_robot = patches.Circle((start[1], start[0]), 0.32, color="blue", zorder=12)
        self.graph_robot = patches.Circle((start[1], start[0]), 0.32, color="blue", zorder=12)
        self.ax_tree.add_patch(self.tree_robot)
        self.ax_graph.add_patch(self.graph_robot)

        self.tree_text = self.ax_tree.text(0.02, 0.98, "", transform=self.ax_tree.transAxes, va="top")
        self.graph_text = self.ax_graph.text(0.02, 0.98, "", transform=self.ax_graph.transAxes, va="top")
        self._update_texts()

        ax_t, ax_g, ax_r = plt.axes([0.18, 0.06, 0.14, 0.07]), plt.axes([0.68, 0.06, 0.14, 0.07]), plt.axes([0.43, 0.03, 0.14, 0.07])
        self.btn_t_play, self.btn_g_play, self.btn_reset = Button(ax_t, "Tree Play"), Button(ax_g, "Graph Play"), Button(ax_r, "Reset Both")
        self.btn_t_play.on_clicked(self._toggle_tree)
        self.btn_g_play.on_clicked(self._toggle_graph)
        self.btn_reset.on_clicked(self._reset_both)
        self._run_loop()

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

    def _draw_explored(self):
        for explored, ax in [(self.tree_explored, self.ax_tree), (self.graph_explored, self.ax_graph)]:
            if explored:
                ys, xs = zip(*explored)
                ax.scatter(xs, ys, s=60, facecolors="none", edgecolors="magenta", linewidth=1.3)

    def _draw_path_lines(self):
        for path, ax in [(self.tree_path, self.ax_tree), (self.graph_path, self.ax_graph)]:
            if path:
                pr, pc = zip(*path)
                ax.plot(pc, pr, "-k", linewidth=3, zorder=8)

    def _toggle_tree(self, _):
        if self.tree_motion:
            self.tree_playing = not self.tree_playing
            self.btn_t_play.label.set_text("Pause" if self.tree_playing else "Tree Play")

    def _toggle_graph(self, _):
        if self.graph_motion:
            self.graph_playing = not self.graph_playing
            self.btn_g_play.label.set_text("Pause" if self.graph_playing else "Graph Play")

    def _reset_both(self, _=None):
        self.tree_playing = self.graph_playing = False
        self.tree_idx = self.graph_idx = 0
        self.tree_robot.center = self.graph_robot.center = (self.start[1], self.start[0])
        self.btn_t_play.label.set_text("Tree Play")
        self.btn_g_play.label.set_text("Graph Play")
        self._update_texts()
        self.fig.canvas.draw_idle()

    def _update_texts(self):
        tpos = self.tree_motion[self.tree_idx] if self.tree_motion else self.start
        gpos = self.graph_motion[self.graph_idx] if self.graph_motion else self.start


    def _run_loop(self):
        try:
            while plt.fignum_exists(self.fig.number):
                moved = False
                if self.tree_playing and self.tree_idx < len(self.tree_motion) - 1:
                    self.tree_idx += 1
                    py, px = self.tree_motion[self.tree_idx]
                    self.tree_robot.center = (px, py)
                    moved = True
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
