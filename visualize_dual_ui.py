# visualize_dual_ui.py
"""
Dual UI: shows Tree-search traversal (left) and Graph-search traversal (right)
in a single figure with independent controls for each panel.

Usage:
    from visualize_dual_ui import DualGridAnimator
    animator = DualGridAnimator(grid, start, goal,
                               tree_explored, tree_path,
                               graph_explored, graph_path)
"""

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Button, CheckButtons
from typing import List, Tuple
import numpy as np


Position = Tuple[int, int]


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
        delay: float = 0.25,
    ):
        self.grid = grid
        self.start = start
        self.goal = goal

        # traversal sequences
        self.tree_explored = tree_explored or []
        self.tree_path = tree_path or []
        self.graph_explored = graph_explored or []
        self.graph_path = graph_path or []

        # state indices
        self.tree_idx = 0
        self.graph_idx = 0

        # play flags
        self.tree_playing = False
        self.graph_playing = False

        # show final path flags
        self.show_paths = True

        self.delay = delay

        # create figure and two axes
        self.fig = plt.figure(figsize=(12, 6))
        self.ax_tree = self.fig.add_subplot(1, 2, 1)
        self.ax_graph = self.fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(bottom=0.22)

        # draw bases
        self._draw_base(self.ax_tree, title="Tree-search DFS Traversal")
        self._draw_base(self.ax_graph, title="Graph-search DFS Traversal")

        # explored markers (empty now) and path overlays
        self._draw_explored_overlay(self.ax_tree, self.tree_explored)
        self._draw_explored_overlay(self.ax_graph, self.graph_explored)
        if self.show_paths:
            self._draw_path_overlay(self.ax_tree, self.tree_path)
            self._draw_path_overlay(self.ax_graph, self.graph_path)

        # robot patches
        t_y, t_x = (self.tree_explored[0] if self.tree_explored else self.start)
        g_y, g_x = (self.graph_explored[0] if self.graph_explored else self.start)
        self.tree_robot = patches.Circle((t_x, t_y), 0.3, color="blue", zorder=12)
        self.graph_robot = patches.Circle((g_x, g_y), 0.3, color="blue", zorder=12)
        self.ax_tree.add_patch(self.tree_robot)
        self.ax_graph.add_patch(self.graph_robot)

        # status texts
        self.tree_text = self.ax_tree.text(0.02, 0.98, "", transform=self.ax_tree.transAxes, va="top")
        self.graph_text = self.ax_graph.text(0.02, 0.98, "", transform=self.ax_graph.transAxes, va="top")
        self._update_texts()

        # controls: layout positions
        ax_t_play = plt.axes([0.10, 0.08, 0.10, 0.06])
        ax_t_stepf = plt.axes([0.21, 0.08, 0.10, 0.06])
        ax_t_stepb = plt.axes([0.32, 0.08, 0.10, 0.06])

        ax_g_play = plt.axes([0.55, 0.08, 0.10, 0.06])
        ax_g_stepf = plt.axes([0.66, 0.08, 0.10, 0.06])
        ax_g_stepb = plt.axes([0.77, 0.08, 0.10, 0.06])

        ax_show_path = plt.axes([0.43, 0.02, 0.12, 0.06])
        ax_reset = plt.axes([0.57, 0.02, 0.12, 0.06])

        self.btn_t_play = Button(ax_t_play, "Tree Play")
        self.btn_t_stepf = Button(ax_t_stepf, "T Step >")
        self.btn_t_stepb = Button(ax_t_stepb, "< T Step")

        self.btn_g_play = Button(ax_g_play, "Graph Play")
        self.btn_g_stepf = Button(ax_g_stepf, "G Step >")
        self.btn_g_stepb = Button(ax_g_stepb, "< G Step")

        self.btn_show_path = Button(ax_show_path, "Hide Paths" if self.show_paths else "Show Paths")
        self.btn_reset = Button(ax_reset, "Reset Both")

        # connect callbacks
        self.btn_t_play.on_clicked(self._toggle_tree_play)
        self.btn_t_stepf.on_clicked(self._tree_step_forward)
        self.btn_t_stepb.on_clicked(self._tree_step_back)

        self.btn_g_play.on_clicked(self._toggle_graph_play)
        self.btn_g_stepf.on_clicked(self._graph_step_forward)
        self.btn_g_stepb.on_clicked(self._graph_step_back)

        self.btn_show_path.on_clicked(self._toggle_show_paths)
        self.btn_reset.on_clicked(self._reset_both)

        # run combined event loop
        self._run_loop()

    def _draw_base(self, ax, title=""):
        rows, cols = self.grid.shape
        img = np.ones((rows, cols, 3), dtype=float)
        img[self.grid == 1] = (0.7, 0.7, 0.7)
        ax.imshow(img, origin="upper", interpolation="none")
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.grid(which="both", color="lightgray", linewidth=0.6)
        ax.set_title(title)

        # start/goal markers
        ax.scatter([self.start[1]], [self.start[0]], c="red", s=120, marker="s", edgecolors="k", zorder=8)
        ax.scatter([self.goal[1]], [self.goal[0]], c="green", s=120, marker="s", edgecolors="k", zorder=8)

    def _draw_explored_overlay(self, ax, explored):
        if not explored:
            return
        ex_r = [p[0] for p in explored]
        ex_c = [p[1] for p in explored]
        ax.scatter(ex_c, ex_r, marker="s", s=80, facecolors="none", edgecolors="magenta", linewidths=1.0, zorder=4)

    def _draw_path_overlay(self, ax, path):
        if not path:
            return
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, "-k", linewidth=3, zorder=6)
        ax.scatter(pc, pr, s=60, facecolors="yellow", edgecolors="k", zorder=7)

    def _toggle_tree_play(self, _):
        self.tree_playing = not self.tree_playing
        self.btn_t_play.label.set_text("Tree Pause" if self.tree_playing else "Tree Play")

    def _toggle_graph_play(self, _):
        self.graph_playing = not self.graph_playing
        self.btn_g_play.label.set_text("Graph Pause" if self.graph_playing else "Graph Play")

    def _tree_step_forward(self, _=None):
        if self.tree_idx < max(0, len(self.tree_explored) - 1):
            self.tree_idx += 1
            self._update_tree_robot()

    def _tree_step_back(self, _=None):
        if self.tree_idx > 0:
            self.tree_idx -= 1
            self._update_tree_robot()

    def _graph_step_forward(self, _=None):
        if self.graph_idx < max(0, len(self.graph_explored) - 1):
            self.graph_idx += 1
            self._update_graph_robot()

    def _graph_step_back(self, _=None):
        if self.graph_idx > 0:
            self.graph_idx -= 1
            self._update_graph_robot()

    def _toggle_show_paths(self, _):
        self.show_paths = not self.show_paths
        # redraw overlays
        self.ax_tree.cla()
        self.ax_graph.cla()
        self._draw_base(self.ax_tree, title="Tree-search DFS Traversal")
        self._draw_base(self.ax_graph, title="Graph-search DFS Traversal")
        self._draw_explored_overlay(self.ax_tree, self.tree_explored)
        self._draw_explored_overlay(self.ax_graph, self.graph_explored)
        if self.show_paths:
            self._draw_path_overlay(self.ax_tree, self.tree_path)
            self._draw_path_overlay(self.ax_graph, self.graph_path)
            self.btn_show_path.label.set_text("Hide Paths")
        else:
            self.btn_show_path.label.set_text("Show Paths")
        # re-add robot patches & texts
        self.ax_tree.add_patch(self.tree_robot)
        self.ax_graph.add_patch(self.graph_robot)
        self.ax_tree.add_artist(self.tree_text)
        self.ax_graph.add_artist(self.graph_text)
        self._update_texts()
        self._update_tree_robot()
        self._update_graph_robot()
        self.fig.canvas.draw_idle()

    def _reset_both(self, _=None):
        self.tree_idx = 0
        self.graph_idx = 0
        self.tree_playing = False
        self.graph_playing = False
        self.btn_t_play.label.set_text("Tree Play")
        self.btn_g_play.label.set_text("Graph Play")
        self._update_tree_robot()
        self._update_graph_robot()

    def _update_tree_robot(self):
        if not self.tree_explored:
            pos = self.start
        else:
            pos = self.tree_explored[self.tree_idx]
        y, x = pos
        self.tree_robot.center = (x, y)
        self._update_texts()
        self.fig.canvas.draw_idle()

    def _update_graph_robot(self):
        if not self.graph_explored:
            pos = self.start
        else:
            pos = self.graph_explored[self.graph_idx]
        y, x = pos
        self.graph_robot.center = (x, y)
        self._update_texts()
        self.fig.canvas.draw_idle()

    def _update_texts(self):
        t_total = len(self.tree_explored) - 1 if self.tree_explored else 0
        g_total = len(self.graph_explored) - 1 if self.graph_explored else 0
        t_cell = self.tree_explored[self.tree_idx] if self.tree_explored else self.start
        g_cell = self.graph_explored[self.graph_idx] if self.graph_explored else self.start
        self.tree_text.set_text(f"Tree: idx {self.tree_idx} / {t_total}\ncell {t_cell}")
        self.graph_text.set_text(f"Graph: idx {self.graph_idx} / {g_total}\ncell {g_cell}")

    def _run_loop(self):
        # main combined loop that updates both panels when playing
        try:
            while plt.fignum_exists(self.fig.number):
                advanced = False
                if self.tree_playing and self.tree_explored:
                    if self.tree_idx < len(self.tree_explored) - 1:
                        self.tree_idx += 1
                        self._update_tree_robot()
                        advanced = True
                    else:
                        self.tree_playing = False
                        self.btn_t_play.label.set_text("Tree Play")
                if self.graph_playing and self.graph_explored:
                    if self.graph_idx < len(self.graph_explored) - 1:
                        self.graph_idx += 1
                        self._update_graph_robot()
                        advanced = True
                    else:
                        self.graph_playing = False
                        self.btn_g_play.label.set_text("Graph Play")
                # small pause to keep UI responsive
                plt.pause(self.delay)
                # avoid busy spin when nothing playing
                if not advanced:
                    plt.pause(0.01)
        except Exception as e:
            print("Dual animator stopped:", e)
