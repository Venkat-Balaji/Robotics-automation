# visualize_ui.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import patches
from typing import List, Tuple

Position = Tuple[int, int]

class GridAnimator:
    def __init__(self, grid, start, goal, path, explored):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.path = path or []
        self.explored = explored

        self.rows, self.cols = grid.shape
        self.index = 0
        self.playing = False
        self.delay = 0.35

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)

        self._draw_base()
        self._draw_path()

        if self.path:
            y, x = self.path[0]
        else:
            y, x = self.start

        self.robot = patches.Circle((x, y), 0.3, color='blue', zorder=10)
        self.ax.add_patch(self.robot)

        self._add_buttons()
        self._idle_loop()

    def _draw_base(self):
        img = np.ones((self.rows, self.cols, 3))
        img[self.grid == 1] = (0.7, 0.7, 0.7)
        self.ax.imshow(img, origin="upper")
        self.ax.set_xticks(range(self.cols))
        self.ax.set_yticks(range(self.rows))
        self.ax.grid(True)
        self.ax.scatter(self.start[1], self.start[0], c='red', s=120, marker='s')
        self.ax.scatter(self.goal[1], self.goal[0], c='green', s=120, marker='s')

    def _draw_path(self):
        if not self.path: return
        pr = [p[0] for p in self.path]
        pc = [p[1] for p in self.path]
        self.ax.plot(pc, pr, '-k')
        self.ax.scatter(pc, pr, c='yellow')

    def _add_buttons(self):
        ax_play = plt.axes([0.2, 0.05, 0.1, 0.06])
        ax_step = plt.axes([0.32, 0.05, 0.1, 0.06])
        ax_back = plt.axes([0.44, 0.05, 0.1, 0.06])
        ax_reset = plt.axes([0.56, 0.05, 0.1, 0.06])

        self.btn_play = Button(ax_play, 'Play')
        self.btn_step = Button(ax_step, 'Step >')
        self.btn_back = Button(ax_back, '< Step')
        self.btn_reset = Button(ax_reset, 'Reset')

        self.btn_play.on_clicked(self._toggle_play)
        self.btn_step.on_clicked(self._step_forward)
        self.btn_back.on_clicked(self._step_back)
        self.btn_reset.on_clicked(self._reset)

    def _toggle_play(self, _):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")

    def _step_forward(self, _):
        if self.index < len(self.path) - 1:
            self.index += 1
            self._update_robot()

    def _step_back(self, _):
        if self.index > 0:
            self.index -= 1
            self._update_robot()

    def _reset(self, _):
        self.index = 0
        self.playing = False
        self._update_robot()

    def _update_robot(self):
        y, x = self.path[self.index]
        self.robot.center = (x, y)
        self.fig.canvas.draw_idle()

    def _idle_loop(self):
        while plt.fignum_exists(self.fig.number):
            if self.playing and self.index < len(self.path) - 1:
                self.index += 1
                self._update_robot()
            plt.pause(self.delay)
