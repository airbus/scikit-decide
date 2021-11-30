"""Utility code for maze notebook."""

import random
from collections import deque
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_deltas_neighbour = [
    (0, 2),
    (2, 0),
    (0, -2),
    (-2, 0),
]


class Maze:
    _odd_row_pattern = (0, 0)
    _even_row_pattern = (0, 1)
    _row_end_cell = 0
    _empty_cell = 1

    def __init__(self, maze_array: np.array):
        self.maze_array = maze_array

    @property
    def height(self):
        return self.maze_array.shape[0]

    @property
    def width(self):
        return self.maze_array.shape[1]

    @staticmethod
    def generate_random_maze(width: int, height: int) -> "Maze":
        """Generate a maze string with given width and height.

        Width and height are assumed to be odd so that the maze is surrounded by a wall
        and 1 pixel over 2 is a cell followed by a connexion or a wall.

        We use here the "recursive backtracker" algorithm which is a randomized depth-first search algorithm.
        The chosen implementation is actually an iterative one to avoid max recursion stack issues.
        See for instance https://en.wikipedia.org/wiki/Maze_generation_algorithm  for more details.

        """

        # initialization of maze strings: 1 pixel over 2 is an empty cell, the remaining are walls.
        semiwidth = width // 2
        semiheight = height // 2
        odd_row = list(Maze._odd_row_pattern) * semiwidth + [Maze._row_end_cell]
        even_row = list(Maze._even_row_pattern) * semiwidth + [Maze._row_end_cell]
        maze = [list(row) for _ in range(semiheight) for row in (odd_row, even_row)] + [
            list(odd_row)
        ]

        # recursive backtracker algorithm:
        #   1. start from a cell
        #   2. open randomly a wall towards a new cell (unvisited) and go to that cell
        #   3. go on until new cell has no more unvisited neighbours
        #   4. go back to a cell with unvisited neigbours and go back to step 2
        first_cell = (1, 1)
        stack = deque([first_cell])
        visited = {first_cell}
        while len(stack) > 0:
            # pick a cell
            current_cell = stack.pop()
            i, j = current_cell
            # find unvisited neighbours
            neighbours = [(i + di, j + dj) for (di, dj) in _deltas_neighbour]
            neighbours = [
                (i, j)
                for (i, j) in neighbours
                if (i > 0) and (i < height) and (j > 0) and (j < width)
            ]
            unvisited_neighbours = [cell for cell in neighbours if cell not in visited]
            # remove a wall towards an unvisited cell
            if len(unvisited_neighbours) > 0:
                stack.append(current_cell)
                next_cell = random.choice(unvisited_neighbours)
                ii, jj = next_cell
                wall_to_remove = ()
                maze[(i + ii) // 2][(j + jj) // 2] = Maze._empty_cell
                visited.add(next_cell)
                stack.append(next_cell)

        # corresponding maze string
        return Maze(maze_array=np.array(maze))

    def get_image_data(
        self,
        current_position: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> np.array:
        """Return a numpy array to be displayed with `matplotlib.pyplot.imshow()`.

        Args:
            current_position: the current position. Will be highlighted if not None.
            goal: the goal to reach position. Will be highlighted if not None.

        Returns:

        """
        image_data = np.array(self.maze_array, dtype=float)
        if goal is not None:
            image_data[goal] = 0.7
        if current_position is not None:
            image_data[current_position] = 0.3
        return image_data

    def render(
        self,
        current_position: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        ax: Optional[Any] = None,
        image: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Render the maze in a matplotlib figure.

        Args:
            current_position: the current position. Will be highlighted if not None.
            goal: the goal to reach position. Will be highlighted if not None.
            ax: the `matplotlib.axes._subplots.AxesSubplot` in which to render
            image: the `matplotlib.image.AxesImage` to update

        Returns:
            ax, image: the matplotlib subplot in which the maze in rendered
                and the actual rendered matplotlib image

        """
        image_data = self.get_image_data(
            current_position=current_position,
            goal=goal,
        )
        if ax is None:
            plt.ioff()
            fig, ax = plt.subplots(1)
            ax.set_aspect("equal")  # set the x and y axes to the same scale
            plt.xticks([])  # remove the tick marks by setting to an empty list
            plt.yticks([])  # remove the tick marks by setting to an empty list
            ax.invert_yaxis()  # invert the y-axis so the first row of data is at the top
            plt.ion()
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.resizable = False
            fig.set_dpi(1)
            fig.set_figwidth(600)
            fig.set_figheight(600)
        if image is None:
            image = ax.imshow(image_data)
        else:
            image.set_data(image_data)
            image.figure.canvas.draw()
        return ax, image

    def is_an_empty_cell(self, i, j):
        return self.maze_array[i, j] == Maze._empty_cell
