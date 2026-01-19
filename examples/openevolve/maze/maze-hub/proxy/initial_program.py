import random
from enum import Enum
from typing import NamedTuple


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


def get_next_state_and_cost(
    state: State, action: Action, maze: list[list[int]]
) -> tuple[State, int]:
    """Compute next state and corresponding cost from current state and chosen action.

    Can be used to apply the planner heuristic to a given maze.

    Args:
        state: current position in the maze
        action: action to apply
        maze: the maze representation as an array. 0's are walls and 1's are valid positions.

    Returns:
        next_state, cost: next position in the maze and cost of the action

    """
    # move in the maze according to action
    if action == Action.left:
        next_state = State(state.x - 1, state.y)
    elif action == Action.right:
        next_state = State(state.x + 1, state.y)
    elif action == Action.up:
        next_state = State(state.x, state.y - 1)
    elif action == Action.down:
        next_state = State(state.x, state.y + 1)
    else:
        raise ValueError("action not valid.")

    # check if valid move and compute cost accordingly
    num_cols = len(maze[0])
    num_rows = len(maze)
    if (
        0 <= next_state.x < num_cols  # not too high or low
        and 0 <= next_state.y < num_rows  # not too on the left or right
        and maze[next_state.y][next_state.x] == 1  # not in a wall
    ):
        cost = 1
    else:
        # hitting a wall: roll back the move and add a penalty cost
        next_state = state
        cost = 2

    return next_state, cost


# EVOLVE-BLOCK-START
class Planner:
    """Random planner.

    It demonstrates the needed api, but performs poorly.

    """

    def __init__(self, goal: State, maze: list[list[int]]):
        """

        Args:
            goal: the position to reach
            maze: the maze representation as an array. 0's are walls and 1's are valid positions.
        """
        self.goal = goal
        self.maze = maze

    def sample_action(self, state: State) -> Action:
        """Sample next action for the given state.

        Pure random sampling, not using any heuristic, not leading to great results.

        Args:
            state: the current position in the maze

        Returns:
            sampled action

        """
        return random.sample(list(Action), k=1)[0]


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Application of the above planner on a maze.

    maze_str = """
+-+-+-+-+o+-+-+-+-+-+
|   |             | |
+ +   +-+-+-+ +-+ + +
| | |   |   | | |   |
+ + +-+ +-+ + + + +-+
| |   |     |   |   |
+-+-+-+-+-+x+-+-+-+-+
"""

    maze = []
    for y, line in enumerate(maze_str.strip().split("\n")):
        line = line.rstrip()
        row = []
        for x, c in enumerate(line):
            if c in {" ", "o", "x"}:
                row.append(1)  # spaces are 1s
                if c == "o":
                    start = State(x, y)
                if c == "x":
                    goal = State(x, y)
            else:
                row.append(0)  # walls are 0s
        maze.append(row)

    planner = Planner(goal=goal, maze=maze)

    # rollout
    num_cols = len(maze[0])
    num_rows = len(maze)
    max_steps = 2 * num_cols * num_rows
    nb_step = 0
    total_cost = 0
    state = start
    while state != goal and nb_step < max_steps:
        action = planner.sample_action(state)
        state, cost = get_next_state_and_cost(state=state, action=action, maze=maze)
        total_cost += cost
        nb_step += 1

    if state == goal:
        print("Goal reached!")
    else:
        print("Goal not reached.")
    print(f"total cost: {total_cost}")
