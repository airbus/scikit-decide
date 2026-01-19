import random
from enum import Enum

State = tuple[int, int]


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class InvalidActionError(Exception): ...


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
        next_state = (state[0] - 1, state[1])
    elif action == Action.right:
        next_state = (state[0] + 1, state[1])
    elif action == Action.up:
        next_state = (state[0], state[1] - 1)
    elif action == Action.down:
        next_state = (state[0], state[1] + 1)
    else:
        raise ValueError("action not valid.")

    # check if valid move and compute cost accordingly
    num_cols = len(maze[0])
    num_rows = len(maze)
    if (
        0 <= next_state[0] < num_cols  # not too high or low
        and 0 <= next_state[1] < num_rows  # not too on the left or right
        and maze[next_state[1]][next_state[0]] == 1  # not in a wall
    ):
        cost = 1
    else:
        raise InvalidActionError()

    return next_state, cost


def get_applicable_actions(state: State, maze: list[list[int]]) -> list[Action]:
    actions = []
    for action in Action:
        try:
            get_next_state_and_cost(state, action, maze)
        except InvalidActionError:
            ...
        else:
            actions.append(action)

    return actions


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

    def sample_action(self, state: State, applicable_actions: list[str]) -> Action:
        """Sample next action for the given state.

        Pure random sampling, not using any heuristic, not leading to great results.

        Args:
            state: the current position in the maze
            applicable_actions: list of applicable actions at the current position

        Returns:
            sampled action

        """
        actions = maze.get_actions()
        return Action[random.sample(applicable_actions, k=1)[0]]


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
                    start = (x, y)
                if c == "x":
                    goal = (x, y)
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
        applicable_actions = get_applicable_actions(state, maze)
        action = planner.sample_action(state, [a.name for a in applicable_actions])
        state, cost = get_next_state_and_cost(state=state, action=action, maze=maze)
        total_cost += cost
        nb_step += 1

    if state == goal:
        print("Goal reached!")
    else:
        print("Goal not reached.")
    print(f"total cost: {total_cost}")
