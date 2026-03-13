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
    """Depth-First-Search planner."""

    def __init__(self, goal: State, maze: list[list[int]]):
        """

        Args:
            goal: the position to reach
            maze: the maze representation as an array. 0's are walls and 1's are valid positions.
        """
        self.goal = goal
        self.maze = maze
        self.max_depth = 1000
        self.policy: dict[State, Action] = {}

    def _is_goal(self, state: State) -> bool:
        return state == self.goal

    def _get_state_n_applicable_actions_as_a_list(
        self, state: State
    ) -> list[tuple[State, Action]]:
        return [(state, action) for action in Action]

    def sample_action(self, state: State) -> Action:
        """Sample next action for the given state.

        Launch the DFS with the current state as root node.

        Args:
            state: the current position in the maze

        Returns:
            sampled action

        """
        current_state = state
        goal_reached = self._is_goal(current_state) or current_state in self.policy
        queue = self._get_state_n_applicable_actions_as_a_list(current_state)
        visited_states = {current_state}
        current_plan = []
        # DFS loop
        while (
            not goal_reached and len(queue) > 0 and len(current_plan) < self.max_depth
        ):
            state_from, action = queue.pop()
            # rollback the plan if needed (if a dead-end was reached)
            while state_from != current_state:
                current_state, _ = current_plan.pop()

            # update the plan with the new action to test
            current_plan.append((state_from, action))
            # apply
            current_state, _ = get_next_state_and_cost(
                state=state_from, action=action, maze=self.maze
            )
            # check if we reach an already visited state (in case of loops in the graph)
            if current_state in visited_states:
                # drop the move
                current_plan.pop()
                current_state = state_from
            else:
                visited_states.add(current_state)
                # check if we reach
                #  - a goal
                #  - a state from which we know already a policy from a previous call to `solve()`
                if self._is_goal(current_state):
                    # bingo
                    goal_reached = True
                elif current_state in self.policy:
                    # from here the previously computed policy get to the goal: bingo
                    goal_reached = True
                else:
                    # goal not yet reached: we add applicable actions from next state
                    # NB: if we are in a deadend, nothing will be added, so next tested action will be from a previous state
                    queue.extend(
                        self._get_state_n_applicable_actions_as_a_list(current_state)
                    )

        # Check stop reason
        if goal_reached:
            # add computed plan to the policy (update only to keep track of previous call to `solve()`)
            self.policy.update(current_plan)
            return self.policy[state]
        else:
            # solve fails => raise error
            if len(current_plan) >= self.max_depth:
                # due to max_depth
                raise RuntimeError(
                    "The solver was unable to find a solution within the given max depth."
                )
            else:
                # no valid path exists
                raise RuntimeError(
                    "The solver was trapped in a deadend. The domain has no solution from the given initial state."
                )


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
