import random

from skdecide.hub.domain.maze import Maze


# EVOLVE-BLOCK-START
class Planner:
    """Random planner.

    It demonstrates the needed api, but performs poorly.

    """

    def __init__(self, domain: Maze):
        """

        Args:
            domain: the maze to solve
        """
        self.domain = domain

    def sample_action(self, state: Maze.T_state) -> Maze.T_event:
        """Sample next action for the given state.

        Pure random sampling, not using any heuristic, not leading to great results.

        Args:
            state: the current position in the maze

        Returns:
            sampled action

        """
        actions = list(self.domain.get_action_space())
        return random.sample(actions, k=1)[0]


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

    maze = Maze(maze_str=maze_str)
    planner = Planner(domain=maze)

    # rollout
    max_steps = 2 * maze._num_cols * maze._num_rows
    nb_step = 0
    total_cost = 0
    state = maze.get_initial_state()
    while not maze.is_goal(state) and nb_step < max_steps:
        action = planner.sample_action(state)
        next_state = maze.get_next_state(memory=state, action=action)
        value = maze.get_transition_value(
            memory=state, action=action, next_state=next_state
        )
        state = next_state
        total_cost += value.cost
        nb_step += 1

    if maze.is_goal(state):
        print("Goal reached!")
    else:
        print("Goal not reached.")
    print(f"total cost: {total_cost}")
