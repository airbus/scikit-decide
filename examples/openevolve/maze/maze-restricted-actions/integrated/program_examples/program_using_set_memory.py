from typing import Optional

from maze_restricted_domain import Maze

from skdecide import Space


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

    def sample_action(
        self,
        state: Maze.T_state,
        applicable_actions: Optional[Space[Maze.T_event]] = None,
    ) -> Maze.T_event:
        """Sample next action for the given state.

        Pure random sampling, not using any heuristic, not leading to great results.

        Args:
            state: the current position in the maze

        Returns:
            sampled action

        """
        goal: Maze.T_state = self.domain.get_goals().get_elements()[0]
        above_goal = Maze.T_state(goal.x, goal.y - 1)
        self.domain.set_memory(above_goal)
        return Maze.T_event.down


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
