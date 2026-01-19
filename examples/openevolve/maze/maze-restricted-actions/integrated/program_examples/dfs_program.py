from typing import Optional

from maze_restricted_domain import Maze

from skdecide import Space


# EVOLVE-BLOCK-START
class Planner:
    """Depth-First-Search planner."""

    def __init__(self, domain: Maze):
        """

        Args:
            domain: the maze to solve
        """
        self.domain = domain
        self.max_depth = 1000
        self.policy: dict[Maze.T_state, Maze.T_event] = {}

    def _get_state_n_applicable_actions_as_a_list(
        self, state: Maze.T_state
    ) -> list[tuple[Maze.T_state, Maze.T_event]]:
        return [
            (state, action)
            for action in self.domain.get_applicable_actions(state).get_elements()
        ]

    def sample_action(
        self,
        state: Maze.T_state,
        applicable_actions: Optional[Space[Maze.T_event]] = None,
    ) -> Maze.T_event:
        """Sample next action for the given state.

        Launch the DFS with the current state as root node.

        Args:
            state: the current position in the maze

        Returns:
            sampled action

        """
        current_state = state
        goal_reached = (
            self.domain.is_goal(current_state) or current_state in self.policy
        )
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
            current_state = self.domain.get_next_state(memory=state_from, action=action)
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
                if self.domain.is_goal(current_state):
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
