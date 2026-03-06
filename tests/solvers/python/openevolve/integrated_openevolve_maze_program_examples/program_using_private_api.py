from __future__ import annotations

from typing import Optional

from skdecide import Space


# EVOLVE-BLOCK-START
class Planner:
    """Random planner.

    It demonstrates the needed api, but performs poorly.

    """

    def __init__(self, domain: "Maze"):
        """

        Args:
            domain: the maze to solve
        """
        self.domain = domain

    def sample_action(
        self,
        state: "Maze.T_state",
        applicable_actions: "Optional[Space[Maze.T_event]]" = None,
    ) -> "Maze.T_event":
        """Sample next action for the given state.

        Pure random sampling, not using any heuristic, not leading to great results.

        Args:
            state: the current position in the maze

        Returns:
            sampled action

        """
        goal = self.domain._hub_maze._goal
        if applicable_actions is None:
            return self.domain.get_applicable_actions(memory=state).sample()
        else:
            return applicable_actions.sample()


# EVOLVE-BLOCK-END
