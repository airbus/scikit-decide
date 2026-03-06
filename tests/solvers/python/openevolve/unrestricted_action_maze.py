from typing import Optional

from skdecide import Domain, Space, Value
from skdecide.builders.domain import (
    DeterministicInitialized,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.hub.domain.maze.maze import Action, State
from skdecide.hub.domain.maze.maze import Maze as HubMaze


class D(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicTransitions,
    UnrestrictedActions,
    DeterministicInitialized,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = None


class BaseMaze(D):
    def __init__(self, maze_str: Optional[str] = None):
        if maze_str is None:
            self._hub_maze = HubMaze()
        else:
            self._hub_maze = HubMaze(maze_str=maze_str)
        self.n_cells = self._hub_maze._num_rows * self._hub_maze._num_cols

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        return self._hub_maze._get_next_state(memory=memory, action=action)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return self._hub_maze._get_transition_value(
            memory=memory, action=action, next_state=next_state
        )

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._hub_maze._is_terminal(state=state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return self._hub_maze._get_action_space_()

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return self._hub_maze._get_goals_()

    def _get_initial_state_(self) -> D.T_state:
        return self._hub_maze._get_initial_state_()

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return self._hub_maze._get_observation_space_()


class Maze(BaseMaze, Goals): ...


class NoGoalMaze(BaseMaze): ...
