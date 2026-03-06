# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

from skdecide import DeterministicPlanningDomain, Space, Value
from skdecide.builders.domain import Renderable
from skdecide.hub.domain.maze.maze import Action, State
from skdecide.hub.domain.maze.maze import Maze as HubMaze
from skdecide.hub.space.gym import ListSpace


class D(DeterministicPlanningDomain, Renderable):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class Maze(D):
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
        if self._is_applicable_action_from(action=action, memory=memory):
            return self._hub_maze._get_next_state(memory=memory, action=action)
        else:
            raise ValueError(
                f"The action {action} is not applicable for the state {memory}."
            )

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if self._is_applicable_action_from(action=action, memory=memory):
            return self._hub_maze._get_transition_value(
                memory=memory, action=action, next_state=next_state
            )
        else:
            raise ValueError(
                f"The action {action} is not applicable for the state {memory}."
            )

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._hub_maze._is_terminal(state=state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return self._hub_maze._get_action_space_()

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(
            [
                action
                for action in Action
                if self._is_applicable_action_from(action=action, memory=memory)
            ]
        )

    def _is_applicable_action_from(
        self, action: D.T_agent[D.T_event], memory: D.T_memory[D.T_state]
    ) -> bool:
        next_state = self._hub_maze._get_next_state(memory=memory, action=action)
        return not (next_state.x == memory.x and next_state.y == memory.y)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return self._hub_maze._get_goals_()

    def _get_initial_state_(self) -> D.T_state:
        return self._hub_maze._get_initial_state_()

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return self._hub_maze._get_observation_space_()

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        return self._hub_maze._render_from(memory=memory, **kwargs)
