# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple 5x5 stochastic grid domain for testing determinization solvers.

States: (x, y) coordinates in a 5x5 grid, goal at (4, 4)
Actions: {north, south, east, west}
Transitions: 80% intended direction, 10% each perpendicular direction
Walls: Boundaries (stay in place on collision)
Cost: 1 per step
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from skdecide import DiscreteDistribution, Domain, ImplicitSpace, Value
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
    UncertainTransitions,
)


class GridAction(Enum):
    """Actions in the grid world."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class GridState(NamedTuple):
    """State is a 2D coordinate."""

    x: int
    y: int


class StochasticGridDomain(
    Domain,
    SingleAgent,
    Sequential,
    UncertainTransitions,
    Actions,
    Goals,
    Markovian,
    FullyObservable,
    PositiveCosts,
    DeterministicInitialized,
):
    """5x5 grid with stochastic transitions.

    The agent starts at (0, 0) and must reach the goal at (4, 4).
    Each action has 80% probability of moving in the intended direction
    and 10% probability of moving in each perpendicular direction.
    Walls at boundaries cause the agent to stay in place.
    """

    T_state = GridState
    T_observation = T_state
    T_event = GridAction
    T_value = float
    T_predicate = bool
    T_info = None

    def __init__(self, size: int = 5):
        """Initialize the stochastic grid domain.

        # Parameters
        size: Grid size (default 5 for a 5x5 grid).
        """
        self._size = size
        self._goal = GridState(size - 1, size - 1)

    def _get_action_space_(self) -> ImplicitSpace:
        """Return the action space."""
        return ImplicitSpace(lambda action: isinstance(action, GridAction))

    def _get_applicable_actions_from(self, memory: GridState):
        """All actions are always applicable.

        Returns a custom space object that supports both enumeration and membership.
        """

        class ActionSpace:
            def __init__(self, actions):
                self._actions = actions

            def get_elements(self):
                return self._actions

            def contains(self, action):
                return action in self._actions

        return ActionSpace(list(GridAction))

    def _get_next_state_distribution(
        self, memory: GridState, action: GridAction
    ) -> DiscreteDistribution:
        """Return the distribution of next states given current state and action.

        Each action has:
        - 80% probability of moving in the intended direction
        - 10% probability of moving in each perpendicular direction
        """
        # Define perpendicular directions for each action
        perpendicular = {
            GridAction.NORTH: [GridAction.EAST, GridAction.WEST],
            GridAction.SOUTH: [GridAction.EAST, GridAction.WEST],
            GridAction.EAST: [GridAction.NORTH, GridAction.SOUTH],
            GridAction.WEST: [GridAction.NORTH, GridAction.SOUTH],
        }

        # Compute next state for each possible outcome
        outcomes = []

        # 80% intended direction
        intended_next = self._apply_action(memory, action)
        outcomes.append((intended_next, 0.8))

        # 10% each perpendicular direction
        for perp_action in perpendicular[action]:
            perp_next = self._apply_action(memory, perp_action)
            outcomes.append((perp_next, 0.1))

        return DiscreteDistribution(outcomes)

    def _apply_action(self, state: GridState, action: GridAction) -> GridState:
        """Apply an action to a state (deterministic transition helper)."""
        x, y = state.x, state.y

        if action == GridAction.NORTH:
            y = min(y + 1, self._size - 1)
        elif action == GridAction.SOUTH:
            y = max(y - 1, 0)
        elif action == GridAction.EAST:
            x = min(x + 1, self._size - 1)
        elif action == GridAction.WEST:
            x = max(x - 1, 0)

        return GridState(x, y)

    def _get_transition_value(
        self,
        memory: GridState,
        action: GridAction,
        next_state: GridState | None = None,
    ) -> Value:
        """Cost is 1 per step."""
        return Value(cost=1.0)

    def _is_terminal(self, state: GridState) -> bool:
        """Terminal when goal is reached."""
        return state == self._goal

    def _get_goals_(self) -> ImplicitSpace:
        """Goal is at (size-1, size-1)."""
        return ImplicitSpace(lambda state: state == self._goal)

    def _state_reset(self) -> GridState:
        """Start at (0, 0)."""
        return GridState(0, 0)

    def _get_observation_space_(self) -> ImplicitSpace:
        """All (x, y) coordinates in the grid."""
        return ImplicitSpace(
            lambda state: isinstance(state, GridState)
            and 0 <= state.x < self._size
            and 0 <= state.y < self._size
        )
