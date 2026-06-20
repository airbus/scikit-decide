# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the ILAOstar C++ solver.

Uses a 10x10 deterministic grid domain where the goal is to reach the
bottom-right corner from the top-left corner with minimal cost.
"""

from __future__ import annotations

from enum import Enum
from math import sqrt
from typing import NamedTuple

from skdecide import (
    DeterministicPlanningDomain,
    ImplicitSpace,
    Space,
    Value,
)
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace


class State(NamedTuple):
    x: int
    y: int
    s: int  # step => to make the domain cycle-free for algorithms like AO*


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class D(DeterministicPlanningDomain, UnrestrictedActions):
    T_state = State
    T_observation = T_state
    T_event = Action
    T_value = float
    T_predicate = bool
    T_info = None


class GridDomain(D):
    def __init__(self, num_cols=10, num_rows=10):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        if action == Action.left:
            next_state = State(max(memory.x - 1, 0), memory.y, memory.s + 1)
        if action == Action.right:
            next_state = State(
                min(memory.x + 1, self.num_cols - 1), memory.y, memory.s + 1
            )
        if action == Action.up:
            next_state = State(memory.x, max(memory.y - 1, 0), memory.s + 1)
        if action == Action.down:
            next_state = State(
                memory.x, min(memory.y + 1, self.num_rows - 1), memory.s + 1
            )

        return next_state

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: D.T_state | None = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if next_state.x == memory.x and next_state.y == memory.y:
            cost = 2  # big penalty when hitting a wall
        else:
            cost = abs(next_state.x - memory.x) + abs(
                next_state.y - memory.y
            )  # every move costs 1

        return Value(cost=cost)

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._is_goal(state) or state.s >= 100

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(Action)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(
            lambda state: state.x == (self.num_cols - 1)
            and state.y == (self.num_rows - 1)
        )

    def _get_initial_state_(self) -> D.T_state:
        return State(x=0, y=0, s=0)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows, 100])


class TestILAOstar:
    """Test ILAOstar (Improved LAO*) on a deterministic grid."""

    def test_best_solution_graph_after_solve(self):
        """get_best_solution_graph() should return a non-empty dict of (action, value)
        tuples after solving, with all entries belonging to the best solution graph."""
        from skdecide.hub.solver.ilaostar import ILAOstar

        with ILAOstar(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            bsg = solver.get_best_solution_graph()

        assert len(bsg) > 0
        # Every entry is a (action, float) tuple
        for state, (action, value) in bsg.items():
            assert isinstance(state, State)
            assert isinstance(action, Action)
            assert isinstance(value, float)
        # The dict cannot have more entries than the full best solution graph size
        assert len(bsg) <= solver.best_solution_graph_size()
        # Initial state should be in the best solution graph with a best action
        assert State(x=0, y=0, s=0) in bsg

    def test_best_solution_graph_subset_of_policy(self):
        """States in get_best_solution_graph() must also appear in get_policy()."""
        from skdecide.hub.solver.ilaostar import ILAOstar

        with ILAOstar(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            bsg = solver.get_best_solution_graph()
            policy = solver.get_policy()

        for state in bsg:
            assert state in policy

    def test_best_solution_graph_in_callback(self):
        """get_best_solution_graph() called from the callback should grow and be
        structurally valid at every iteration."""
        from skdecide.hub.solver.ilaostar import ILAOstar

        snapshots = []

        def cb(slv):
            snap = slv.get_best_solution_graph()
            snapshots.append(snap)
            return False  # never stop early

        with ILAOstar(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            discount=1.0,
            epsilon=0.001,
            callback=cb,
        ) as solver:
            solver.solve()

        assert len(snapshots) > 0
        for snap in snapshots:
            for state, (action, value) in snap.items():
                assert isinstance(state, State)
                assert isinstance(action, Action)
                assert isinstance(value, float)
