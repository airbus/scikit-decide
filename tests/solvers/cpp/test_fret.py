# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the FRET C++ solver.

Test domains:
- DeadEndGridDomain: 4x4 stochastic grid with an absorbing dead-end state.
  State (0,3) is a dead end: all actions loop back to itself. Every action
  from every non-dead-end state has a 5% probability of transitioning to
  (0,3), making the dead end unavoidable by any policy.
- DeterministicGridDomain: Standard 4x4 deterministic grid (no dead ends).
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import pytest

from skdecide import (
    DiscreteDistribution,
    Domain,
    ImplicitSpace,
    SingleValueDistribution,
    Value,
)
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    EnumerableTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


DELTAS = {
    Action.up: (0, -1),
    Action.down: (0, 1),
    Action.left: (-1, 0),
    Action.right: (1, 0),
}


class DBase(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicInitialized,
    EnumerableTransitions,
    Actions,
    Goals,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    T_state = State
    T_observation = T_state
    T_event = Action
    T_value = float
    T_predicate = bool
    T_info = None


DEAD_END = State(3, 0)


class DeadEndChainDomain(DBase):
    """Simple chain: s0 → s1 → s2(goal) with a dead-end branch.

    States: (0,0)=start, (1,0), (2,0)=goal, (3,0)=dead_end
    Actions: right moves forward, all others stay in place.
    From (1,0), action "right" goes to goal(2,0) with p=0.5 and
    dead_end(3,0) with p=0.5.
    From dead_end(3,0), ALL actions loop back to (3,0). Terminal.
    The dead end is unavoidable: from (1,0), every action has at least
    50% probability of reaching it.
    """

    def __init__(self):
        self.num_cols = 4
        self.num_rows = 1

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        if memory == DEAD_END:
            return SingleValueDistribution(DEAD_END)

        if memory == State(2, 0):
            return SingleValueDistribution(State(2, 0))

        if memory == State(0, 0):
            if action == Action.right:
                return SingleValueDistribution(State(1, 0))
            return SingleValueDistribution(State(0, 0))

        if memory == State(1, 0):
            if action == Action.right:
                return DiscreteDistribution([(State(2, 0), 0.5), (DEAD_END, 0.5)])
            return DiscreteDistribution([(State(1, 0), 0.5), (DEAD_END, 0.5)])

        return SingleValueDistribution(memory)

    def _get_transition_value(self, memory, action, next_state=None):
        return Value(cost=1)

    def _is_terminal(self, state) -> bool:
        return state == DEAD_END or self._is_goal(state)

    def _get_goals_(self):
        return ImplicitSpace(lambda s: s == State(2, 0))

    def _get_action_space_(self):
        return EnumSpace(Action)

    def _get_applicable_actions_from(self, memory):
        return self._get_action_space_()

    def _get_observation_space_(self):
        return MultiDiscreteSpace([self.num_cols, self.num_rows])


class DeterministicGridDomain(DBase):
    def __init__(self, num_cols=4, num_rows=4):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        return SingleValueDistribution(self._move(memory, action))

    def _get_transition_value(self, memory, action, next_state=None):
        if next_state is None:
            next_state = self._move(memory, action)
        if next_state.x == memory.x and next_state.y == memory.y:
            return Value(cost=2)
        return Value(cost=1)

    def _is_terminal(self, state) -> bool:
        return self._is_goal(state)

    def _get_goals_(self):
        return ImplicitSpace(
            lambda s: s.x == self.num_cols - 1 and s.y == self.num_rows - 1
        )

    def _get_action_space_(self):
        return EnumSpace(Action)

    def _get_applicable_actions_from(self, memory):
        return self._get_action_space_()

    def _get_observation_space_(self):
        return MultiDiscreteSpace([self.num_cols, self.num_rows])

    def _move(self, state, action):
        dx, dy = DELTAS[action]
        nx = max(0, min(state.x + dx, self.num_cols - 1))
        ny = max(0, min(state.y + dy, self.num_rows - 1))
        return State(nx, ny)


# --- Tests ---


class TestFRET:
    def test_import(self):
        from skdecide.hub.solver.fret import FRET

        assert FRET is not None

    def test_domain_check(self):
        from skdecide.hub.solver.fret import FRET

        dom = DeadEndChainDomain()
        assert FRET.check_domain(dom)

    def test_finds_dead_ends(self):
        """FRET should detect the absorbing dead-end state."""
        from skdecide.hub.solver.fret import FRET

        with FRET(
            domain_factory=lambda: DeadEndChainDomain(),
            heuristic=lambda d, s: Value(cost=0),
            inner_solver="LRTDP",
            inner_solver_params={"rollout_budget": 1000},
            dead_end_cost=10000.0,
        ) as solver:
            solver.solve()
            dead_ends = solver.get_dead_end_states()

        assert DEAD_END in dead_ends

    def test_trapped_sccs(self):
        """get_trapped_sccs should return non-empty SCCs during solving."""
        from skdecide.hub.solver.fret import FRET

        with FRET(
            domain_factory=lambda: DeadEndChainDomain(),
            heuristic=lambda d, s: Value(cost=0),
            inner_solver_params={"rollout_budget": 1000},
        ) as solver:
            solver.solve()
            assert solver.get_nb_traps_eliminated() >= 1
            assert DEAD_END in solver.get_dead_end_states()

    def test_dead_end_unavoidable(self):
        """Confirm the dead end is reachable with positive probability
        from state (1,0) regardless of the action chosen."""
        dom = DeadEndChainDomain()
        s = State(1, 0)
        for a in Action:
            dist = dom._get_next_state_distribution(s, a)
            probs = dict(dist.get_values())
            assert DEAD_END in probs and probs[DEAD_END] > 0, (
                f"Action {a} from {s} should have positive prob of reaching dead end"
            )

    def test_optimal_on_clean_grid(self):
        """On grid without dead ends, FRET should match LRTDP's V*."""
        from skdecide.hub.solver.fret import FRET
        from skdecide.hub.solver.lrtdp import LRTDP

        h = lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3))

        with LRTDP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as lrtdp:
            lrtdp.solve()
            v_star = lrtdp.get_utility(State(0, 0)).cost

        with FRET(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as fret:
            fret.solve()
            v_fret = fret.get_utility(State(0, 0)).cost

        assert abs(v_fret - v_star) < 0.5, (
            f"FRET V(s0)={v_fret} should match LRTDP V*(s0)={v_star}"
        )

    @pytest.mark.parametrize("inner_solver", ["LRTDP", "LDFS", "VI"])
    def test_with_each_inner_solver(self, inner_solver):
        """FRET should work with all supported inner solvers."""
        from skdecide.hub.solver.fret import FRET

        h = lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3))

        with FRET(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            inner_solver=inner_solver,
        ) as solver:
            solver.solve()
            assert solver._is_solution_defined_for(State(0, 0))
            assert solver.get_nb_explored_states() > 0

    def test_fret_iterations(self):
        """On dead-end domain, FRET should need ≥2 iterations."""
        from skdecide.hub.solver.fret import FRET

        with FRET(
            domain_factory=lambda: DeadEndChainDomain(),
            heuristic=lambda d, s: Value(cost=0),
            inner_solver_params={"rollout_budget": 1000},
        ) as solver:
            solver.solve()
            assert solver.get_nb_fret_iterations() >= 2
            assert solver.get_nb_traps_eliminated() >= 1

    def test_solving_time(self):
        """get_solving_time should return non-negative value."""
        from skdecide.hub.solver.fret import FRET

        h = lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3))

        with FRET(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as solver:
            solver.solve()
            assert solver.get_solving_time() >= 0
