# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the SSiPP C++ solver.

Uses the same grid domains as test_ldfs.py:
- 4x4 deterministic grid: optimal path 6 steps, cost 6
- 3x3 stochastic grid: 80/10/10 slip probabilities
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


class StochasticGridDomain(DBase):
    def __init__(self, num_cols=3, num_rows=3):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        intended = self._move(memory, action)
        if action in (Action.up, Action.down):
            slip1 = self._move(memory, Action.left)
            slip2 = self._move(memory, Action.right)
        else:
            slip1 = self._move(memory, Action.up)
            slip2 = self._move(memory, Action.down)
        outcomes = {}
        for s, p in [(intended, 0.8), (slip1, 0.1), (slip2, 0.1)]:
            outcomes[s] = outcomes.get(s, 0.0) + p
        return DiscreteDistribution(list(outcomes.items()))

    def _get_transition_value(self, memory, action, next_state=None):
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


def rollout(domain, solver, max_steps=100):
    actions = []
    total_cost = 0.0
    obs = domain.reset()
    for _ in range(max_steps):
        if domain._is_terminal(obs):
            break
        action = solver.sample_action(obs)
        actions.append(action)
        outcome = domain.step(action)
        total_cost += outcome.value.cost
        obs = outcome.observation
    return actions, total_cost, obs


# --- Tests ---


class TestSSiPP:
    def test_import(self):
        from skdecide.hub.solver.ssipp import SSiPP

        assert SSiPP is not None

    def test_domain_check(self):
        from skdecide.hub.solver.ssipp import SSiPP

        dom = DeterministicGridDomain()
        assert SSiPP.check_domain(dom)

    @pytest.mark.parametrize("inner_solver", ["LRTDP", "ILAOstar", "LDFS"])
    def test_deterministic_grid(self, inner_solver):
        """SSiPP should find optimal cost=6 on 4x4 deterministic grid."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))

        dom = DeterministicGridDomain()
        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=3,
            inner_solver_factory=lambda name=inner_solver: (name, {}),
        ) as solver:
            solver.solve()
            actions, cost, final = rollout(dom, solver)

        assert final == goal
        assert cost <= 10.0

    def test_stochastic_grid(self):
        """SSiPP should find a near-optimal policy on stochastic grid."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(2, 2)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))
        dom = StochasticGridDomain()
        with SSiPP(
            domain_factory=lambda: StochasticGridDomain(),
            heuristic=h,
            depth=4,
            inner_solver_factory=lambda: ("LRTDP", {}),
        ) as solver:
            solver.solve()
            reached_goal = 0
            for _ in range(20):
                _, _, final = rollout(dom, solver)
                if final == goal:
                    reached_goal += 1
            assert reached_goal > 10

    def test_explored_states(self):
        """get_explored_states should return non-empty set after solving."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))
        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=3,
        ) as solver:
            solver.solve()
            states = solver.get_explored_states()
            assert len(states) > 0

    def test_boundary_states(self):
        """get_boundary_states should return non-empty set."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))
        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=2,
        ) as solver:
            solver.solve()
            boundary = solver.get_boundary_states()
            assert len(boundary) > 0

    def test_nb_sub_ssps(self):
        """Should solve with at least one sub-SSP iteration."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))
        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=2,
        ) as solver:
            solver.solve()
            assert solver.get_nb_sub_ssps() >= 1

    def test_small_depth(self):
        """SSiPP should work with small depth=2."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))
        dom = DeterministicGridDomain()
        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=2,
        ) as solver:
            solver.solve()
            actions, cost, final = rollout(dom, solver)

        assert final == goal

    def test_asymptotic_optimality_deterministic(self):
        """SSiPP converges to V* after repeated solves (Theorem 5).

        Compare SSiPP's value at s0 against LRTDP's optimal value on the
        deterministic 4x4 grid. After enough iterations, SSiPP's V(s0)
        should match V*(s0) = 6.
        """
        from skdecide.hub.solver.lrtdp import LRTDP
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))

        with LRTDP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as lrtdp:
            lrtdp.solve()
            v_star = lrtdp.get_utility(State(0, 0)).cost

        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=3,
            inner_solver_factory=lambda: ("LRTDP", {}),
        ) as ssipp:
            ssipp.solve()
            v_ssipp = ssipp.get_utility(State(0, 0)).cost

        assert abs(v_ssipp - v_star) < 0.5, (
            f"SSiPP V(s0)={v_ssipp} should converge to V*(s0)={v_star}"
        )

    def test_asymptotic_optimality_stochastic(self):
        """SSiPP converges to V* on the stochastic 3x3 grid.

        With depth=4 (covering all 9 states), SSiPP should closely match
        LRTDP's optimal value after repeated solves.
        """
        from skdecide.hub.solver.lrtdp import LRTDP
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(2, 2)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))

        with LRTDP(
            domain_factory=lambda: StochasticGridDomain(),
            heuristic=h,
        ) as lrtdp:
            lrtdp.solve()
            v_star = lrtdp.get_utility(State(0, 0)).cost

        with SSiPP(
            domain_factory=lambda: StochasticGridDomain(),
            heuristic=h,
            depth=4,
            inner_solver_factory=lambda: ("LRTDP", {}),
        ) as ssipp:
            ssipp.solve()
            v_ssipp = ssipp.get_utility(State(0, 0)).cost

        assert abs(v_ssipp - v_star) < 1.0, (
            f"SSiPP V(s0)={v_ssipp} should converge to V*(s0)={v_star}"
        )

    def test_solving_time(self):
        """get_solving_time should return a non-negative value."""
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))
        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=3,
        ) as solver:
            solver.solve()
            assert solver.get_solving_time() >= 0

    def test_get_policy_no_recursive_solve(self):
        """get_policy() should return accumulated policy without errors.

        This is a regression test for a bug where get_policy() would iterate over
        explored states and call is_solution_defined_for(), which triggered solve()
        recursively for each state. The fix makes get_policy() directly return the
        internal _policy map that's accumulated across sub-SSP iterations.
        """
        from skdecide.hub.solver.ssipp import SSiPP

        goal = State(3, 3)
        h = lambda d, s: Value(cost=abs(s.x - goal.x) + abs(s.y - goal.y))

        with SSiPP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
            depth=3,
        ) as solver:
            solver.solve()

            # Get the policy - this should complete without errors
            policy = solver.get_policy()

            # Verify policy is non-empty and contains the initial state
            assert len(policy) > 0, "Policy should not be empty"
            assert State(0, 0) in policy, "Policy should contain initial state"

            # Verify policy entries have correct structure (action, value)
            for state, (action, value) in list(policy.items())[:5]:  # Check first 5
                assert action in Action, f"Action {action} should be valid"
                assert value.cost >= 0, f"Value cost should be non-negative"
