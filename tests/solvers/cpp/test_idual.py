# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for IDual and CIDual C++ solvers.

Tests incremental dual LP heuristic search for SSPs and constrained SSPs.
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
    Constrained,
    DeterministicInitialized,
    EnumerableTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.core import BoundConstraint
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


# Constrained domain: secondary cost = fuel (movement costs more fuel
# when going right or down)
class ConstrainedGridDomain(DBase, Constrained):
    def __init__(self, num_cols=3, num_rows=3, fuel_budget=10.0):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self._fuel_budget = fuel_budget

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        return SingleValueDistribution(self._move(memory, action))

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

    def _get_constraints_(self):
        def fuel_cost(memory, action, next_state=None):
            if action in (Action.right, Action.down):
                return 2.0
            return 1.0

        return [
            BoundConstraint(
                fuel_cost, "<=", self._fuel_budget, depends_on_next_state=False
            )
        ]

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


# --- Unconstrained IDual tests ---


class TestIDual:
    def test_import(self):
        from skdecide.hub.solver.idual import IDual

        assert IDual is not None

    def test_domain_check(self):
        from skdecide.hub.solver.idual import IDual

        dom = DeterministicGridDomain()
        assert IDual.check_domain(dom)

    def test_deterministic_grid(self):
        """IDual should find optimal cost=6 on 4x4 deterministic grid."""
        from skdecide.hub.solver.idual import IDual

        dom = DeterministicGridDomain()
        with IDual(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3)),
        ) as solver:
            solver.solve()
            actions, cost, final = rollout(dom, solver)

        assert final == State(3, 3)
        assert cost == 6.0

    @pytest.mark.parametrize("reference_solver", ["LRTDP", "VI"])
    def test_matches_optimal(self, reference_solver):
        """IDual V(s0) should match VI/LRTDP V* on deterministic grid."""
        from skdecide.hub.solver.idual import IDual

        h = lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3))

        if reference_solver == "LRTDP":
            from skdecide.hub.solver.lrtdp import LRTDP

            with LRTDP(
                domain_factory=lambda: DeterministicGridDomain(),
                heuristic=h,
            ) as ref:
                ref.solve()
                v_ref = ref.get_utility(State(0, 0)).cost
        else:
            from skdecide.hub.solver.vi import VI

            with VI(
                domain_factory=lambda: DeterministicGridDomain(),
            ) as ref:
                ref.solve()
                v_ref = ref.get_utility(State(0, 0)).cost

        with IDual(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as idual:
            idual.solve()
            v_idual = idual.get_utility(State(0, 0)).cost

        assert abs(v_idual - v_ref) < 0.05, (
            f"IDual V(s0)={v_idual} should match {reference_solver} V*(s0)={v_ref}"
        )

    def test_stochastic_matches_vi(self):
        """IDual V(s0) should match VI on stochastic grid."""
        from skdecide.hub.solver.idual import IDual
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: StochasticGridDomain(),
        ) as vi:
            vi.solve()
            v_vi = vi.get_utility(State(0, 0)).cost

        with IDual(
            domain_factory=lambda: StochasticGridDomain(),
        ) as idual:
            idual.solve()
            v_idual = idual.get_utility(State(0, 0)).cost

        assert abs(v_idual - v_vi) < 0.05, (
            f"IDual V(s0)={v_idual} should match VI V*(s0)={v_vi}"
        )

    def test_stochastic_grid(self):
        """IDual should find near-optimal policy on stochastic grid."""
        from skdecide.hub.solver.idual import IDual

        dom = StochasticGridDomain()
        with IDual(
            domain_factory=lambda: StochasticGridDomain(),
        ) as solver:
            solver.solve()
            reached_goal = 0
            for _ in range(20):
                _, _, final = rollout(dom, solver)
                if final == State(2, 2):
                    reached_goal += 1

        assert reached_goal > 10

    def test_explores_fewer_states(self):
        """IDual with heuristic should explore fewer states than total."""
        from skdecide.hub.solver.idual import IDual

        with IDual(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3)),
        ) as solver:
            solver.solve()
            n_explored = solver.get_nb_explored_states()
            n_lp_iters = solver.get_nb_lp_iterations()

        assert n_explored <= 16
        assert n_lp_iters >= 1

    def test_statistics(self):
        """Statistics should be reasonable."""
        from skdecide.hub.solver.idual import IDual

        with IDual(
            domain_factory=lambda: DeterministicGridDomain(),
        ) as solver:
            solver.solve()
            assert solver.get_nb_explored_states() > 0
            assert solver.get_nb_lp_iterations() > 0
            assert solver.get_solving_time() >= 0
            assert len(solver.get_explored_states()) > 0


# --- Constrained CIDual tests ---


class TestCIDual:
    def test_import(self):
        from skdecide.hub.solver.idual import CIDual

        assert CIDual is not None

    def test_domain_check(self):
        from skdecide.hub.solver.idual import CIDual

        dom = ConstrainedGridDomain()
        assert CIDual.check_domain(dom)

    def test_loose_constraint_matches_idual(self):
        """CIDual with very loose constraint should match IDual."""
        from skdecide.hub.solver.idual import CIDual, IDual

        h = lambda d, s: Value(cost=abs(s.x - 2) + abs(s.y - 2))

        with IDual(
            domain_factory=lambda: DeterministicGridDomain(3, 3),
            heuristic=h,
        ) as idual:
            idual.solve()
            v_idual = idual.get_utility(State(0, 0)).cost

        with CIDual(
            domain_factory=lambda: ConstrainedGridDomain(3, 3, fuel_budget=100.0),
            heuristic=h,
        ) as cidual:
            cidual.solve()
            v_cidual = cidual.get_utility(State(0, 0)).cost

        assert abs(v_idual - v_cidual) < 0.5, (
            f"CIDual V(s0)={v_cidual} should match IDual V(s0)={v_idual}"
        )

    def test_feasible_constraint(self):
        """CIDual with feasible constraint should find a policy."""
        from skdecide.hub.solver.idual import CIDual

        dom = ConstrainedGridDomain(3, 3, fuel_budget=10.0)
        with CIDual(
            domain_factory=lambda: ConstrainedGridDomain(3, 3, fuel_budget=10.0),
        ) as solver:
            solver.solve()
            _, _, final = rollout(dom, solver)

        assert final == State(2, 2)

    def test_statistics(self):
        """Statistics should be reasonable."""
        from skdecide.hub.solver.idual import CIDual

        with CIDual(
            domain_factory=lambda: ConstrainedGridDomain(3, 3, fuel_budget=10.0),
        ) as solver:
            solver.solve()
            assert solver.get_nb_explored_states() > 0
            assert solver.get_nb_lp_iterations() > 0
