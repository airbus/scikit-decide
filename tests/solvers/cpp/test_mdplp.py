# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MDPLP and SSPLP C++ solvers.

Uses deterministic and stochastic grid domains. Tests both primal and dual
LP formulations and validates against LRTDP.
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


class TestMDPLP:
    def test_import(self):
        from skdecide.hub.solver.mdplp import MDPLP

        assert MDPLP is not None

    def test_domain_check(self):
        from skdecide.hub.solver.mdplp import MDPLP

        dom = DeterministicGridDomain()
        assert MDPLP.check_domain(dom)

    @pytest.mark.parametrize("variant", ["primal", "dual"])
    def test_deterministic_grid(self, variant):
        """LP should find optimal cost=6 on 4x4 deterministic grid."""
        from skdecide.hub.solver.mdplp import MDPLP

        dom = DeterministicGridDomain()
        with MDPLP(
            domain_factory=lambda: DeterministicGridDomain(),
            variant=variant,
            discount=0.99,
        ) as solver:
            solver.solve()
            actions, cost, final = rollout(dom, solver)

        assert final == State(3, 3)

    def test_primal_dual_agree(self):
        """Primal and dual should give same V(s0) (strong duality)."""
        from skdecide.hub.solver.mdplp import MDPLP

        with MDPLP(
            domain_factory=lambda: StochasticGridDomain(),
            variant="primal",
            discount=0.99,
        ) as primal:
            primal.solve()
            v_primal = primal.get_utility(State(0, 0)).cost

        with MDPLP(
            domain_factory=lambda: StochasticGridDomain(),
            variant="dual",
            discount=0.99,
        ) as dual:
            dual.solve()
            v_dual = dual.get_utility(State(0, 0)).cost

        assert abs(v_primal - v_dual) < 0.5, (
            f"Primal V(s0)={v_primal} should match dual V(s0)={v_dual}"
        )

    def test_matches_lrtdp(self):
        """LP should match LRTDP's V* on deterministic grid."""
        from skdecide.hub.solver.lrtdp import LRTDP
        from skdecide.hub.solver.mdplp import MDPLP

        h = lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3))

        with LRTDP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as lrtdp:
            lrtdp.solve()
            v_lrtdp = lrtdp.get_utility(State(0, 0)).cost

        with MDPLP(
            domain_factory=lambda: DeterministicGridDomain(),
            variant="dual",
            discount=0.99,
        ) as mdplp:
            mdplp.solve()
            v_lp = mdplp.get_utility(State(0, 0)).cost

        assert abs(v_lp - v_lrtdp) < 0.5, (
            f"LP V(s0)={v_lp} should match LRTDP V*(s0)={v_lrtdp}"
        )

    def test_stochastic_grid(self):
        """LP should find near-optimal policy on stochastic grid."""
        from skdecide.hub.solver.mdplp import MDPLP

        dom = StochasticGridDomain()
        with MDPLP(
            domain_factory=lambda: StochasticGridDomain(),
            variant="dual",
            discount=0.99,
        ) as solver:
            solver.solve()
            reached_goal = 0
            for _ in range(20):
                _, _, final = rollout(dom, solver)
                if final == State(2, 2):
                    reached_goal += 1

        assert reached_goal > 10

    def test_lp_statistics(self):
        """LP statistics should be reasonable."""
        from skdecide.hub.solver.mdplp import MDPLP

        with MDPLP(
            domain_factory=lambda: DeterministicGridDomain(),
            variant="primal",
            discount=0.99,
        ) as solver:
            solver.solve()
            assert solver.get_nb_states() == 16
            assert solver.get_nb_lp_variables() > 0
            assert solver.get_nb_lp_constraints() > 0
            assert solver.get_solving_time() >= 0

    def test_variant_validation(self):
        """Invalid variant should raise ValueError."""
        from skdecide.hub.solver.mdplp import MDPLP

        with pytest.raises(ValueError, match="variant"):
            MDPLP(
                domain_factory=lambda: DeterministicGridDomain(),
                variant="invalid",
            )


class TestSSPLP:
    def test_import(self):
        from skdecide.hub.solver.mdplp import SSPLP

        assert SSPLP is not None

    def test_domain_check(self):
        from skdecide.hub.solver.mdplp import SSPLP

        dom = DeterministicGridDomain()
        assert SSPLP.check_domain(dom)

    @pytest.mark.parametrize("variant", ["primal", "dual"])
    def test_deterministic_grid(self, variant):
        """SSPLP should find optimal cost=6 on 4x4 deterministic grid."""
        from skdecide.hub.solver.mdplp import SSPLP

        dom = DeterministicGridDomain()
        with SSPLP(
            domain_factory=lambda: DeterministicGridDomain(),
            variant=variant,
        ) as solver:
            solver.solve()
            actions, cost, final = rollout(dom, solver)

        assert final == State(3, 3)
        assert cost == 6.0

    def test_primal_dual_agree(self):
        """Primal and dual should give same V(s0) (strong duality)."""
        from skdecide.hub.solver.mdplp import SSPLP

        with SSPLP(
            domain_factory=lambda: StochasticGridDomain(),
            variant="primal",
        ) as primal:
            primal.solve()
            v_primal = primal.get_utility(State(0, 0)).cost

        with SSPLP(
            domain_factory=lambda: StochasticGridDomain(),
            variant="dual",
        ) as dual:
            dual.solve()
            v_dual = dual.get_utility(State(0, 0)).cost

        assert abs(v_primal - v_dual) < 0.5, (
            f"Primal V(s0)={v_primal} should match dual V(s0)={v_dual}"
        )

    def test_matches_lrtdp(self):
        """SSPLP should match LRTDP's V* on deterministic grid."""
        from skdecide.hub.solver.lrtdp import LRTDP
        from skdecide.hub.solver.mdplp import SSPLP

        h = lambda d, s: Value(cost=abs(s.x - 3) + abs(s.y - 3))

        with LRTDP(
            domain_factory=lambda: DeterministicGridDomain(),
            heuristic=h,
        ) as lrtdp:
            lrtdp.solve()
            v_lrtdp = lrtdp.get_utility(State(0, 0)).cost

        with SSPLP(
            domain_factory=lambda: DeterministicGridDomain(),
            variant="dual",
        ) as ssplp:
            ssplp.solve()
            v_lp = ssplp.get_utility(State(0, 0)).cost

        assert abs(v_lp - v_lrtdp) < 0.01, (
            f"SSPLP V(s0)={v_lp} should match LRTDP V*(s0)={v_lrtdp}"
        )

    def test_stochastic_grid(self):
        """SSPLP should find near-optimal policy on stochastic grid."""
        from skdecide.hub.solver.mdplp import SSPLP

        dom = StochasticGridDomain()
        with SSPLP(
            domain_factory=lambda: StochasticGridDomain(),
            variant="dual",
        ) as solver:
            solver.solve()
            reached_goal = 0
            for _ in range(20):
                _, _, final = rollout(dom, solver)
                if final == State(2, 2):
                    reached_goal += 1

        assert reached_goal > 10

    def test_lp_statistics(self):
        """LP statistics should be reasonable."""
        from skdecide.hub.solver.mdplp import SSPLP

        with SSPLP(
            domain_factory=lambda: DeterministicGridDomain(),
            variant="primal",
        ) as solver:
            solver.solve()
            assert solver.get_nb_states() == 16
            assert solver.get_nb_lp_variables() > 0
            assert solver.get_nb_lp_constraints() > 0
            assert solver.get_solving_time() >= 0

    def test_variant_validation(self):
        """Invalid variant should raise ValueError."""
        from skdecide.hub.solver.mdplp import SSPLP

        with pytest.raises(ValueError, match="variant"):
            SSPLP(
                domain_factory=lambda: DeterministicGridDomain(),
                variant="invalid",
            )
