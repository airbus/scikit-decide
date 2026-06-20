# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the GPCI C++ solver.

Test domains:
- StochasticGridDomain: 3x3 stochastic grid, no dead-ends. P*(s) = 1.0
  everywhere.
- DeadEndGridDomain: 4x4 stochastic grid with avoidable absorbing dead-end
  cells at (1,2) and (2,1).
- DeadEndChainDomain: Chain s0 -> s1 -> s2(goal) with unavoidable dead-end
  at (3,0). From (1,0), every action has >= 50% probability of reaching
  the dead-end. P*(s0) = 0.5 under the optimal policy.
- DeterministicGridDomain: 4x4 deterministic grid, no dead-ends.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

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


DEAD_ENDS = {State(1, 2), State(2, 1)}


class DeadEndGridDomain(DBase):
    """4x4 stochastic grid with absorbing dead-end cells at (1,2) and (2,1)."""

    def __init__(self, num_cols=4, num_rows=4):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        if memory in DEAD_ENDS:
            return SingleValueDistribution(memory)

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
        return self._is_goal(state) or state in DEAD_ENDS

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


CHAIN_DEAD_END = State(3, 0)


class DeadEndChainDomain(DBase):
    """Chain: s0 -> s1 -> s2(goal) with unavoidable dead-end at (3,0).

    From (1,0), action "right" goes to goal(2,0) with p=0.5 and
    dead_end(3,0) with p=0.5. All other actions from (1,0) go to
    (1,0) with p=0.5 and dead_end(3,0) with p=0.5.
    The dead end is unavoidable from (1,0).
    """

    def __init__(self):
        self.num_cols = 4
        self.num_rows = 1

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        if memory == CHAIN_DEAD_END:
            return SingleValueDistribution(CHAIN_DEAD_END)
        if memory == State(2, 0):
            return SingleValueDistribution(State(2, 0))
        if memory == State(0, 0):
            if action == Action.right:
                return SingleValueDistribution(State(1, 0))
            return SingleValueDistribution(State(0, 0))
        if memory == State(1, 0):
            if action == Action.right:
                return DiscreteDistribution([(State(2, 0), 0.5), (CHAIN_DEAD_END, 0.5)])
            return DiscreteDistribution([(State(1, 0), 0.5), (CHAIN_DEAD_END, 0.5)])
        return SingleValueDistribution(memory)

    def _get_transition_value(self, memory, action, next_state=None):
        return Value(cost=1)

    def _is_terminal(self, state) -> bool:
        return state == CHAIN_DEAD_END or self._is_goal(state)

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


class TestGPCI:
    def test_import(self):
        from skdecide.hub.solver.gpci import GPCI

        assert GPCI is not None

    def test_domain_check(self):
        from skdecide.hub.solver.gpci import GPCI

        dom = StochasticGridDomain()
        assert GPCI.check_domain(dom)

    def test_no_dead_ends_probability_one(self):
        """On a grid without dead-ends, P*(s0) should be 1.0."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: StochasticGridDomain(),
        ) as solver:
            solver.solve()
            p_s0 = solver.get_goal_probability(State(0, 0))

        assert abs(p_s0 - 1.0) < 0.01, f"P*(s0) = {p_s0}, expected 1.0"

    def test_no_dead_ends_matches_vi(self):
        """On a grid without dead-ends, C*(s0) should match VI's V*(s0)."""
        from skdecide.hub.solver.gpci import GPCI
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: StochasticGridDomain(),
            discount=1.0,
        ) as vi:
            vi.solve()
            v_vi = vi.get_utility(State(0, 0)).cost

        with GPCI(
            domain_factory=lambda: StochasticGridDomain(),
        ) as gpci:
            gpci.solve()
            c_gpci = gpci.get_goal_cost(State(0, 0))

        assert abs(c_gpci - v_vi) < 0.1, (
            f"GPCI C*(s0)={c_gpci} should match VI V*(s0)={v_vi}"
        )

    def test_avoidable_dead_end_grid(self):
        """On grid with avoidable dead-ends, P*(s0) should be < 1.0 but > 0."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: DeadEndGridDomain(),
        ) as solver:
            solver.solve()
            p_s0 = solver.get_goal_probability(State(0, 0))
            c_s0 = solver.get_goal_cost(State(0, 0))

        assert 0 < p_s0 < 1.0, f"P*(s0) = {p_s0}, expected 0 < P*(s0) < 1"
        assert c_s0 > 0, f"C*(s0) = {c_s0}, expected > 0"

    def test_unavoidable_dead_end_chain(self):
        """On chain with unavoidable dead-end, P*(s0) = 0.5."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: DeadEndChainDomain(),
        ) as solver:
            solver.solve()
            p_s0 = solver.get_goal_probability(State(0, 0))
            p_s1 = solver.get_goal_probability(State(1, 0))

        assert abs(p_s1 - 0.5) < 0.01, f"P*(s1) = {p_s1}, expected 0.5"
        assert abs(p_s0 - 0.5) < 0.01, f"P*(s0) = {p_s0}, expected 0.5"

    def test_dead_end_probability_zero(self):
        """P*(dead_end) should be 0.0."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: DeadEndChainDomain(),
        ) as solver:
            solver.solve()
            p_de = solver.get_goal_probability(CHAIN_DEAD_END)

        assert abs(p_de) < 0.001, f"P*(dead_end) = {p_de}, expected 0.0"

    def test_goal_state(self):
        """P*(goal) = 1.0 and C*(goal) = 0.0."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: DeadEndChainDomain(),
        ) as solver:
            solver.solve()
            p_goal = solver.get_goal_probability(State(2, 0))
            c_goal = solver.get_goal_cost(State(2, 0))

        assert abs(p_goal - 1.0) < 0.001, f"P*(goal) = {p_goal}, expected 1.0"
        assert abs(c_goal) < 0.001, f"C*(goal) = {c_goal}, expected 0.0"

    def test_rollout_stochastic_grid(self):
        """Rollout on stochastic grid should reach goal reliably."""
        from skdecide.hub.solver.gpci import GPCI

        dom = StochasticGridDomain()
        with GPCI(
            domain_factory=lambda: StochasticGridDomain(),
        ) as solver:
            solver.solve()
            reached_goal = 0
            for _ in range(20):
                _, _, final = rollout(dom, solver)
                if final == State(2, 2):
                    reached_goal += 1

        assert reached_goal > 15, f"Reached goal {reached_goal}/20 times, expected > 15"

    def test_rollout_dead_end_chain(self):
        """Rollout on dead-end chain should reach goal ~50% of the time."""
        from skdecide.hub.solver.gpci import GPCI

        dom = DeadEndChainDomain()
        n_trials = 200
        with GPCI(
            domain_factory=lambda: DeadEndChainDomain(),
        ) as solver:
            solver.solve()
            reached_goal = 0
            for _ in range(n_trials):
                _, _, final = rollout(dom, solver)
                if final == State(2, 0):
                    reached_goal += 1

        rate = reached_goal / n_trials
        assert 0.3 < rate < 0.7, f"Goal rate = {rate:.2f}, expected ~0.5 (±0.2)"

    def test_statistics(self):
        """Statistics should be reasonable."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: StochasticGridDomain(),
        ) as solver:
            solver.solve()
            assert solver.get_nb_of_explored_states() > 0
            assert solver.get_nb_prob_iterations() > 0
            assert solver.get_nb_cost_iterations() > 0
            assert solver.get_solving_time() >= 0
            assert len(solver.get_explored_states()) > 0

    def test_deterministic_grid_matches_vi(self):
        """GPCI on deterministic grid should match VI cost."""
        from skdecide.hub.solver.gpci import GPCI
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(),
            discount=1.0,
        ) as vi:
            vi.solve()
            v_vi = vi.get_utility(State(0, 0)).cost

        with GPCI(
            domain_factory=lambda: DeterministicGridDomain(),
        ) as gpci:
            gpci.solve()
            c_gpci = gpci.get_goal_cost(State(0, 0))
            p_gpci = gpci.get_goal_probability(State(0, 0))

        assert abs(p_gpci - 1.0) < 0.01
        assert abs(c_gpci - v_vi) < 0.1, (
            f"GPCI C*(s0)={c_gpci} should match VI V*(s0)={v_vi}"
        )

    def test_current_phase_in_callback(self):
        """Callback should observe PROBABILITY and COST phases."""
        from skdecide.hub.solver.gpci import GPCI, GPCIPhase

        observed_phases = set()

        def cb(solver):
            observed_phases.add(solver.get_current_phase())
            return False

        with GPCI(
            domain_factory=lambda: StochasticGridDomain(),
            callback=cb,
        ) as solver:
            solver.solve()

        assert GPCIPhase.PROBABILITY in observed_phases
        assert GPCIPhase.COST in observed_phases
        assert GPCIPhase.ENUMERATION not in observed_phases

    def test_policy(self):
        """get_policy should return a non-empty dict."""
        from skdecide.hub.solver.gpci import GPCI

        with GPCI(
            domain_factory=lambda: StochasticGridDomain(),
        ) as solver:
            solver.solve()
            policy = solver.get_policy()

        assert len(policy) > 0
        for state, (action, cost) in policy.items():
            assert isinstance(action, Action)
            assert isinstance(cost, Value)
