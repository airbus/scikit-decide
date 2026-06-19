# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Policy Iteration C++ solver.

Uses the same grid domains as test_vi.py:
- 4x4 deterministic grid: optimal path 6 steps, cost 6, reward -6
- 3x3 stochastic grid: 80/10/10 slip probabilities
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from skdecide import (
    DiscreteDistribution,
    Domain,
    SingleValueDistribution,
    Value,
)
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    EnumerableTransitions,
    FullyObservable,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace

# --- Domain definitions (shared with test_vi.py) ---


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
        return state.x == self.num_cols - 1 and state.y == self.num_rows - 1

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
        return state.x == self.num_cols - 1 and state.y == self.num_rows - 1

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


# --- Helpers ---


def rollout(domain, solver, max_steps=100):
    actions = []
    cost = 0.0
    obs = domain.reset()
    for _ in range(max_steps):
        if domain.is_terminal(obs):
            break
        action = solver.sample_action(obs, domain=domain)
        actions.append(action)
        outcome = domain.step(action)
        cost += outcome.value.cost
        obs = outcome.observation
    return actions, cost


# --- Tests ---


class TestPIDeterministic:
    """Test PI on a small deterministic 4x4 grid.

    Optimal path from (0,0) to (3,3) is 6 steps, cost 6, reward -6.
    """

    def test_solves_and_finds_optimal_cost(self):
        from skdecide.hub.solver.pi import PI

        dom = DeterministicGridDomain(4, 4)

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            actions, cost = rollout(dom, solver)

        assert cost == 6
        assert len(actions) == 6

    def test_optimal_policy_directions(self):
        from skdecide.hub.solver.pi import PI

        dom = DeterministicGridDomain(4, 4)

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            actions, _ = rollout(dom, solver)

        assert all(a in (Action.right, Action.down) for a in actions)

    def test_optimal_value_at_initial_state(self):
        from skdecide.hub.solver.pi import PI

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            v = solver.get_utility(State(0, 0))

        assert abs(v.reward - (-6.0)) < 0.01

    def test_explored_states_count(self):
        from skdecide.hub.solver.pi import PI

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            explored = solver.get_explored_states()

        assert len(explored) == 16

    def test_policy_completeness(self):
        from skdecide.hub.solver.pi import PI

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            policy = solver.get_policy()

        non_terminal = [
            State(x, y) for x in range(4) for y in range(4) if (x, y) != (3, 3)
        ]
        for s in non_terminal:
            assert s in policy, f"State {s} missing from policy"

    def test_few_iterations(self):
        """PI should converge in very few iterations on a small grid."""
        from skdecide.hub.solver.pi import PI

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            n = solver.get_nb_iterations()

        assert n <= 10

    def test_initial_policy_warm_start(self):
        """Providing a good initial policy should converge in fewer iterations."""
        from skdecide.hub.solver.pi import PI

        def good_policy(d, s):
            if s.x < d.num_cols - 1:
                return Action.right
            return Action.down

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            initial_policy=good_policy,
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            actions, cost = rollout(DeterministicGridDomain(4, 4), solver)

        assert cost == 6
        assert len(actions) == 6

    def test_policy_changed_states_empty_after_convergence(self):
        from skdecide.hub.solver.pi import PI

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            changed = solver.get_policy_changed_states()

        assert len(changed) == 0

    def test_callback_called(self):
        from skdecide.hub.solver.pi import PI

        iteration_counts = []

        def cb(solver):
            iteration_counts.append(solver.get_nb_iterations())
            return False

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
            callback=cb,
        ) as solver:
            solver.solve()

        assert len(iteration_counts) > 0

    def test_matches_vi_value(self):
        """PI and VI should converge to the same optimal value."""
        from skdecide.hub.solver.pi import PI
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as vi_solver:
            vi_solver.solve()
            v_vi = vi_solver.get_utility(State(0, 0))

        with PI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as pi_solver:
            pi_solver.solve()
            v_pi = pi_solver.get_utility(State(0, 0))

        assert abs(v_vi.reward - v_pi.reward) < 0.01


class TestPIStochastic:
    def test_converges(self):
        from skdecide.hub.solver.pi import PI

        with PI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as solver:
            solver.solve()
            explored = solver.get_explored_states()

        assert len(explored) == 9

    def test_matches_vi_value(self):
        """PI and VI should agree on the stochastic grid."""
        from skdecide.hub.solver.pi import PI
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
        ) as vi_solver:
            vi_solver.solve()
            v_vi = vi_solver.get_utility(State(0, 0))

        with PI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
            max_eval_sweeps=100,
        ) as pi_solver:
            pi_solver.solve()
            v_pi = pi_solver.get_utility(State(0, 0))

        assert abs(v_vi.reward - v_pi.reward) < 0.1
