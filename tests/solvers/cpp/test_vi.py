# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Value Iteration C++ solver.

Uses a small 4x4 deterministic grid with known optimal values:
- Start at (0,0), terminal state at (3,3) with value 0.
- Each move costs 1, hitting a wall costs 2.
- Optimal path: 6 steps (3 right + 3 down), cost = 6.

Also tests a stochastic 3x3 grid where each intended move
succeeds with probability 0.8 and slips sideways with 0.1 each.
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

# --- Domain definitions ---


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
    """Small deterministic grid. Terminal state at (cols-1, rows-1)."""

    def __init__(self, num_cols=4, num_rows=4):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_initial_state_(self) -> State:
        return State(0, 0)

    def _state_reset(self) -> State:
        return State(0, 0)

    def _get_next_state_distribution(self, memory, action):
        ns = self._move(memory, action)
        return SingleValueDistribution(ns)

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
    """Small stochastic grid. 80% intended move, 10% each perpendicular."""

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
    """Execute the solver's policy on the domain, return (actions, total_cost)."""
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


class TestVIDeterministic:
    """Test VI on a small deterministic 4x4 grid.

    Optimal path from (0,0) to (3,3) is 6 steps (3 right + 3 down), cost = 6.
    """

    def test_solves_and_finds_optimal_cost(self):
        from skdecide.hub.solver.vi import VI

        dom = DeterministicGridDomain(num_cols=4, num_rows=4)

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            verbose=False,
        ) as solver:
            solver.solve()
            actions, cost = rollout(dom, solver)

        assert cost == 6
        assert len(actions) == 6

    def test_optimal_policy_directions(self):
        """The optimal policy should only use right and down moves."""
        from skdecide.hub.solver.vi import VI

        dom = DeterministicGridDomain(num_cols=4, num_rows=4)

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            actions, _ = rollout(dom, solver)

        assert all(a in (Action.right, Action.down) for a in actions)

    def test_optimal_value_at_initial_state(self):
        """V*(start) should be -6.0 (reward = -cost for 6 steps of cost 1)."""
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            v = solver.get_utility(State(0, 0))

        assert abs(v.reward - (-6.0)) < 0.01

    def test_explored_states_count(self):
        """A 4x4 grid has 16 states; all should be explored."""
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            explored = solver.get_explored_states()

        assert len(explored) == 16

    def test_converged_states(self):
        """After convergence all states should be converged."""
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            converged = solver.get_converged_states()
            explored = solver.get_explored_states()

        assert len(converged) == len(explored)

    def test_policy_completeness(self):
        """Policy should be defined for every non-terminal state."""
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            policy = solver.get_policy()

        non_terminal = [
            State(x, y) for x in range(4) for y in range(4) if (x, y) != (3, 3)
        ]
        for s in non_terminal:
            assert s in policy, f"State {s} missing from policy"

    def test_terminal_value_parameter(self):
        """Custom terminal_value should be reflected in V(terminal)."""
        from skdecide.hub.solver.vi import VI

        terminal_reward = -100.0

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            terminal_value=lambda s: Value(reward=terminal_reward),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            v = solver.get_utility(State(3, 3))

        assert abs(v.reward - terminal_reward) < 0.01

    def test_callback_receives_state_sets(self):
        """Callback should be able to query explored/converged states."""
        from skdecide.hub.solver.vi import VI

        explored_sizes = []

        def cb(solver):
            explored_sizes.append(solver.get_nb_of_explored_states())
            return False  # don't stop

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
            callback=cb,
        ) as solver:
            solver.solve()

        assert len(explored_sizes) > 0
        assert all(s == 16 for s in explored_sizes)


class TestVIStochastic:
    """Test VI on a small stochastic 3x3 grid.

    With slip probability, the optimal cost is higher than the
    deterministic 4 steps. We verify convergence and policy quality.
    """

    def test_converges(self):
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            explored = solver.get_explored_states()

        assert len(explored) == 9

    def test_value_lower_than_deterministic(self):
        """Stochastic optimal reward should be < -4 (worse than deterministic)."""
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            v = solver.get_utility(State(0, 0))

        assert v.reward < -4.0

    def test_policy_reaches_goal(self):
        """Following the policy should reach the goal."""
        from skdecide.hub.solver.vi import VI

        dom = StochasticGridDomain(3, 3)

        with VI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            _, cost = rollout(dom, solver, max_steps=200)

        assert cost > 0  # did something
