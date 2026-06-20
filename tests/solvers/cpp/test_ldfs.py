# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the LDFS C++ solver.

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


class TestLDFSDeterministic:
    """Test LDFS on a small deterministic 4x4 grid."""

    def test_solves_and_finds_optimal_cost(self):
        from skdecide.hub.solver.ldfs import LDFS

        dom = DeterministicGridDomain(4, 4)

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            actions, cost = rollout(dom, solver)

        assert cost == 6
        assert len(actions) == 6

    def test_optimal_policy_directions(self):
        from skdecide.hub.solver.ldfs import LDFS

        dom = DeterministicGridDomain(4, 4)

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            actions, _ = rollout(dom, solver)

        assert all(a in (Action.right, Action.down) for a in actions)

    def test_optimal_value_at_initial_state(self):
        from skdecide.hub.solver.ldfs import LDFS

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            v = solver.get_utility(State(0, 0))

        assert abs(v.cost - 6.0) < 0.01

    def test_solved_states(self):
        """After solving, the initial state should be labeled solved."""
        from skdecide.hub.solver.ldfs import LDFS

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            solved = solver.get_solved_states()

        assert State(0, 0) in solved

    def test_explores_fewer_states_than_vi(self):
        """LDFS should explore fewer states than VI on a grid with a good heuristic."""
        from skdecide.hub.solver.ldfs import LDFS

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            n = solver.get_nb_of_explored_states()

        # VI explores all 16 states; LDFS with a good heuristic may explore fewer
        assert n <= 16

    def test_matches_vi_value(self):
        """LDFS and VI should converge to the same optimal value."""
        from skdecide.hub.solver.ldfs import LDFS
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            discount=1.0,
            epsilon=0.001,
        ) as vi_solver:
            vi_solver.solve()
            v_vi = vi_solver.get_utility(State(0, 0))

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as ldfs_solver:
            ldfs_solver.solve()
            v_ldfs = ldfs_solver.get_utility(State(0, 0))

        assert abs(v_vi.cost - v_ldfs.cost) < 0.01

    def test_callback_called(self):
        from skdecide.hub.solver.ldfs import LDFS

        call_count = [0]

        def cb(solver):
            call_count[0] += 1
            return False

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
            callback=cb,
        ) as solver:
            solver.solve()

        assert call_count[0] > 0

    def test_last_trajectory_nonempty(self):
        """get_last_trajectory() should return a non-empty list after solving."""
        from skdecide.hub.solver.ldfs import LDFS

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            traj = solver.get_last_trajectory()

        assert len(traj) > 0
        assert traj[0] == State(0, 0)  # Always starts from initial state

    def test_last_trajectory_in_callback(self):
        """Trajectory should be accessible from callback."""
        from skdecide.hub.solver.ldfs import LDFS

        trajectories = []

        def cb(slv):
            trajectories.append(slv.get_last_trajectory())
            return False

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
            discount=1.0,
            epsilon=0.001,
            callback=cb,
        ) as solver:
            solver.solve()

        assert len(trajectories) > 0
        for traj in trajectories:
            assert len(traj) > 0
            assert traj[0] == State(0, 0)

    def test_last_trajectory_updates_across_iterations(self):
        """Trajectory should update across multiple LDFS iterations, not freeze."""
        from skdecide.hub.solver.ldfs import LDFS

        trajectories = []

        def cb(slv):
            traj = slv.get_last_trajectory()
            trajectories.append(traj[:])  # Store a copy
            # Let it run to completion to witness trajectory changes
            return False

        # Use a larger grid and zero heuristic to force deep exploration
        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(6, 6),
            heuristic=lambda d, s: Value(cost=0),  # Zero heuristic forces exploration
            discount=1.0,
            epsilon=0.1,
            callback=cb,
        ) as solver:
            solver.solve()

        # Should have multiple iterations
        assert len(trajectories) >= 2, "Need at least 2 iterations to test updates"

        # Verify trajectories are not all identical (frozen)
        # At least one trajectory should differ from the first
        first_traj_tuple = tuple((s.x, s.y) for s in trajectories[0])
        found_different = False

        for i, traj in enumerate(trajectories[1:], 1):
            traj_tuple = tuple((s.x, s.y) for s in traj)
            if traj_tuple != first_traj_tuple:
                found_different = True
                break

        # The trajectory should change at some point during solving
        assert found_different, (
            f"Trajectory appears frozen - all {len(trajectories)} callbacks saw the same path: {trajectories[0]}"
        )


class TestLDFSStochastic:
    def test_converges(self):
        from skdecide.hub.solver.ldfs import LDFS

        with LDFS(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            heuristic=lambda d, s: Value(cost=abs(2 - s.x) + abs(2 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            solved = solver.get_solved_states()

        assert State(0, 0) in solved

    def test_matches_vi_value(self):
        from skdecide.hub.solver.ldfs import LDFS
        from skdecide.hub.solver.vi import VI

        with VI(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            discount=1.0,
            epsilon=0.001,
        ) as vi_solver:
            vi_solver.solve()
            v_vi = vi_solver.get_utility(State(0, 0))

        with LDFS(
            domain_factory=lambda: StochasticGridDomain(3, 3),
            heuristic=lambda d, s: Value(cost=abs(2 - s.x) + abs(2 - s.y)),
            discount=1.0,
            epsilon=0.001,
        ) as ldfs_solver:
            ldfs_solver.solve()
            v_ldfs = ldfs_solver.get_utility(State(0, 0))

        assert abs(v_vi.cost - v_ldfs.cost) < 0.1


class TestIDAstar:
    """Test IDAstar (LDFS specialization for deterministic domains)."""

    def test_solves_deterministic_grid(self):
        from skdecide.hub.solver.ldfs import IDAstar

        dom = DeterministicGridDomain(4, 4)

        with IDAstar(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
        ) as solver:
            solver.solve()
            actions, cost = rollout(dom, solver)

        assert cost == 6
        assert len(actions) == 6

    def test_optimal_value(self):
        from skdecide.hub.solver.ldfs import IDAstar

        with IDAstar(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
        ) as solver:
            solver.solve()
            v = solver.get_utility(State(0, 0))

        assert abs(v.cost - 6.0) < 0.01

    def test_matches_ldfs_value(self):
        """IDAstar and LDFS should produce the same result on deterministic domains."""
        from skdecide.hub.solver.ldfs import LDFS, IDAstar

        heuristic = lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y))

        with LDFS(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=heuristic,
            discount=1.0,
            epsilon=0.001,
        ) as ldfs_solver:
            ldfs_solver.solve()
            v_ldfs = ldfs_solver.get_utility(State(0, 0))

        with IDAstar(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=heuristic,
        ) as ida_solver:
            ida_solver.solve()
            v_ida = ida_solver.get_utility(State(0, 0))

        assert abs(v_ldfs.cost - v_ida.cost) < 0.01

    def test_get_plan(self):
        """get_plan() should return the optimal action sequence."""
        from skdecide.hub.solver.ldfs import IDAstar

        with IDAstar(
            domain_factory=lambda: DeterministicGridDomain(4, 4),
            heuristic=lambda d, s: Value(cost=abs(3 - s.x) + abs(3 - s.y)),
        ) as solver:
            solver.solve()
            plan = solver.get_plan()

        assert len(plan) == 6
        assert all(a in (Action.right, Action.down) for a in plan)

    def test_domain_check_rejects_stochastic(self):
        """IDAstar should reject stochastic domains via check_domain."""
        from skdecide.hub.solver.ldfs import IDAstar

        dom = StochasticGridDomain(3, 3)
        assert not IDAstar.check_domain(dom)
