# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the LRTDP and LRTAstar C++ solvers.

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


def get_plan(domain, solver):
    """Execute a rollout using the solver's policy and return the plan and cost."""
    plan = []
    cost = 0
    observation = domain.reset()
    nb_steps = 0
    while (not domain.is_goal(observation)) and nb_steps < 20:
        plan.append(solver.sample_action(observation, domain=domain))
        outcome = domain.step(plan[-1])
        cost += outcome.value.cost
        observation = outcome.observation
        nb_steps += 1
    return plan, cost


class TestLRTDP:
    """Test LRTDP (Labeled RTDP) on a deterministic grid."""

    def test_terminal_value(self):
        """terminal_value parameter should be accepted without error."""
        from skdecide.hub.solver.lrtdp import LRTDP

        with LRTDP(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            terminal_value=lambda s: Value(cost=42.0),
            use_labels=True,
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            assert solver.get_utility(State(x=0, y=0, s=0)) is not None

    def test_explored_states(self):
        """get_explored_states should return a non-empty set including the initial state."""
        from skdecide.hub.solver.lrtdp import LRTDP

        with LRTDP(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            use_labels=True,
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            explored = solver.get_explored_states()

        assert len(explored) > 0
        assert State(x=0, y=0, s=0) in explored

    def test_solved_states(self):
        """get_solved_states should include the initial state after convergence."""
        from skdecide.hub.solver.lrtdp import LRTDP

        with LRTDP(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            use_labels=True,
            discount=1.0,
            epsilon=0.001,
        ) as solver:
            solver.solve()
            solved = solver.get_solved_states()

        assert State(x=0, y=0, s=0) in solved


class TestLRTAstar:
    """Test LRTAstar (LRTDP specialization for deterministic domains)."""

    def test_solves_deterministic(self):
        """LRTAstar should find a path on the deterministic grid.

        LRTAstar (RTDP without labels) is not guaranteed to find the optimal
        solution in bounded time, so we only check that a valid plan is found.
        """
        from skdecide.hub.solver.lrtdp import LRTAstar

        dom = GridDomain()

        with LRTAstar(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
        ) as solver:
            solver.solve()
            plan, cost = get_plan(dom, solver)

        assert len(plan) > 0
        assert cost > 0

    def test_get_plan(self):
        """LRTAstar.get_plan() should return a non-empty action sequence."""
        from skdecide.hub.solver.lrtdp import LRTAstar

        with LRTAstar(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
        ) as solver:
            solver.solve()
            plan = solver.get_plan()

        assert len(plan) > 0

    def test_domain_check(self):
        """LRTAstar should accept the deterministic GridDomain."""
        from skdecide.hub.solver.lrtdp import LRTAstar

        dom = GridDomain()
        assert LRTAstar.check_domain(dom)

    def test_get_last_trajectory(self):
        """get_last_trajectory() should return (state, action) pairs from the last trial."""
        from skdecide.hub.solver.lrtdp import LRTDP

        trajectories_seen = []

        def callback(solver, domain):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (state, action) tuple
            assert all(isinstance(sa, tuple) and len(sa) == 2 for sa in trajectory)
            # Store trajectory as tuple of (state, action) tuples for comparison
            traj_tuple = tuple((s, a) for s, a in trajectory)
            trajectories_seen.append(traj_tuple)
            # Stop after 3 iterations
            return len(trajectories_seen) >= 3

        with LRTDP(
            domain_factory=lambda: GridDomain(),
            heuristic=lambda d, s: Value(
                cost=sqrt((d.num_cols - 1 - s.x) ** 2 + (d.num_rows - 1 - s.y) ** 2)
            ),
            use_labels=True,
            max_depth=10,
            callback=callback,
        ) as solver:
            solver.solve()

        # Should have 3 trajectories (one per iteration before callback stopped it)
        assert len(trajectories_seen) == 3

        # All trajectories should be non-empty
        for traj in trajectories_seen:
            assert len(traj) > 0

        # All trajectories should start with the initial state
        for traj in trajectories_seen:
            state, action = traj[0]
            assert state == State(x=0, y=0, s=0)
            # Action should be one of the valid Action enum values
            assert action in [Action.up, Action.down, Action.left, Action.right]

        # Trajectories should change between iterations (LRTDP explores randomly)
        # At least some should be different
        unique_trajectories = set(trajectories_seen)
        assert len(unique_trajectories) > 1, "Trajectories should vary between trials"
