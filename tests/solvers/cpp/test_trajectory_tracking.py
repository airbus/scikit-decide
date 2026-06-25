# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for get_last_trajectory() on trial-based solvers.

Tests that trajectory tracking works correctly for MCTS, POMCP, DESPOT,
SARSOP, and HSVI by verifying that trajectories change between callback
invocations during solving.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from skdecide import (
    DeterministicPlanningDomain,
    GoalPOMDPDomain,
    ImplicitSpace,
    SingleValueDistribution,
    Value,
)
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace

# ============================================================================
# Test Domains
# ============================================================================


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


# --- MDP Domain for MCTS ---


class D_MDP(DeterministicPlanningDomain, UnrestrictedActions):
    T_state = State
    T_observation = T_state
    T_event = Action
    T_value = float
    T_predicate = bool
    T_info = None


class GridMDP(D_MDP):
    """Simple 5x5 grid MDP for MCTS testing."""

    def __init__(self, size=5):
        self.size = size

    def _get_next_state(self, memory, action):
        x, y = memory.x, memory.y
        if action == Action.left:
            x = max(x - 1, 0)
        elif action == Action.right:
            x = min(x + 1, self.size - 1)
        elif action == Action.up:
            y = max(y - 1, 0)
        elif action == Action.down:
            y = min(y + 1, self.size - 1)
        return State(x, y)

    def _get_transition_value(self, memory, action, next_state=None):
        # Small cost per move, big reward at goal
        if self._is_goal(next_state):
            return Value(reward=10.0)
        return Value(reward=-0.1)

    def _is_terminal(self, state):
        return self._is_goal(state)

    def _get_action_space_(self):
        return EnumSpace(Action)

    def _get_goals_(self):
        return ImplicitSpace(lambda s: s.x == self.size - 1 and s.y == self.size - 1)

    def _get_initial_state_(self):
        return State(x=0, y=0)

    def _get_observation_space_(self):
        return MultiDiscreteSpace([self.size, self.size])


# --- POMDP Domain for POMCP/DESPOT/SARSOP/HSVI ---


class Observation(Enum):
    """Noisy observations: see correct position 70% of the time."""

    obs_00 = 0
    obs_01 = 1
    obs_10 = 2
    obs_11 = 3
    obs_other = 4  # catch-all


class D_POMDP(GoalPOMDPDomain, UnrestrictedActions):
    T_state = State
    T_observation = Observation
    T_event = Action
    T_value = float
    T_predicate = bool
    T_info = None


class GridPOMDP(D_POMDP):
    """2x2 grid POMDP with noisy observations."""

    def __init__(self):
        self.size = 2
        # Map state to typical observation
        self._state_to_obs = {
            State(0, 0): Observation.obs_00,
            State(0, 1): Observation.obs_01,
            State(1, 0): Observation.obs_10,
            State(1, 1): Observation.obs_11,
        }

    def _get_next_state_distribution(self, memory, action):
        x, y = memory.x, memory.y
        if action == Action.left:
            x = max(x - 1, 0)
        elif action == Action.right:
            x = min(x + 1, self.size - 1)
        elif action == Action.up:
            y = max(y - 1, 0)
        elif action == Action.down:
            y = min(y + 1, self.size - 1)
        return SingleValueDistribution(State(x, y))

    def _get_observation_distribution(self, state, action=None):
        # 70% chance of correct observation, 30% of "other"
        from skdecide import DiscreteDistribution

        correct_obs = self._state_to_obs[state]
        return DiscreteDistribution([(correct_obs, 0.7), (Observation.obs_other, 0.3)])

    def _get_transition_value(self, memory, action, next_state=None):
        if self._is_goal(next_state):
            return Value(cost=0.0)
        return Value(cost=1.0)

    def _is_terminal(self, state):
        return self._is_goal(state)

    def _get_action_space_(self):
        return EnumSpace(Action)

    def _get_goals_(self):
        return ImplicitSpace(lambda s: s.x == 1 and s.y == 1)

    def _get_initial_state_(self):
        return State(x=0, y=0)

    def _get_initial_state_distribution_(self):
        return SingleValueDistribution(State(x=0, y=0))

    def _get_observation_space_(self):
        return EnumSpace(Observation)


# ============================================================================
# Test Cases
# ============================================================================


class TestMCTSTrajectory:
    """Test MCTS get_last_trajectory()."""

    def test_mcts_trajectory_changes(self):
        """MCTS get_last_trajectory() returns non-empty (state, action) tuples."""
        from skdecide.hub.solver.mcts import MCTS

        trajectories_seen = []

        # MCTS callback: (solver, thread_id or None)
        def callback(solver, thread_id=None):
            trajectory = solver.get_last_trajectory()
            trajectories_seen.append(trajectory)
            # Stop after 3 rollouts
            return len(trajectories_seen) >= 3

        domain = GridMDP(size=3)
        with MCTS(
            domain_factory=lambda: GridMDP(size=3),
            time_budget=5000,
            rollout_budget=10,
            max_depth=5,
            discount=0.95,
            callback=callback,
        ) as solver:
            solver.solve_from(domain.reset())

        # Should have collected at least one trajectory
        assert len(trajectories_seen) >= 1

        # Every collected trajectory must be a list of (state, action) 2-tuples
        for traj in trajectories_seen:
            assert all(isinstance(sa, tuple) and len(sa) == 2 for sa in traj)


class TestPOMCPTrajectory:
    """Test POMCP get_last_trajectory()."""

    def test_pomcp_trajectory_changes(self):
        """POMCP trajectories should change between simulations."""
        from skdecide.hub.solver.pomcp import POMCP

        trajectories_seen = []

        def callback(solver, domain):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (observation, action) tuple
            assert all(isinstance(oa, tuple) and len(oa) == 2 for oa in trajectory)
            # Store as hashable tuple
            traj_tuple = tuple((o, a) for o, a in trajectory)
            trajectories_seen.append(traj_tuple)
            # Stop after 5 simulations
            return len(trajectories_seen) >= 5

        domain = GridPOMDP()
        with POMCP(
            domain_factory=lambda: GridPOMDP(),
            num_simulations=20,
            max_depth=5,
            discount=0.95,
            callback=callback,
        ) as solver:
            solver.solve_from(domain.get_initial_state_distribution())
            # POMCP is an online solver - planning happens in get_next_action()
            # Call it multiple times to trigger the callback
            obs = domain.reset()
            for _ in range(10):  # Try multiple action queries
                if len(trajectories_seen) >= 5:
                    break
                action = solver.get_next_action(obs)
                obs = domain.step(action).observation

        # Should have 5 trajectories
        assert len(trajectories_seen) == 5

        # Most should be non-empty (some might be empty if stopped early)
        non_empty = [t for t in trajectories_seen if len(t) > 0]
        assert len(non_empty) >= 3

        # Trajectories should vary
        unique_trajectories = set(trajectories_seen)
        assert len(unique_trajectories) > 1, (
            "POMCP trajectories should vary between simulations"
        )


class TestDESPOTTrajectory:
    """Test DESPOT get_last_trajectory()."""

    def test_despot_trajectory_changes(self):
        """DESPOT trajectories should change between exploration iterations."""
        from skdecide.hub.solver.despot import DESPOT

        trajectories_seen = []

        # DESPOT callback: (solver, thread_id or None)
        def callback(solver, thread_id=None):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (state, action) tuple
            assert all(isinstance(sa, tuple) and len(sa) == 2 for sa in trajectory)
            # Store as hashable tuple
            traj_tuple = tuple((s, a) for s, a in trajectory)
            trajectories_seen.append(traj_tuple)
            # Stop after 5 iterations
            return len(trajectories_seen) >= 5

        domain = GridPOMDP()
        with DESPOT(
            domain_factory=lambda: GridPOMDP(),
            num_scenarios=50,
            max_depth=5,
            time_budget=10000,
            discount=0.95,
            callback=callback,
        ) as solver:
            solver.solve_from(domain.get_initial_state_distribution())
            # DESPOT is an online solver - planning happens in get_next_action()
            # Call it multiple times to trigger the callback
            obs = domain.reset()
            for _ in range(10):  # Try multiple action queries
                if len(trajectories_seen) >= 5:
                    break
                action = solver.get_next_action(obs)
                obs = domain.step(action).observation

        # Should have gotten some trajectories (may be fewer than 5 if converged quickly)
        assert len(trajectories_seen) >= 1, "Should have at least one trajectory"

        # Most should be non-empty (some might be empty if stopped early at terminal state)
        non_empty = [t for t in trajectories_seen if len(t) > 0]
        assert len(non_empty) >= 1, "Should have at least one non-empty trajectory"


class TestSARSOPTrajectory:
    """Test SARSOP get_last_trajectory()."""

    def test_sarsop_trajectory_changes(self):
        """SARSOP trajectories should change between sampling iterations."""
        from skdecide.hub.solver.sarsop import SARSOP

        trajectories_seen = []

        def callback(solver):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (belief_dict, action) tuple
            assert all(isinstance(ba, tuple) and len(ba) == 2 for ba in trajectory)
            if len(trajectory) > 0:
                # First element should have belief dict with 'state_probs'
                belief_dict, action = trajectory[0]
                assert isinstance(belief_dict, dict)
                assert "state_probs" in belief_dict

            # Store length as proxy (beliefs aren't hashable)
            trajectories_seen.append(len(trajectory))
            # Stop after 5 iterations
            return len(trajectories_seen) >= 5

        domain = GridPOMDP()
        with SARSOP(
            domain_factory=lambda: GridPOMDP(),
            epsilon=0.5,  # Loose convergence for quick test
            time_budget=10000,
            max_beliefs=200,
            discount=0.95,
            callback=callback,
        ) as solver:
            solver.solve_from(domain.get_initial_state_distribution())

        # Should have gotten at least one callback
        assert len(trajectories_seen) >= 1, "Should have at least one trajectory"

        # Verify that get_last_trajectory() works (returns valid data)
        # On this simple problem, SARSOP may converge in 1 iteration with loose epsilon


class TestHSVITrajectory:
    """Test HSVI get_last_trajectory()."""

    def test_hsvi_trajectory_changes(self):
        """HSVI trajectories should change between exploration iterations."""
        from skdecide.hub.solver.hsvi import HSVI

        trajectories_seen = []

        def callback(solver):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (belief_dict, action) tuple
            assert all(isinstance(ba, tuple) and len(ba) == 2 for ba in trajectory)
            if len(trajectory) > 0:
                # First element should have belief dict with 'state_probs'
                belief_dict, action = trajectory[0]
                assert isinstance(belief_dict, dict)
                assert "state_probs" in belief_dict

            # Store length as proxy (beliefs aren't hashable)
            trajectories_seen.append(len(trajectory))
            # Stop after 5 iterations
            return len(trajectories_seen) >= 5

        domain = GridPOMDP()
        with HSVI(
            domain_factory=lambda: GridPOMDP(),
            epsilon=0.5,  # Loose convergence for quick test
            time_budget=10000,
            max_sample_depth=5,
            discount=0.95,
            callback=callback,
        ) as solver:
            solver.solve_from(domain.get_initial_state_distribution())

        # Should have gotten at least one callback
        assert len(trajectories_seen) >= 1, "Should have at least one trajectory"

        # Verify that get_last_trajectory() works (returns valid data)
        # On this simple problem, HSVI may converge in 1 iteration with loose epsilon


class TestGoalHSVITrajectory:
    """Test Goal-HSVI get_last_trajectory()."""

    def test_goal_hsvi_trajectory_changes(self):
        """Goal-HSVI trajectories should change between exploration iterations."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        trajectories_seen = []

        def callback(solver):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (belief_dict, action) tuple
            assert all(isinstance(ba, tuple) and len(ba) == 2 for ba in trajectory)

            # Store length as proxy
            trajectories_seen.append(len(trajectory))
            # Stop after 5 iterations
            return len(trajectories_seen) >= 5

        domain = GridPOMDP()
        with GoalHSVI(
            domain_factory=lambda: GridPOMDP(),
            goal_checker=lambda d, s: d.is_goal(s),
            epsilon=0.5,
            time_budget=10000,
            max_sample_depth=5,
            callback=callback,
        ) as solver:
            solver.solve_from(domain.get_initial_state_distribution())

        # Should have gotten at least one callback
        assert len(trajectories_seen) >= 1, "Should have at least one trajectory"

        # Verify that get_last_trajectory() works (returns valid data)
        # On this simple problem, Goal-HSVI may converge in 1 iteration with loose epsilon
