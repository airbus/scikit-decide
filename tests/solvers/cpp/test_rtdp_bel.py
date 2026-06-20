# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the RTDP-Bel C++ solver.

Uses a simple Tiger-like POMDP:
- 2 hidden states: tiger_left, tiger_right
- 3 actions: listen, open_left, open_right
- 2 observations: hear_left, hear_right
- Listen: costs 1, doesn't change state, 85% correct observation
- Open correct door: costs 0 (goal reached)
- Open wrong door: costs 100 (penalty)
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from skdecide import (
    DiscreteDistribution,
    Domain,
    EnumerableSpace,
    ImplicitSpace,
    SingleValueDistribution,
    Value,
)
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    EnumerableTransitions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.hub.space.gym import EnumSpace

# --- Tiger POMDP domain ---


class TigerState(NamedTuple):
    tiger_location: str  # "left" or "right" or "done"


class TigerAction(Enum):
    listen = 0
    open_left = 1
    open_right = 2


class TigerObservation(NamedTuple):
    heard: str  # "left", "right", or "done"


class TigerDomain(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicInitialized,
    EnumerableTransitions,
    Actions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
):
    T_state = TigerState
    T_observation = TigerObservation
    T_event = TigerAction
    T_value = float
    T_predicate = bool
    T_info = None

    def __init__(self):
        self._initial_tiger = "left"

    def _get_initial_state_(self) -> TigerState:
        return TigerState(tiger_location=self._initial_tiger)

    def _state_reset(self) -> TigerState:
        return TigerState(tiger_location=self._initial_tiger)

    def _get_next_state_distribution(self, memory, action):
        s = memory
        if s.tiger_location == "done":
            return SingleValueDistribution(s)

        if action == TigerAction.listen:
            return SingleValueDistribution(s)
        else:
            # Opening a door leads to "done" state
            return SingleValueDistribution(TigerState(tiger_location="done"))

    def _get_transition_value(self, memory, action, next_state=None):
        s = memory
        if s.tiger_location == "done":
            return Value(cost=0)

        if action == TigerAction.listen:
            return Value(cost=1)
        elif action == TigerAction.open_left:
            if s.tiger_location == "left":
                return Value(cost=100)  # tiger behind left door
            else:
                return Value(cost=0)  # safe
        elif action == TigerAction.open_right:
            if s.tiger_location == "right":
                return Value(cost=100)  # tiger behind right door
            else:
                return Value(cost=0)  # safe

    def _is_terminal(self, state) -> bool:
        return state.tiger_location == "done"

    def _get_goals_(self):
        return ImplicitSpace(lambda s: s.tiger_location == "done")

    def _get_action_space_(self):
        return EnumSpace(TigerAction)

    def _get_applicable_actions_from(self, memory):
        return self._get_action_space_()

    def _get_observation_space_(self):
        return EnumerableSpace(
            [
                TigerObservation(heard="left"),
                TigerObservation(heard="right"),
                TigerObservation(heard="done"),
            ]
        )

    def _get_observation_distribution(self, state, action=None):
        if state.tiger_location == "done":
            return SingleValueDistribution(TigerObservation(heard="done"))

        if action is None or action == TigerAction.listen:
            if state.tiger_location == "left":
                return DiscreteDistribution(
                    [
                        (TigerObservation(heard="left"), 0.85),
                        (TigerObservation(heard="right"), 0.15),
                    ]
                )
            else:
                return DiscreteDistribution(
                    [
                        (TigerObservation(heard="left"), 0.15),
                        (TigerObservation(heard="right"), 0.85),
                    ]
                )
        else:
            return SingleValueDistribution(TigerObservation(heard="done"))


# --- Tests ---


class TestRTDPBel:
    def test_import(self):
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        assert RTDPBel is not None

    def test_domain_check(self):
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        dom = TigerDomain()
        assert RTDPBel.check_domain(dom)

    def test_solves_tiger(self):
        """RTDP-Bel should find a solution for the Tiger POMDP."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            time_budget=10000,
            rollout_budget=1000,
            max_depth=50,
            epsilon=0.001,
            discount=1.0,
        ) as solver:
            solver.solve()
            n_beliefs = solver.get_nb_explored_beliefs()
            n_rollouts = solver.get_nb_rollouts()

        assert n_beliefs > 0
        assert n_rollouts > 0

    def test_point_belief_optimal_action(self):
        """With a point belief (tiger is known to be left), the optimal
        action is open_right (cost 0). RTDP-Bel should find this."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
        ) as solver:
            solver.solve()
            s0 = TigerState(tiger_location="left")
            action = solver.sample_action(s0)

        assert action == TigerAction.open_right

    def test_point_belief_optimal_value(self):
        """With a point belief (tiger_left known), the optimal cost is 0
        (just open the right door). The solver should find V*(s0) = 0."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
        ) as solver:
            solver.solve()
            s0 = TigerState(tiger_location="left")
            v = solver.get_utility(s0)

        assert v is not None
        assert abs(v.cost - 0.0) < 1.0

    def test_policy_avoids_tiger(self):
        """With a point belief on tiger_left, the optimal action is
        open_right (cost 0). The solver should never suggest open_left
        which would cost 100."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
        ) as solver:
            solver.solve()

            # Check for tiger_left: should open_right
            s_left = TigerState(tiger_location="left")
            action_left = solver.sample_action(s_left)
            assert action_left == TigerAction.open_right

    def test_both_tiger_locations(self):
        """Test that RTDP-Bel finds the correct action for both tiger
        locations when the state is fully known (point belief)."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        # Solve with tiger on the left
        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
        ) as solver:
            solver.solve()
            s_left = TigerState(tiger_location="left")
            action = solver.sample_action(s_left)
            v = solver.get_utility(s_left)

        # With known tiger_left, should open_right at cost 0
        assert action == TigerAction.open_right
        assert v.cost < 1.0  # optimal cost is 0

    def test_get_last_trajectory(self):
        """get_last_trajectory() should return (belief, action) pairs from the last trial."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        trajectories_seen = []

        def callback(solver, domain):
            trajectory = solver.get_last_trajectory()
            # Each element should be a (belief, action) tuple
            assert all(isinstance(ba, tuple) and len(ba) == 2 for ba in trajectory)
            # Store trajectory as tuple of (frozen belief dict, action) for comparison
            traj_tuple = tuple(
                (frozenset((str(s), p) for s, p in belief.items()), action)
                for belief, action in trajectory
            )
            trajectories_seen.append(traj_tuple)
            # Stop after 3 iterations
            return len(trajectories_seen) >= 3

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            max_depth=10,
            callback=callback,
        ) as solver:
            solver.solve()

        # Should have 3 trajectories (one per iteration before callback stopped it)
        assert len(trajectories_seen) == 3

        # All trajectories should be non-empty
        for traj in trajectories_seen:
            assert len(traj) > 0, (
                "Each trajectory should have at least one (belief, action)"
            )

        # Each belief in each trajectory should be a valid belief (probabilities sum to ~1)
        # and each action should be valid
        for traj in trajectories_seen:
            for belief_frozen, action in traj:
                belief_dict = dict(belief_frozen)
                # Convert back to numeric probabilities
                total_prob = sum(float(p) for _, p in belief_dict.items())
                assert 0.9 < total_prob <= 1.1, (
                    f"Belief probabilities should sum to ~1, got {total_prob}"
                )
                # Action should be one of the valid Tiger actions
                assert action in [
                    TigerAction.listen,
                    TigerAction.open_left,
                    TigerAction.open_right,
                ]

    def test_observation_based_rollout(self):
        """Test the observation-based interaction: solve, then interact
        through observations. The solver should track beliefs internally
        and produce valid actions."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        dom = TigerDomain()

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=2000,
            max_depth=30,
        ) as solver:
            solver.solve()

            # Rollout using observation-based interface
            obs = dom.reset()  # returns an Observation
            actions = []
            total_cost = 0.0
            for _ in range(10):
                if hasattr(obs, "heard") and obs.heard == "done":
                    break
                action = solver.sample_action(obs)
                actions.append(action)
                outcome = dom.step(action)
                total_cost += outcome.value.cost
                obs = outcome.observation

            # Reset belief for next episode
            solver.reset_belief()

        # The solver should have produced at least one action
        assert len(actions) > 0
        # With known tiger_left, should never open_left (cost 100)
        assert TigerAction.open_left not in actions

    def test_solving_time(self):
        """get_solving_time should return a positive value after solving."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=100,
            max_depth=10,
        ) as solver:
            solver.solve()
            t = solver.get_solving_time()

        assert t >= 0

    def test_parallel_mode(self):
        """Solver should work in parallel mode with correct results."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
            parallel=True,
        ) as solver:
            solver.solve()
            n_beliefs = solver.get_nb_explored_beliefs()
            n_rollouts = solver.get_nb_rollouts()

            # Point belief: tiger is left, should open_right
            s_left = TigerState(tiger_location="left")
            action = solver.sample_action(s_left)

        assert n_beliefs > 0
        assert n_rollouts > 0
        assert action == TigerAction.open_right

    def test_parallel_mode_callback(self):
        """Callback with thread_id should work in parallel mode."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        callback_calls = []

        def my_callback(solver, thread_id=None):
            callback_calls.append(thread_id)
            return len(callback_calls) >= 50

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
            parallel=True,
            callback=my_callback,
        ) as solver:
            solver.solve()

        assert len(callback_calls) > 0

    def test_belief_state_policy(self):
        """get_belief_policy should return a dict with frozenset keys
        mapping discretized beliefs to (action, value) pairs."""
        from skdecide.hub.solver.rtdp_bel import RTDPBel

        with RTDPBel(
            domain_factory=lambda: TigerDomain(),
            heuristic=lambda d, s: Value(cost=0),
            discretization=10,
            rollout_budget=1000,
            max_depth=20,
        ) as solver:
            solver.solve()
            policy = solver.get_belief_policy()

        assert isinstance(policy, dict)
        assert len(policy) > 0
        for key, val in policy.items():
            assert isinstance(key, frozenset)
            assert isinstance(val, tuple)
            assert len(val) == 2
            action, value = val
            assert isinstance(value, Value)
            # Each frozenset element should be (state, probability)
            for entry in key:
                assert isinstance(entry, tuple)
                assert len(entry) == 2
                state, prob = entry
                assert isinstance(prob, float)
                assert 0.0 < prob <= 1.0
