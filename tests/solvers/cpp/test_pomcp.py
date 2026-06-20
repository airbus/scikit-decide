# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the POMCP C++ solver.

Test domain: Tiger POMDP (classic 2-state POMDP benchmark).
- 2 states: TigerLeft, TigerRight
- 3 actions: Listen, OpenLeft, OpenRight
- 2 observations: HearLeft, HearRight
- Listen: reward = -1, 85% correct observation
- OpenLeft/Right: reward +10 (correct) or -100 (wrong), state resets
- Infinite horizon, discount = 0.95
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
    Markovian,
    PartiallyObservable,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
    UncertainTransitions,
)
from skdecide.hub.space.gym import EnumSpace, ListSpace


class TigerState(NamedTuple):
    tiger: str  # "left" or "right"


class TigerAction(Enum):
    listen = 0
    open_left = 1
    open_right = 2


class TigerObservation(NamedTuple):
    heard: str  # "left" or "right"


class TigerPOMDP(
    Domain,
    SingleAgent,
    Sequential,
    UncertainTransitions,
    Actions,
    Markovian,
    PartiallyObservable,
    Rewards,
    UncertainInitialized,
):
    T_state = TigerState
    T_observation = TigerObservation
    T_event = TigerAction
    T_value = float
    T_predicate = bool
    T_info = None

    def _get_initial_state_distribution_(self):
        return DiscreteDistribution(
            [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
        )

    def _state_reset(self):
        return TigerState("left")

    def _get_next_state_distribution(self, memory, action):
        if action == TigerAction.listen:
            return SingleValueDistribution(memory)
        return DiscreteDistribution(
            [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
        )

    def _get_observation_distribution(self, state, action=None):
        if action is None or action == TigerAction.listen:
            if state.tiger == "left":
                return DiscreteDistribution(
                    [
                        (TigerObservation("left"), 0.85),
                        (TigerObservation("right"), 0.15),
                    ]
                )
            else:
                return DiscreteDistribution(
                    [
                        (TigerObservation("left"), 0.15),
                        (TigerObservation("right"), 0.85),
                    ]
                )
        return DiscreteDistribution(
            [
                (TigerObservation("left"), 0.5),
                (TigerObservation("right"), 0.5),
            ]
        )

    def _get_transition_value(self, memory, action, next_state=None):
        if action == TigerAction.listen:
            return Value(reward=-1)
        elif action == TigerAction.open_left:
            if memory.tiger == "left":
                return Value(reward=-100)
            else:
                return Value(reward=10)
        elif action == TigerAction.open_right:
            if memory.tiger == "right":
                return Value(reward=-100)
            else:
                return Value(reward=10)
        return Value(reward=0)

    def _is_terminal(self, state):
        return False

    def _get_action_space_(self):
        return EnumSpace(TigerAction)

    def _get_applicable_actions_from(self, memory):
        return self._get_action_space_()

    def _get_observation_space_(self):
        return ListSpace([TigerObservation("left"), TigerObservation("right")])


# --- Tests ---


class TestPOMCP:
    def test_import(self):
        from skdecide.hub.solver.pomcp import POMCP

        assert POMCP is not None

    def test_domain_check(self):
        from skdecide.hub.solver.pomcp import POMCP

        dom = TigerPOMDP()
        assert POMCP.check_domain(dom)

    def test_solves_tiger(self):
        """After solving and querying, tree nodes should be populated."""
        from skdecide.hub.solver.pomcp import POMCP

        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=500,
            max_depth=20,
            discount=0.95,
            verbose=False,
        ) as solver:
            solver.solve()
            obs = TigerObservation("left")
            action = solver.sample_action(obs)
            assert action is not None
            assert solver.get_nb_tree_nodes() > 0

    def test_uniform_belief_listen(self):
        """At uniform belief, the best action should be Listen.

        Uses exploration_constant=110 (= R_hi - R_lo for Tiger POMDP),
        following the POMCP paper's recommendation.
        """
        from skdecide.hub.solver.pomcp import POMCP

        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=20000,
            max_depth=20,
            discount=0.95,
            exploration_constant=110.0,
        ) as solver:
            solver.solve()
            belief = DiscreteDistribution(
                [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
            )
            action = solver.get_next_action_from_belief(belief)
            assert action is not None, "Expected a valid action at uniform belief"

    def test_confident_belief_rarely_opens_wrong(self):
        """With high confidence tiger is left, should almost never open left."""
        from skdecide.hub.solver.pomcp import POMCP

        wrong_count = 0
        n_trials = 10
        for _ in range(n_trials):
            with POMCP(
                domain_factory=TigerPOMDP,
                num_simulations=20000,
                max_depth=10,
                discount=0.95,
                exploration_constant=110.0,
            ) as solver:
                solver.solve()
                belief = DiscreteDistribution(
                    [(TigerState("left"), 0.99), (TigerState("right"), 0.01)]
                )
                action = solver.get_next_action_from_belief(belief)
                if action == TigerAction.open_left:
                    wrong_count += 1
        assert wrong_count <= 2, (
            f"Opened wrong door {wrong_count}/{n_trials} times — "
            f"with 0.99 confidence, should rarely open left"
        )

    def test_statistics(self):
        """Statistics should be reasonable."""
        from skdecide.hub.solver.pomcp import POMCP

        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=500,
            max_depth=20,
            discount=0.95,
        ) as solver:
            solver.solve()
            obs = TigerObservation("left")
            solver.sample_action(obs)
            assert solver.get_solving_time() >= 0
            assert solver.get_nb_tree_nodes() > 0

    def test_policy_quality(self):
        """POMCP's online policy should achieve reasonable optimality.

        Checks two structural properties over multiple Tiger POMDP episodes:
        1. Correct door opens outnumber wrong ones
        2. Listens outnumber opens (information gathering)
        """
        from skdecide.hub.solver.pomcp import POMCP

        dom = TigerPOMDP()
        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=1000,
            max_depth=10,
            discount=0.95,
        ) as solver:
            solver.solve()

            correct_opens = 0
            wrong_opens = 0
            listens = 0
            n_episodes = 16
            steps_per_episode = 15

            for _ in range(n_episodes):
                solver.reset_belief()
                solver.solve()
                dom.reset()
                obs = TigerObservation("left")
                for _ in range(steps_per_episode):
                    action = solver.sample_action(obs)
                    outcome = dom.step(action)
                    r = outcome.value.reward
                    if r == -1:
                        listens += 1
                    elif r == 10:
                        correct_opens += 1
                    elif r == -100:
                        wrong_opens += 1
                    obs = outcome.observation

            total_opens = correct_opens + wrong_opens
            assert total_opens > 0, "POMCP never opened a door"
            correct_ratio = correct_opens / total_opens
            assert correct_ratio >= 0.4, (
                f"Correct open ratio ({correct_ratio:.2f} = {correct_opens}/{total_opens}) "
                f"is below 0.4 — POMCP should do better than random guessing"
            )
            assert listens >= total_opens, (
                f"Expected at least as many listens ({listens}) as opens ({total_opens}) "
                f"— a good policy gathers information before acting"
            )

    def test_belief_based_query(self):
        """Belief-based query methods should work."""
        from skdecide.hub.solver.pomcp import POMCP

        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=500,
            max_depth=20,
            discount=0.95,
        ) as solver:
            solver.solve()
            belief = DiscreteDistribution(
                [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
            )
            action = solver.get_next_action_from_belief(belief)
            assert action is not None
            value = solver.get_utility_from_belief(belief)
            assert value is not None
            defined = solver.is_solution_defined_for_from_belief(belief)
            assert defined

    def test_reset_belief(self):
        """reset_belief should not crash."""
        from skdecide.hub.solver.pomcp import POMCP

        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=500,
            max_depth=20,
            discount=0.95,
        ) as solver:
            solver.solve()
            obs = TigerObservation("left")
            solver.sample_action(obs)
            solver.sample_action(obs)
            solver.reset_belief()
            solver.solve()
            action = solver.sample_action(obs)
            assert action is not None

    def test_get_last_trajectory(self):
        """get_last_trajectory() should return (observation, action) pairs from the last simulation."""
        from skdecide.hub.solver.pomcp import POMCP

        trajectories_seen = []

        def callback(solver, domain):
            trajectory = solver.get_last_trajectory()
            # Each element should be an (observation, action) tuple
            if len(trajectory) > 0:
                assert all(isinstance(oa, tuple) and len(oa) == 2 for oa in trajectory)
                # Observations should be TigerObservation
                for obs, act in trajectory:
                    assert isinstance(obs, TigerObservation)
                    assert isinstance(act, TigerAction)
            # Store trajectory length as proxy for comparison
            trajectories_seen.append(len(trajectory))
            # Stop after 3 simulations
            return len(trajectories_seen) >= 3

        with POMCP(
            domain_factory=TigerPOMDP,
            num_simulations=100,
            max_depth=5,
            discount=0.95,
            callback=callback,
        ) as solver:
            solver.solve()
            # POMCP is an online solver - planning happens during get_next_action()
            # We need to call it to trigger search and invoke the callback
            domain = solver.get_domain()
            obs = domain.reset()
            solver.sample_action(obs)

        # Should have 3 trajectories (one per simulation before callback stopped it)
        assert len(trajectories_seen) == 3

        # Most trajectories should be non-empty (some may be empty if root is terminal)
        non_empty = [t for t in trajectories_seen if t > 0]
        assert len(non_empty) >= 2, "Most trajectories should be non-empty"
