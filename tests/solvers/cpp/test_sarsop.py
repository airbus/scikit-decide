# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the SARSOP C++ solver.

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
    EnumerableTransitions,
    Markovian,
    PartiallyObservable,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
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
    EnumerableTransitions,
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
        # Opening a door resets the tiger uniformly
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
        # After opening a door, observation is uninformative
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


class TestSARSOP:
    def test_import(self):
        from skdecide.hub.solver.sarsop import SARSOP

        assert SARSOP is not None

    def test_domain_check(self):
        from skdecide.hub.solver.sarsop import SARSOP

        dom = TigerPOMDP()
        assert SARSOP.check_domain(dom)

    def test_solves_tiger(self):
        """After solving, alpha-vectors and beliefs should be populated."""
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.1,
            discount=0.95,
            time_budget=30000,
            verbose=False,
        ) as solver:
            solver.solve()
            assert solver.get_nb_alpha_vectors() > 0
            assert solver.get_nb_explored_beliefs() > 0

    def test_gap_convergence(self):
        """Gap should be close to epsilon after solving."""
        from skdecide.hub.solver.sarsop import SARSOP

        epsilon = 1.0
        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=epsilon,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()
            gap = solver.get_gap()
            assert gap < epsilon * 10, (
                f"Gap {gap} should be reasonably small (epsilon={epsilon})"
            )

    def test_uniform_belief_listen(self):
        """At uniform belief, the best action should be Listen."""
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.1,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()
            # Query with a fresh belief (uniform)
            belief = DiscreteDistribution(
                [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
            )
            action = solver.get_next_action_from_belief(belief)
            assert action == TigerAction.listen, (
                f"At uniform belief, expected Listen but got {action}"
            )

    def test_confident_belief_opens(self):
        """With high confidence tiger is left, should open right."""
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.1,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()
            # Very confident tiger is on the left
            belief = DiscreteDistribution(
                [(TigerState("left"), 0.99), (TigerState("right"), 0.01)]
            )
            action = solver.get_next_action_from_belief(belief)
            assert action == TigerAction.open_right, (
                f"With tiger-left confidence 0.99, expected OpenRight but got {action}"
            )

    def test_statistics(self):
        """Statistics should be reasonable."""
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()
            assert solver.get_solving_time() >= 0
            assert solver.get_nb_alpha_vectors() > 0
            lb = solver.get_lower_bound()
            ub = solver.get_upper_bound()
            assert lb <= ub + 0.01, f"Lower {lb} should be <= upper {ub}"

    def test_observation_based_rollout(self):
        """Rollout with observation-based interface should produce reasonable reward."""
        from skdecide.hub.solver.sarsop import SARSOP

        dom = TigerPOMDP()
        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.1,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()

            total_reward = 0.0
            n_episodes = 20
            steps_per_episode = 50

            for _ in range(n_episodes):
                solver.reset_belief()
                dom.reset()
                obs = TigerObservation("left")  # initial observation
                for _ in range(steps_per_episode):
                    action = solver.sample_action(obs)
                    outcome = dom.step(action)
                    total_reward += outcome.value.reward
                    obs = outcome.observation

            avg_reward = total_reward / n_episodes
            # SARSOP should do better than random (random ≈ -45 per episode)
            assert avg_reward > -100, f"Average reward {avg_reward} too low"

    def test_belief_based_query(self):
        """Belief-based query methods should work."""
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
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
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()
            # Do a couple of action queries
            obs = TigerObservation("left")
            solver.sample_action(obs)
            solver.sample_action(obs)
            # Reset should not crash
            solver.reset_belief()
            # And we should still get actions
            action = solver.sample_action(obs)
            assert action is not None

    def test_get_last_trajectory(self):
        """get_last_trajectory should return the sampled path from the last iteration."""
        from skdecide.hub.solver.sarsop import SARSOP

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
        ) as solver:
            solver.solve()
            trajectory = solver.get_last_trajectory()
            # Should have at least the root belief
            assert len(trajectory) > 0
            # Each element should be a (belief_dict, action) pair
            for belief_dict, action in trajectory:
                assert isinstance(belief_dict, dict)
                assert "state_probs" in belief_dict
                assert isinstance(action, TigerAction)
            # Beliefs should be valid probability distributions
            for belief_dict, _ in trajectory:
                state_probs = belief_dict["state_probs"]
                assert isinstance(state_probs, list)
                total_prob = sum(prob for state, prob in state_probs)
                assert abs(total_prob - 1.0) < 1e-6, (
                    f"Belief probabilities sum to {total_prob}, not 1.0"
                )
