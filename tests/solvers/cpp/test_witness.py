# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Witness exact POMDP solver.

Reuses the Tiger POMDP domain from test_sarsop.py.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from skdecide import DiscreteDistribution

_sarsop_test = Path(__file__).parent / "test_sarsop.py"
spec = importlib.util.spec_from_file_location("test_sarsop", _sarsop_test)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

TigerState = _mod.TigerState
TigerAction = _mod.TigerAction
TigerObservation = _mod.TigerObservation
TigerPOMDP = _mod.TigerPOMDP


@pytest.fixture(scope="module")
def fast_witness_solver():
    """Shared Witness solver with fast parameters for policy-correctness tests."""
    from skdecide.hub.solver.witness import Witness

    with Witness(
        domain_factory=TigerPOMDP,
        epsilon=0.1,
        discount=0.95,
        max_iterations=200,
    ) as solver:
        solver.solve()
        yield solver


class TestWitness:
    def test_import(self):
        from skdecide.hub.solver.witness import Witness

        assert Witness is not None

    def test_domain_check(self):
        from skdecide.hub.solver.witness import Witness

        dom = TigerPOMDP()
        assert Witness.check_domain(dom)

    def test_solves_tiger(self, fast_witness_solver):
        """After solving, alpha-vectors should be populated."""
        assert fast_witness_solver.get_nb_alpha_vectors() > 0
        assert fast_witness_solver.get_nb_iterations() > 0

    def test_uniform_belief_listen(self, fast_witness_solver):
        """At uniform belief, the best action should be Listen."""
        belief = DiscreteDistribution(
            [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
        )
        action = fast_witness_solver.get_next_action_from_belief(belief)
        assert action == TigerAction.listen, (
            f"At uniform belief, expected Listen but got {action}"
        )

    def test_confident_belief_opens(self, fast_witness_solver):
        """With high confidence tiger is left, should open right."""
        belief = DiscreteDistribution(
            [(TigerState("left"), 0.99), (TigerState("right"), 0.01)]
        )
        action = fast_witness_solver.get_next_action_from_belief(belief)
        assert action == TigerAction.open_right, (
            f"With tiger-left confidence 0.99, expected OpenRight but got {action}"
        )

    def test_exact_matches_sarsop(self):
        """Witness and SARSOP should agree on values at test beliefs."""
        from skdecide.hub.solver.sarsop import SARSOP
        from skdecide.hub.solver.witness import Witness

        test_beliefs = [
            DiscreteDistribution(
                [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
            ),
            DiscreteDistribution(
                [(TigerState("left"), 0.85), (TigerState("right"), 0.15)]
            ),
            DiscreteDistribution(
                [(TigerState("left"), 0.15), (TigerState("right"), 0.85)]
            ),
        ]

        with Witness(
            domain_factory=TigerPOMDP,
            epsilon=0.01,
            discount=0.85,
            max_iterations=200,
        ) as w_solver:
            w_solver.solve()
            w_values = [
                w_solver.get_utility_from_belief(b).reward for b in test_beliefs
            ]

        with SARSOP(
            domain_factory=TigerPOMDP,
            epsilon=0.01,
            discount=0.85,
            time_budget=30000,
        ) as s_solver:
            s_solver.solve()
            s_values = [
                s_solver.get_utility_from_belief(b).reward for b in test_beliefs
            ]

        for i, (wv, sv) in enumerate(zip(w_values, s_values)):
            assert abs(wv - sv) < 0.5, (
                f"Belief {i}: Witness value {wv:.4f} vs SARSOP value {sv:.4f} "
                f"differ by {abs(wv - sv):.4f}"
            )

    def test_observation_based_rollout(self, fast_witness_solver):
        """Rollout with observation-based interface should produce reasonable reward."""
        dom = TigerPOMDP()
        total_reward = 0.0
        n_episodes = 20
        steps_per_episode = 50

        for _ in range(n_episodes):
            fast_witness_solver.reset_belief()
            dom.reset()
            obs = TigerObservation("left")
            for _ in range(steps_per_episode):
                action = fast_witness_solver.sample_action(obs)
                outcome = dom.step(action)
                total_reward += outcome.value.reward
                obs = outcome.observation

        avg_reward = total_reward / n_episodes
        assert avg_reward > -100, f"Average reward {avg_reward} too low"

    def test_belief_based_query(self, fast_witness_solver):
        """Belief-based query methods should work."""
        belief = DiscreteDistribution(
            [(TigerState("left"), 0.5), (TigerState("right"), 0.5)]
        )
        action = fast_witness_solver.get_next_action_from_belief(belief)
        assert action is not None
        value = fast_witness_solver.get_utility_from_belief(belief)
        assert value is not None
        defined = fast_witness_solver.is_solution_defined_for_from_belief(belief)
        assert defined

    def test_reset_belief(self, fast_witness_solver):
        """reset_belief should not crash."""
        obs = TigerObservation("left")
        fast_witness_solver.sample_action(obs)
        fast_witness_solver.sample_action(obs)
        fast_witness_solver.reset_belief()
        action = fast_witness_solver.sample_action(obs)
        assert action is not None

    def test_statistics(self, fast_witness_solver):
        """Statistics should be reasonable."""
        assert fast_witness_solver.get_solving_time() >= 0
        assert fast_witness_solver.get_nb_alpha_vectors() > 0
        assert fast_witness_solver.get_nb_iterations() > 0
