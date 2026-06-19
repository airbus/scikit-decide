# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl_domains")
BLOCKS_DOMAIN = os.path.join(PDDL_DIR, "blocks", "domain.pddl")
BLOCKS_PROBLEM = os.path.join(PDDL_DIR, "blocks", "probBLOCKS-3-0.pddl")
TIREWORLD_DOMAIN = os.path.join(PDDL_DIR, "tireworld", "domain.pddl")
TIREWORLD_PROBLEM = os.path.join(PDDL_DIR, "tireworld", "p01.pddl")


@pytest.fixture
def blocks_domain():
    from skdecide.hub.domain.pddl import PDDLDomain

    return PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)


@pytest.fixture
def tireworld_domain():
    from skdecide.hub.domain.pddl import PPDDLDomain

    return PPDDLDomain(TIREWORLD_DOMAIN, TIREWORLD_PROBLEM)


class TestPDDLDomain:
    def test_construction(self, blocks_domain):
        assert blocks_domain is not None
        assert blocks_domain.task is not None

    def test_initial_state(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        assert s0 is not None
        assert hash(s0) == hash(s0)
        s0b = blocks_domain._get_initial_state_()
        assert s0 == s0b

    def test_initial_state_hashable(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        d = {s0: "initial"}
        assert d[s0] == "initial"

    def test_applicable_actions_nonempty(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        elts = actions.get_elements()
        assert len(elts) > 0

    def test_applicable_actions_initial_blocks(self, blocks_domain):
        """In blocks-3-0 initial state (C clear, A clear on B, B on table,
        handempty), applicable actions should be: pick-up(C) and
        unstack(A, B). A is NOT ontable so pick-up(A) is not applicable."""
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        elts = actions.get_elements()
        assert len(elts) == 2
        for a in elts:
            assert a.action_id >= 0

    def test_action_hashable(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        elts = actions.get_elements()
        a0 = elts[0]
        d = {a0: "first"}
        assert d[a0] == "first"

    def test_deterministic_transition(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        elts = actions.get_elements()
        a = elts[0]
        s1 = blocks_domain._get_next_state(s0, a)
        assert s1 is not None
        assert isinstance(s1, type(s0))

    def test_transition_value_unit_cost(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        a = actions.get_elements()[0]
        s1 = blocks_domain._get_next_state(s0, a)
        v = blocks_domain._get_transition_value(s0, a, s1)
        assert v.cost >= 0

    def test_goal_checking(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        assert not blocks_domain._goal_checker.is_goal(s0.to_cpp())

    def test_is_terminal_initial(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        assert not blocks_domain._is_terminal(s0)

    def test_state_changes_after_action(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        a = actions.get_elements()[0]
        s1 = blocks_domain._get_next_state(s0, a)
        assert s0 != s1

    def test_action_repr(self, blocks_domain):
        s0 = blocks_domain._get_initial_state_()
        actions = blocks_domain._get_applicable_actions_from(s0)
        a = actions.get_elements()[0]
        r = repr(a)
        assert r.startswith("(")
        assert r.endswith(")")

    def test_astar_solve_blocks(self, blocks_domain):
        """End-to-end: A* should solve blocks-3-0 (goal: B on C)."""
        from skdecide import utils

        Astar = utils.load_registered_solver("Astar")
        domain = blocks_domain
        with Astar(domain_factory=lambda: domain) as solver:
            solver.solve()
            obs = domain.reset()
            plan = []
            for _ in range(20):
                if domain._is_terminal(obs):
                    break
                action = solver.sample_action(obs)
                plan.append(action)
                outcome = domain.step(action)
                obs = outcome.observation
            assert domain._goal_checker.is_goal(obs.to_cpp()), (
                f"A* did not reach the goal. Plan: {plan}"
            )


class TestPPDDLDomain:
    def test_construction(self, tireworld_domain):
        assert tireworld_domain is not None

    def test_initial_state(self, tireworld_domain):
        s0 = tireworld_domain._get_initial_state_()
        assert s0 is not None

    def test_applicable_actions_nonempty(self, tireworld_domain):
        s0 = tireworld_domain._get_initial_state_()
        actions = tireworld_domain._get_applicable_actions_from(s0)
        elts = actions.get_elements()
        assert len(elts) > 0

    def test_probabilistic_distribution(self, tireworld_domain):
        """move-car has probabilistic effect: 2/5 chance of flat tire."""
        s0 = tireworld_domain._get_initial_state_()
        actions = tireworld_domain._get_applicable_actions_from(s0)
        elts = actions.get_elements()
        move_actions = [a for a in elts if "move" in repr(a).lower()]
        if not move_actions:
            pytest.skip("No move actions available from initial state")
        a = move_actions[0]
        dist = tireworld_domain._get_next_state_distribution(s0, a)
        values = dist.get_values()
        total_prob = sum(p for _, p in values)
        assert abs(total_prob - 1.0) < 1e-9
        assert len(values) >= 2

    def test_is_terminal_initial(self, tireworld_domain):
        s0 = tireworld_domain._get_initial_state_()
        assert not tireworld_domain._is_terminal(s0)
