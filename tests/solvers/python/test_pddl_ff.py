# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

PDDL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "domains", "python", "pddl_domains"
)
BLOCKS_DOMAIN = os.path.join(PDDL_DIR, "blocks", "domain.pddl")
BLOCKS_PROBLEM = os.path.join(PDDL_DIR, "blocks", "probBLOCKS-3-0.pddl")
TIREWORLD_DOMAIN = os.path.join(PDDL_DIR, "tireworld", "domain.pddl")
TIREWORLD_PROBLEM = os.path.join(PDDL_DIR, "tireworld", "p01.pddl")


class TestHFFHeuristic:
    def test_hff_construction(self):
        from skdecide.hub.domain.pddl import HFF, PDDLDomain

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        hff = HFF(domain._task, verbose=True)
        assert hff.num_atoms > 0
        assert hff.num_relaxed_actions > 0

    def test_hff_initial_not_goal(self):
        from skdecide.hub.domain.pddl import HFF, PDDLDomain

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        hff = HFF(domain._task)
        init = domain._task.initial_state()
        h_val = hff(init)
        assert h_val > 0, "Initial state should have positive h_FF"

    def test_hff_goal_zero(self):
        from skdecide.hub.__skdecide_hub_cpp import _PDDL_GoalChecker_ as GoalChecker
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_SuccessorGenerator_ as SuccGen,
        )

        from skdecide.hub.domain.pddl import HFF, PDDLDomain

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        hff = HFF(domain._task)
        task = domain._task
        succ_gen = SuccGen(task)
        goal_checker = GoalChecker(task)
        aops_gen = domain._aops_gen

        from collections import deque

        visited = set()
        queue = deque()
        init = task.initial_state()
        queue.append(init)
        visited.add(hash(init))

        goal_state = None
        for _ in range(10000):
            if not queue:
                break
            state = queue.popleft()
            if goal_checker.is_goal(state):
                goal_state = state
                break
            actions = aops_gen.get_applicable_actions(state)
            for action in actions:
                succs = succ_gen.get_successors(state, action)
                for s in succs:
                    h = hash(s.state)
                    if h not in visited:
                        visited.add(h)
                        queue.append(s.state)

        assert goal_state is not None, "Should find a goal state via BFS"
        h_val = hff(goal_state)
        assert h_val == 0, f"h_FF at goal should be 0, got {h_val}"

    def test_hff_leq_hadd(self):
        from skdecide.hub.domain.pddl import HFF, HAdd, PDDLDomain

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        hff = HFF(domain._task)
        hadd = HAdd(domain._task)
        init = domain._task.initial_state()
        assert hff(init) <= hadd(init), "h_FF should be <= h_add"

    def test_hff_helpful_actions(self):
        from skdecide.hub.domain.pddl import HFF, PDDLDomain

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        hff = HFF(domain._task)
        init = domain._task.initial_state()
        h_val, helpful = hff.compute_with_helpful(init)
        assert h_val > 0
        assert len(helpful) > 0, "Should have helpful actions at initial state"


class TestHFFProbabilistic:
    def test_hff_ppddl(self):
        from skdecide.hub.domain.pddl import HFF, PPDDLDomain

        domain = PPDDLDomain(TIREWORLD_DOMAIN, TIREWORLD_PROBLEM)
        hff = HFF(domain._task, discount_factor=1.0)
        init = domain._task.initial_state()
        assert hff(init) > 0

    def test_hff_discounted(self):
        from skdecide.hub.domain.pddl import HFF, PPDDLDomain

        domain = PPDDLDomain(TIREWORLD_DOMAIN, TIREWORLD_PROBLEM)
        hff = HFF(domain._task, discount_factor=0.9)
        init = domain._task.initial_state()
        assert hff(init) > 0

    def test_hff_deterministic_discount_error(self):
        from skdecide.hub.domain.pddl import HFF, PDDLDomain

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        with pytest.raises(ValueError):
            HFF(domain._task, discount_factor=0.9)


class TestFFSolver:
    def test_ff_solve(self):
        from skdecide.hub.domain.pddl import PDDLDomain
        from skdecide.hub.solver.pddl.ff import FF

        ff = FF(domain_factory=lambda: PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM))
        ff.solve()
        assert ff.get_nb_explored_states() > 0

    def test_ff_plan(self):
        from skdecide.hub.domain.pddl import PDDLDomain
        from skdecide.hub.solver.pddl.ff import FF

        ff = FF(domain_factory=lambda: PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM))
        ff.solve()
        plan = ff.get_plan()
        assert len(plan) > 0, "Plan should be non-empty"

    def test_ff_follow_plan(self):
        from skdecide.hub.domain.pddl import PDDLDomain
        from skdecide.hub.solver.pddl.ff import FF

        ff = FF(domain_factory=lambda: PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM))
        ff.solve()

        domain = PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM)
        obs = domain.reset()
        steps = 0
        for _ in range(20):
            assert ff._is_solution_defined_for(obs), (
                f"Solution should be defined at step {steps}"
            )
            action = ff._get_next_action(obs)
            if action is None:
                break
            outcome = domain.step(action)
            obs = outcome.observation
            steps += 1
            if outcome.termination:
                break

        assert outcome.termination, "Should reach goal"
        assert steps == len(ff.get_plan()), "Steps should match plan length"

    def test_ff_is_solution_defined(self):
        from skdecide.hub.domain.pddl import PDDLDomain
        from skdecide.hub.solver.pddl.ff import FF

        ff = FF(domain_factory=lambda: PDDLDomain(BLOCKS_DOMAIN, BLOCKS_PROBLEM))
        ff.solve()
        plan = ff.get_plan()

        for i, (state, action) in enumerate(plan):
            assert ff._cpp_solver.is_solution_defined_for(state), (
                f"Plan state {i} should have solution defined"
            )
