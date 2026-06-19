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
def hmax(blocks_domain):
    from skdecide.hub.domain.pddl import HMax

    return HMax(blocks_domain._task, verbose=True)


@pytest.fixture
def hadd(blocks_domain):
    from skdecide.hub.domain.pddl import HAdd

    return HAdd(blocks_domain._task, verbose=True)


class TestDeleteRelaxationHeuristics:
    def test_construction(self, hmax, hadd):
        assert hmax.num_atoms > 0
        assert hmax.num_relaxed_actions > 0
        assert hadd.num_atoms > 0
        assert hadd.num_relaxed_actions > 0

    def test_initial_state_not_goal(self, blocks_domain, hmax, hadd):
        init = blocks_domain._task.initial_state()
        h_max_val = hmax(init)
        h_add_val = hadd(init)
        assert h_max_val > 0, "Initial state should not be at goal"
        assert h_add_val > 0, "Initial state should not be at goal"

    def test_hadd_geq_hmax(self, blocks_domain, hmax, hadd):
        init = blocks_domain._task.initial_state()
        h_max_val = hmax(init)
        h_add_val = hadd(init)
        assert h_add_val >= h_max_val, "h_add should always be >= h_max"

    def test_goal_state_zero(self, blocks_domain, hmax, hadd):
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_GoalChecker_ as CppGoalChecker,
        )
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_SuccessorGenerator_ as CppSuccessorGenerator,
        )

        task = blocks_domain._task
        succ_gen = CppSuccessorGenerator(task)
        goal_checker = CppGoalChecker(task)
        aops_gen = blocks_domain._aops_gen

        # BFS to find a goal state
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
        assert hmax(goal_state) == 0.0, "h_max at goal must be 0"
        assert hadd(goal_state) == 0.0, "h_add at goal must be 0"

    def test_not_dead_end(self, blocks_domain, hmax, hadd):
        init = blocks_domain._task.initial_state()
        assert hmax(init) < 1e9, "Initial state should not be a dead end"
        assert hadd(init) < 1e9, "Initial state should not be a dead end"

    def test_solver_heuristic(self, blocks_domain, hadd):
        from skdecide.hub.domain.pddl.domain import PDDLState

        task = blocks_domain._task
        init_cpp = task.initial_state()
        init_pddl = PDDLState(init_cpp, blocks_domain._cost_function_ids)
        heuristic_fn = hadd()
        val = heuristic_fn(blocks_domain, init_pddl)
        assert val.cost > 0


@pytest.fixture
def tireworld_domain():
    from skdecide.hub.domain.pddl import PPDDLDomain

    return PPDDLDomain(TIREWORLD_DOMAIN, TIREWORLD_PROBLEM)


class TestPPDDLHeuristics:
    """Test h⁺_max / h⁺_add on probabilistic domains (Proposition 1 — SSP)."""

    @pytest.fixture
    def hmax(self, tireworld_domain):
        from skdecide.hub.domain.pddl import HMax

        return HMax(tireworld_domain._task, verbose=True)

    @pytest.fixture
    def hadd(self, tireworld_domain):
        from skdecide.hub.domain.pddl import HAdd

        return HAdd(tireworld_domain._task, verbose=True)

    def test_ppddl_construction(self, hmax, hadd):
        assert hmax.num_atoms > 0
        assert hmax.num_relaxed_actions > 0
        assert hadd.num_atoms > 0
        assert hadd.num_relaxed_actions > 0

    def test_ppddl_initial_not_goal(self, tireworld_domain, hmax, hadd):
        init = tireworld_domain._task.initial_state()
        assert hmax(init) > 0, "Initial state should not be at goal"
        assert hadd(init) > 0, "Initial state should not be at goal"

    def test_ppddl_hadd_geq_hmax(self, tireworld_domain, hmax, hadd):
        init = tireworld_domain._task.initial_state()
        assert hadd(init) >= hmax(init), "h_add should always be >= h_max"

    def test_ppddl_goal_zero(self, tireworld_domain, hmax, hadd):
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_GoalChecker_ as CppGoalChecker,
        )
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_SuccessorGenerator_ as CppSuccessorGenerator,
        )

        task = tireworld_domain._task
        succ_gen = CppSuccessorGenerator(task)
        goal_checker = CppGoalChecker(task)
        aops_gen = tireworld_domain._aops_gen

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
        assert hmax(goal_state) == 0.0, "h_max at goal must be 0"
        assert hadd(goal_state) == 0.0, "h_add at goal must be 0"


class TestDiscountedHeuristics:
    """Test h^γ_max / h^γ_add (Theorem 2 — DSSP with dead-ends)."""

    @pytest.fixture
    def hmax_disc(self, tireworld_domain):
        from skdecide.hub.domain.pddl import HMax

        return HMax(tireworld_domain._task, discount_factor=0.9, verbose=True)

    @pytest.fixture
    def hadd_disc(self, tireworld_domain):
        from skdecide.hub.domain.pddl import HAdd

        return HAdd(tireworld_domain._task, discount_factor=0.9, verbose=True)

    @pytest.fixture
    def hmax_undiscounted(self, tireworld_domain):
        from skdecide.hub.domain.pddl import HMax

        return HMax(tireworld_domain._task, discount_factor=1.0)

    @pytest.fixture
    def hadd_undiscounted(self, tireworld_domain):
        from skdecide.hub.domain.pddl import HAdd

        return HAdd(tireworld_domain._task, discount_factor=1.0)

    def test_discounted_construction(self, hmax_disc, hadd_disc):
        assert hmax_disc.num_atoms > 0
        assert hmax_disc.num_relaxed_actions > 0
        assert hadd_disc.num_atoms > 0
        assert hadd_disc.num_relaxed_actions > 0

    def test_discounted_initial_positive(self, tireworld_domain, hmax_disc, hadd_disc):
        init = tireworld_domain._task.initial_state()
        assert hmax_disc(init) > 0
        assert hadd_disc(init) > 0

    def test_discounted_goal_zero(self, tireworld_domain, hmax_disc, hadd_disc):
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_GoalChecker_ as CppGoalChecker,
        )
        from skdecide.hub.__skdecide_hub_cpp import (
            _PDDL_SuccessorGenerator_ as CppSuccessorGenerator,
        )

        task = tireworld_domain._task
        succ_gen = CppSuccessorGenerator(task)
        goal_checker = CppGoalChecker(task)
        aops_gen = tireworld_domain._aops_gen

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
        assert hmax_disc(goal_state) == 0.0, "h^gamma_max at goal must be 0"
        assert hadd_disc(goal_state) == 0.0, "h^gamma_add at goal must be 0"

    def test_discounted_leq_undiscounted(
        self,
        tireworld_domain,
        hmax_disc,
        hmax_undiscounted,
        hadd_disc,
        hadd_undiscounted,
    ):
        init = tireworld_domain._task.initial_state()
        h_max_d = hmax_disc(init)
        h_max_u = hmax_undiscounted(init)
        h_add_d = hadd_disc(init)
        h_add_u = hadd_undiscounted(init)
        assert h_max_d <= h_max_u + 1e-9, (
            f"h^gamma_max ({h_max_d}) should be <= h_max ({h_max_u})"
        )
        assert h_add_d <= h_add_u + 1e-9, (
            f"h^gamma_add ({h_add_d}) should be <= h_add ({h_add_u})"
        )

    def test_deterministic_discount_error(self, blocks_domain):
        from skdecide.hub.domain.pddl import HMax

        with pytest.raises(ValueError, match="probabilistic effects"):
            HMax(blocks_domain._task, discount_factor=0.9)
