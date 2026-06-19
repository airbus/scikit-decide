# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the HSVI and Goal-HSVI C++ solvers.

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
    EnumerableTransitions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
)
from skdecide.hub.space.gym import EnumSpace

# --- Tiger POMDP domain (cost version for Goal-HSVI) ---


class TigerState(NamedTuple):
    tiger_location: str  # "left" or "right" or "done"


class TigerAction(Enum):
    listen = 0
    open_left = 1
    open_right = 2


class TigerObservation(NamedTuple):
    heard: str  # "left", "right", or "done"


class TigerDomainCost(
    Domain,
    SingleAgent,
    Sequential,
    EnumerableTransitions,
    Actions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
    UncertainInitialized,
):
    """Tiger POMDP with cost minimization and goals (for Goal-HSVI)."""

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

    def _get_initial_state_distribution_(self):
        return DiscreteDistribution(
            [
                (TigerState(tiger_location="left"), 0.5),
                (TigerState(tiger_location="right"), 0.5),
            ]
        )

    def _state_reset(self) -> TigerState:
        return TigerState(tiger_location=self._initial_tiger)

    def _get_next_state_distribution(self, memory, action):
        s = memory
        if s.tiger_location == "done":
            return SingleValueDistribution(s)

        if action == TigerAction.listen:
            return SingleValueDistribution(s)
        else:
            return SingleValueDistribution(TigerState(tiger_location="done"))

    def _get_transition_value(self, memory, action, next_state=None):
        s = memory
        if s.tiger_location == "done":
            return Value(cost=0)

        if action == TigerAction.listen:
            return Value(cost=1)
        elif action == TigerAction.open_left:
            if s.tiger_location == "left":
                return Value(cost=100)
            else:
                return Value(cost=0)
        elif action == TigerAction.open_right:
            if s.tiger_location == "right":
                return Value(cost=100)
            else:
                return Value(cost=0)

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


class TigerDomainReward(
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
    """Tiger POMDP with reward maximization (for HSVI)."""

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

    def _get_initial_state_distribution_(self):
        return DiscreteDistribution(
            [
                (TigerState(tiger_location="left"), 0.5),
                (TigerState(tiger_location="right"), 0.5),
            ]
        )

    def _state_reset(self) -> TigerState:
        return TigerState(tiger_location=self._initial_tiger)

    def _get_next_state_distribution(self, memory, action):
        s = memory
        if s.tiger_location == "done":
            return SingleValueDistribution(s)

        if action == TigerAction.listen:
            return SingleValueDistribution(s)
        else:
            return SingleValueDistribution(TigerState(tiger_location="done"))

    def _get_transition_value(self, memory, action, next_state=None):
        s = memory
        if s.tiger_location == "done":
            return Value(reward=0)

        if action == TigerAction.listen:
            return Value(reward=-1)
        elif action == TigerAction.open_left:
            if s.tiger_location == "left":
                return Value(reward=-100)
            else:
                return Value(reward=10)
        elif action == TigerAction.open_right:
            if s.tiger_location == "right":
                return Value(reward=-100)
            else:
                return Value(reward=10)

    def _is_terminal(self, state) -> bool:
        return state.tiger_location == "done"

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


# --- Goal-HSVI Tests ---


class TestGoalHSVI:
    def test_import(self):
        from skdecide.hub.solver.hsvi import GoalHSVI

        assert GoalHSVI is not None

    def test_domain_check(self):
        from skdecide.hub.solver.hsvi import GoalHSVI

        dom = TigerDomainCost()
        assert GoalHSVI.check_domain(dom)

    def test_solves_tiger(self):
        """Goal-HSVI should converge on the Tiger POMDP."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
            verbose=False,
        ) as solver:
            solver.solve()
            n_alphas = solver.get_nb_alpha_vectors()
            gap = solver.get_gap()

        assert n_alphas > 0
        assert gap < 100  # should have made progress

    def test_point_belief_optimal_action(self):
        """With tiger known to be left, Goal-HSVI should find open_right."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
        ) as solver:
            solver.solve()

            # Query with a point belief: tiger is definitely left
            point_belief = DiscreteDistribution(
                [(TigerState(tiger_location="left"), 1.0)]
            )
            action = solver.get_next_action_from_belief(point_belief)

        assert action == TigerAction.open_right

    def test_point_belief_optimal_value(self):
        """With tiger_left known, cost should be 0 (just open right door)."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
        ) as solver:
            solver.solve()

            point_belief = DiscreteDistribution(
                [(TigerState(tiger_location="left"), 1.0)]
            )
            v = solver.get_utility_from_belief(point_belief)

        assert v is not None
        assert v.cost < 2.0  # optimal is 0, allow some tolerance

    def test_observation_interface(self):
        """Test the observation-based interaction interface."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        dom = TigerDomainCost()

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
        ) as solver:
            solver.solve()

            obs = dom.reset()
            actions = []
            for _ in range(10):
                if hasattr(obs, "heard") and obs.heard == "done":
                    break
                action = solver.sample_action(obs)
                actions.append(action)
                outcome = dom.step(action)
                obs = outcome.observation

            solver.reset_belief()

        assert len(actions) > 0

    def test_belief_interface(self):
        """Test the explicit belief-based query interface."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
        ) as solver:
            solver.solve()

            # Uniform belief — should listen or open_right (not open_left)
            uniform_belief = DiscreteDistribution(
                [
                    (TigerState(tiger_location="left"), 0.5),
                    (TigerState(tiger_location="right"), 0.5),
                ]
            )
            action = solver.get_next_action_from_belief(uniform_belief)
            is_defined = solver.is_solution_defined_for_from_belief(uniform_belief)

        assert is_defined
        assert action is not None

    def test_solving_time(self):
        """get_solving_time should return a non-negative value."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=10000,
            max_sample_depth=20,
        ) as solver:
            solver.solve()
            t = solver.get_solving_time()

        assert t >= 0


# --- HSVI Tests ---


class TestHSVI:
    def test_import(self):
        from skdecide.hub.solver.hsvi import HSVI

        assert HSVI is not None

    def test_solves_tiger_reward(self):
        """HSVI should find a solution for the discounted reward Tiger POMDP."""
        from skdecide.hub.solver.hsvi import HSVI

        with HSVI(
            domain_factory=lambda: TigerDomainReward(),
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
            max_sample_depth=50,
        ) as solver:
            solver.solve()
            n_alphas = solver.get_nb_alpha_vectors()
            gap = solver.get_gap()

        assert n_alphas > 0
        assert gap < 200  # should have made progress

    def test_point_belief_action(self):
        """With tiger_left known, HSVI should find open_right (reward 10)."""
        from skdecide.hub.solver.hsvi import HSVI

        with HSVI(
            domain_factory=lambda: TigerDomainReward(),
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
            max_sample_depth=50,
        ) as solver:
            solver.solve()

            point_belief = DiscreteDistribution(
                [(TigerState(tiger_location="left"), 1.0)]
            )
            action = solver.get_next_action_from_belief(point_belief)

        assert action == TigerAction.open_right


# --- Parallel Goal-HSVI Tests ---


class TestGoalHSVIParallel:
    def test_solves_tiger_parallel(self):
        """Goal-HSVI should converge on the Tiger POMDP with parallel=True."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
            parallel=True,
        ) as solver:
            solver.solve()
            n_alphas = solver.get_nb_alpha_vectors()
            gap = solver.get_gap()

        assert n_alphas > 0
        assert gap < 100

    def test_point_belief_optimal_action_parallel(self):
        """With tiger known to be left, parallel Goal-HSVI should find open_right."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
            parallel=True,
        ) as solver:
            solver.solve()

            point_belief = DiscreteDistribution(
                [(TigerState(tiger_location="left"), 1.0)]
            )
            action = solver.get_next_action_from_belief(point_belief)

        assert action == TigerAction.open_right

    def test_belief_interface_parallel(self):
        """Test the explicit belief-based query interface with parallel=True."""
        from skdecide.hub.solver.hsvi import GoalHSVI

        with GoalHSVI(
            domain_factory=lambda: TigerDomainCost(),
            epsilon=0.5,
            time_budget=30000,
            max_sample_depth=50,
            parallel=True,
        ) as solver:
            solver.solve()

            uniform_belief = DiscreteDistribution(
                [
                    (TigerState(tiger_location="left"), 0.5),
                    (TigerState(tiger_location="right"), 0.5),
                ]
            )
            action = solver.get_next_action_from_belief(uniform_belief)
            is_defined = solver.is_solution_defined_for_from_belief(uniform_belief)

        assert is_defined
        assert action is not None


# --- Parallel HSVI Tests ---


class TestHSVIParallel:
    def test_solves_tiger_reward_parallel(self):
        """HSVI should find a solution with parallel=True."""
        from skdecide.hub.solver.hsvi import HSVI

        with HSVI(
            domain_factory=lambda: TigerDomainReward(),
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
            max_sample_depth=50,
            parallel=True,
        ) as solver:
            solver.solve()
            n_alphas = solver.get_nb_alpha_vectors()
            gap = solver.get_gap()

        assert n_alphas > 0
        assert gap < 200

    def test_point_belief_action_parallel(self):
        """With tiger_left known, parallel HSVI should find open_right."""
        from skdecide.hub.solver.hsvi import HSVI

        with HSVI(
            domain_factory=lambda: TigerDomainReward(),
            epsilon=0.5,
            discount=0.95,
            time_budget=30000,
            max_sample_depth=50,
            parallel=True,
        ) as solver:
            solver.solve()

            point_belief = DiscreteDistribution(
                [(TigerState(tiger_location="left"), 1.0)]
            )
            action = solver.get_next_action_from_belief(point_belief)

        assert action == TigerAction.open_right
