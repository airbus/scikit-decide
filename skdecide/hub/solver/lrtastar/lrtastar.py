# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# TODO comment.

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import Domain, Solver, Value
from skdecide.builders.domain import (
    Actions,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities


class D(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicTransitions,
    Actions,
    Goals,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    pass


class LRTAstar(Solver, DeterministicPolicies, Utilities, FromAnyState):
    """Learning Real-Time A* solver."""

    T_domain = D

    hyperparameters = [
        FloatHyperparameter(
            name="weight",
            low=0.0,
            high=1.0,
            suggest_high=True,
            suggest_low=True,
        ),
        IntegerHyperparameter(name="max_iter"),
        IntegerHyperparameter(name="max_depth"),
    ]

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._policy.get(observation, None)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return observation in self._policy

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        if observation not in self.values:
            return self._heuristic(self._domain, observation).cost
        return self.values[observation]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        heuristic: Optional[
            Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]]
        ] = None,
        weight: float = 1.0,
        verbose: bool = False,
        max_iter=5000,
        max_depth=200,
        callback: Callable[[LRTAstar], bool] = lambda solver: False,
    ) -> None:
        """

        # Parameters
        domain_factory
        heuristic
        weight
        verbose
        max_iter
        max_depth
        callback: function called at each solver iteration. If returning true, the solve process stops.

        """
        self.callback = callback
        Solver.__init__(self, domain_factory=domain_factory)
        self._domain = self._domain_factory()
        self._heuristic = (
            (lambda _, __: Value(cost=0.0)) if heuristic is None else heuristic
        )
        self._weight = weight
        self.max_iter = max_iter
        self.max_depth = max_depth
        self._plan: List[D.T_event] = []
        self.values = {}

        self._verbose = verbose

        self.heuristic_changed = False
        self._policy: Dict[D.T_observation, Optional[D.T_event]] = {}

    def get_policy(self) -> Dict[D.T_observation, Optional[D.T_event]]:
        """Return the computed policy."""
        return self._policy

    def _solve_from(
        self,
        memory: D.T_state,
    ) -> None:
        """Run the solving process from a given state.

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """
        self.values = {}
        iteration = 0
        best_cost = float("inf")
        # best_path = None
        while not self.callback(self):
            print(memory)
            dead_end, cumulated_cost, current_roll, list_action = self.doTrial(memory)
            if self._verbose:
                print(
                    "iter ",
                    iteration,
                    "/",
                    self.max_iter,
                    " : dead end, ",
                    dead_end,
                    " cost : ",
                    cumulated_cost,
                )
            if not dead_end and cumulated_cost < best_cost:
                best_cost = cumulated_cost
                # best_path = current_roll
                for k in range(len(current_roll)):
                    self._policy[current_roll[k][0]] = current_roll[k][1]["action"]
            if not self.heuristic_changed:
                print(self.heuristic_changed)
                return
            iteration += 1
            if iteration > self.max_iter:
                return

    def doTrial(self, from_observation: D.T_agent[D.T_observation]):
        list_action = []
        current_state = from_observation
        depth = 0
        dead_end = False
        cumulated_reward = 0.0
        current_roll = [current_state]
        current_roll_and_action = []
        self.heuristic_changed = False
        while (not self._domain.is_goal(current_state)) and (depth < self.max_depth):
            next_action = None
            next_state = None
            best_estimated_cost = float("inf")
            applicable_actions = self._domain.get_applicable_actions(current_state)
            for action in applicable_actions.get_elements():
                st = self._domain.get_next_state(current_state, action)
                r = self._domain.get_transition_value(current_state, action, st).cost
                if st in current_roll:
                    continue
                if st not in self.values:
                    self.values[st] = self._heuristic(self._domain, st).cost
                if r + self.values[st] < best_estimated_cost:
                    next_state = st
                    next_action = action
                    best_estimated_cost = r + self.values[st]
            if next_action is None:
                self.values[current_state] = float("inf")
                dead_end = True
                self.heuristic_changed = True
                break
            else:
                if (not current_state in self.values) or (
                    self.values[current_state] != best_estimated_cost
                ):
                    self.heuristic_changed = True
                    self.values[current_state] = best_estimated_cost
            cumulated_reward += best_estimated_cost - (
                self.values[next_state]
                if next_state in self.values
                else self._heuristic(self._domain, next_state).cost
            )
            list_action.append(next_action)
            current_roll_and_action.append((current_state, {"action": next_action}))
            current_state = next_state
            depth += 1
            current_roll.append(current_state)
        current_roll_and_action.append((current_state, {"action": None}))
        cumulated_reward += self.values[current_state]
        return dead_end, cumulated_reward, current_roll_and_action, list_action
