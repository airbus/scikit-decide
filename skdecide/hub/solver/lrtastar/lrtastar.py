# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# TODO comment.

from __future__ import annotations

from typing import Optional, Callable, Any, Iterable

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, \
                                     Actions, Goals, Markovian, \
                                     FullyObservable, PositiveCosts
from skdecide.builders.solver import DeterministicPolicies, Utilities


class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, FullyObservable,
        PositiveCosts):
    pass


class LRTAstar(Solver, DeterministicPolicies, Utilities):
    T_domain = D

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._policy.get(observation, None)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return observation is self._policy

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        if observation not in self.values:
            return self._heuristic(observation, self._domain)
        return self.values[observation]

    def __init__(self,
                 from_state: Optional[D.T_state] = None,
                 heuristic: Optional[Callable[[D.T_state, Domain], float]] = None,
                 weight: float = 1.,
                 verbose: bool = False,
                 max_iter=5000,
                 max_depth=200) -> None:
        self._from_state = from_state
        self._heuristic = (lambda _, __: 0.) if heuristic is None else heuristic
        self._weight = weight
        self.max_iter = max_iter
        self.max_depth = max_depth
        self._plan = []
        self.values = {}

        self._verbose = verbose

        self.heuristic_changed = False
        self._policy = {}

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self._domain = domain_factory()
        self.values = {}
        iteration = 0
        best_cost = float('inf')
        if self._from_state is None:
            # get initial observation from domain (assuming DeterministicInitialized)
            from_observation = self._domain.get_initial_state()
        else:
            from_observation = self._from_state
        # best_path = None
        while True:
            print(from_observation)
            dead_end, cumulated_cost, current_roll, list_action = self.doTrial(from_observation)
            if self._verbose:
                print("iter ", iteration, '/', self.max_iter, " : dead end, ", dead_end, " cost : ", cumulated_cost)
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
        cumulated_reward = 0.
        current_roll = [current_state]
        current_roll_and_action = []
        self.heuristic_changed = False
        while (not self._domain.is_goal(current_state)) and (depth < self.max_depth):
            next_action = None
            next_state = None
            best_estimated_cost = float('inf')
            applicable_actions = self._domain.get_applicable_actions(current_state)
            for action in applicable_actions.get_elements():
                st = self._domain.get_next_state(current_state, action)
                r = self._domain.get_transition_value(current_state,
                                                      action,
                                                      st).cost
                if st in current_roll:
                    continue
                if st not in self.values:
                    self.values[st] = self._heuristic(st,
                                                      self._domain)
                if r + self.values[st] < best_estimated_cost:
                    next_state = st
                    next_action = action
                    best_estimated_cost = r + self.values[st]
            if next_action is None:
                self.values[current_state] = float('inf')
                dead_end = True
                self.heuristic_changed = True
                break
            else:
                if (not current_state in self.values) or (self.values[current_state] != best_estimated_cost):
                    self.heuristic_changed = True
                    self.values[current_state] = best_estimated_cost
            cumulated_reward += best_estimated_cost - (self.values[next_state] if next_state in self.values else
                                                       self._heuristic(next_state, self._domain))
            list_action.append(next_action)
            current_roll_and_action.append((current_state,
                                            {"action": next_action}))
            current_state = next_state
            depth += 1
            current_roll.append(current_state)
        current_roll_and_action.append((current_state,
                                        {"action": None}))
        cumulated_reward += self.values[current_state]
        return dead_end, cumulated_reward, current_roll_and_action, list_action
