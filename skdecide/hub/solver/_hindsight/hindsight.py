# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: update to new API
from __future__ import annotations

import collections
from typing import Dict, Hashable, Union

import pathos.pools as pp

from skdecide import Memory
from skdecide.builders.solver import DeterministicPolicySolver, SolutionSolver


def tree():
    return collections.defaultdict(tree)


class HindsightPlanner:
    dict_planner: Dict[int, Union[SolutionSolver, DeterministicPolicySolver]]
    weight_scenario: Dict[int, float]
    # 1st key : a hashable state, 2nd key: the id of scenario, and finally the value is a path returned by the planner.
    plan_by_scenario: Dict[Hashable, Dict[int, object]]

    def __init__(self, dict_planner,
                 weight_scenario,
                 reuse_plans: bool = True,
                 multithread: bool = True,
                 nb_thread: int = 5,
                 verbose: bool = False):
        self.dict_planner = dict_planner
        self.weight_scenario = weight_scenario
        self.plan_by_scenario = tree()
        self.action_by_scenario = tree()
        self.q_values_scenar = tree()
        self.q_values = tree()
        self.planned = tree()
        self.nb_thread = nb_thread
        self.multithread = multithread
        self.pool_threads = pp.ThreadPool(self.nb_thread)
        self.launch_things = self.pool_threads.map if self.multithread else map
        self.reuse_plans = reuse_plans
        self.verbose = verbose

    def give_first_action(self, source):
        self.first_pass(source)
        self.second_pass(source)
        print("QVALUES", [(str(a), self.q_values[source][a]) for a in self.q_values[source]])
        return min(self.q_values[source],
                   key=lambda x: self.q_values[source][x])

    def first_pass(self, source):
        missing = list(self.dict_planner.keys())
        if self.reuse_plans:
            print("reuse plans")
            print(self.planned)
            print(source in self.planned)
            if source in self.planned:
                missing = [k for k in self.dict_planner.keys() if k not in self.planned[source]]
        if self.verbose:
            print("Missing, first pass", missing)
        list_results = self.launch_things(lambda x: (
            x, self.dict_planner[x].solve(from_observation=Memory([source], maxlen=1), verbose=self.verbose,
                                          render=False)), missing)
        for l in list_results:
            cost = l[1][0]
            action = self.dict_planner[l[0]].get_next_action(Memory([source]))
            self.action_by_scenario[source][l[0]] = action
            self.q_values_scenar[source][action][l[0]] = cost
            self.planned[source][l[0]] = True
            if source not in self.q_values:
                self.q_values[source] = {}
            if action not in self.q_values[source]:
                self.q_values[source][action] = 0.
            self.q_values[source][action] += cost * self.weight_scenario[l[0]]
            self.plan_by_scenario[source][l[0]] = l[1][1]

    def second_pass(self, source):
        list_action = set([self.action_by_scenario[source][k]
                           for k in self.action_by_scenario[source]])
        to_do = []
        for action in list_action:
            for key in self.dict_planner:
                if key not in self.q_values_scenar[source][action]:
                    to_do += [(source, key, action)]
        if self.verbose:
            print("to do second pass, ", to_do)
        l = self.launch_things(lambda x: self.look_one_step_ahead(x[0], x[1], x[2]),
                               to_do)
        for m in l:
            pass

    def look_one_step_ahead(self, source, key_scenario, action):
        next_state = self.dict_planner[key_scenario]._domain.get_next_state(Memory([source]),
                                                                            action)
        cost = self.dict_planner[key_scenario]._domain.get_transition_value(Memory([source]), action,
                                                                            next_state).cost
        cost_f, path_f = self.dict_planner[key_scenario].solve(from_observation=Memory([next_state], maxlen=1),
                                                               verbose=self.verbose, render=False)
        self.q_values_scenar[source][action][key_scenario] = cost + cost_f
        if self.reuse_plans:
            self.plan_by_scenario[next_state][key_scenario] = path_f
        action_next_state = self.dict_planner[key_scenario].get_next_action(Memory([next_state]))
        if self.reuse_plans:
            self.action_by_scenario[next_state][key_scenario] = action_next_state
        self.q_values_scenar[next_state][action_next_state][key_scenario] = cost_f
        if next_state not in self.q_values:
            self.q_values[next_state] = {}
        if action_next_state not in self.q_values[next_state] and self.reuse_plans:
            self.q_values[next_state][action_next_state] = 0.
        if self.reuse_plans:
            self.q_values[next_state][action_next_state] += cost_f * self.weight_scenario[key_scenario]
        if self.reuse_plans:
            self.planned[next_state][key_scenario] = True
        self.q_values[source][action] += (cost + cost_f) * self.weight_scenario[key_scenario]
        return None
