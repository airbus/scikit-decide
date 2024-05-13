# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Any, Dict, Optional, Tuple

from skdecide import D, GoalMDPDomain
from skdecide.hub.domain.graph_domain.graph_domain_builders.GraphExploration import (
    GraphExploration,
)
from skdecide.hub.domain.graph_domain.GraphDomain import GraphDomainUncertain

# WARNING : adapted for the scheduling domains.


class DFSExploration(GraphExploration):
    """DFS based exploration for MDP domains"""

    def __init__(
        self,
        domain: GoalMDPDomain,
        score_function=None,
        max_edges: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_path: Optional[int] = None,
    ):
        self.domain = domain
        self.score_function = score_function
        self.c = count()
        if score_function is None:
            self.score_function = lambda s: (next(self.c))
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_path = max_path
        if self.max_edges is None:
            self.max_edges = float("inf")
        if self.max_nodes is None:
            self.max_nodes = float("inf")
        if self.max_path is None:
            self.max_path = float("inf")

    def build_graph_domain(self, init_state: Any = None) -> GraphDomainUncertain:
        if init_state is None:
            initial_state = self.domain.get_initial_state()
        else:
            initial_state = init_state
        stack = [(self.score_function(initial_state), initial_state)]
        domain = self.domain
        goal_states = set()
        terminal_states = set()
        num_s = 0
        state_to_ind = {}
        nb_states = 1
        nb_edges = 0
        result = {initial_state}
        next_state_map: Dict[
            D.T_state, Dict[D.T_event, Dict[D.T_state, Tuple[float, float]]]
        ] = {}
        state_terminal: Dict[D.T_state, bool] = dict()
        state_goal: Dict[D.T_state, bool] = dict()
        state_terminal[initial_state] = self.domain.is_terminal(initial_state)
        state_goal[initial_state] = self.domain.is_goal(initial_state)
        while len(stack) > 0:
            if not len(result) % 100 and len(result) > nb_states:
                print("Expanded {} states.".format(len(result)))
                nb_states = len(result)
            _, s = heappop(stack)
            if s not in state_to_ind:
                state_to_ind[s] = num_s
                num_s += 1
            if domain.is_terminal(s):
                terminal_states.add(s)
            if domain.is_goal(s):
                goal_states.add(s)
            if domain.is_goal(s) or domain.is_terminal(s):
                continue
            actions = domain.get_applicable_actions(s).get_elements()
            for action in actions:
                successors = domain.get_next_state_distribution(s, action).get_values()
                for succ, prob in successors:
                    if s not in next_state_map:
                        next_state_map[s] = {}
                    if action not in next_state_map[s]:
                        next_state_map[s][action] = {}
                    if prob != 0 and succ not in result:
                        nb_states += 1
                        nb_edges += 1
                        result.add(succ)
                        heappush(stack, (self.score_function(succ), succ))
                        cost = domain.get_transition_value(s, action, succ)
                        next_state_map[s][action][succ] = (prob, cost.cost)
                        state_goal[succ] = domain.is_goal(succ)
                        state_terminal[succ] = domain.is_terminal(succ)
            if (nb_states > self.max_nodes) or (nb_edges > self.max_edges):
                break
        return GraphDomainUncertain(
            next_state_map=next_state_map,
            state_terminal=state_terminal,
            state_goal=state_goal,
        )
