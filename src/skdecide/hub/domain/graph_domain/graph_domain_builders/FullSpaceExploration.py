# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

from skdecide import DeterministicPlanningDomain
from skdecide.hub.domain.graph_domain.graph_domain_builders.GraphExploration import (
    GraphExploration,
)
from skdecide.hub.domain.graph_domain.GraphDomain import GraphDomain


class FullSpaceExploration(GraphExploration):
    """Exhaustive computation of deterministic domain transitions to build a GraphDomain from it."""

    def __init__(
        self,
        domain: DeterministicPlanningDomain,
        max_edges: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_path: Optional[int] = None,
    ):
        self.domain = domain
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_path = max_path
        if self.max_edges is None:
            self.max_edges = float("inf")
        if self.max_nodes is None:
            self.max_nodes = float("inf")
        if self.max_path is None:
            self.max_path = float("inf")

    def build_graph_domain(self, init_state: Any = None) -> GraphDomain:
        next_state_map = {}
        next_state_attributes = {}
        if init_state is None:
            init_state = self.domain.get_initial_state()
        stack = [(init_state, [init_state])]
        nb_nodes = 1
        nb_edges = 0
        nb_path = 0
        next_state_map[init_state] = {}
        next_state_attributes[init_state] = {}
        targets = set()
        while stack:
            (vertex, path) = stack.pop()
            actions = self.domain.get_applicable_actions(vertex).get_elements()
            for action in actions:
                next = self.domain.get_next_state(vertex, action)
                if next not in next_state_map:
                    next_state_map[next] = {}
                    next_state_attributes[next] = {}
                    stack.append((next, path + [next]))
                    nb_nodes += 1
                if action not in next_state_map[vertex]:
                    nb_edges += 1
                next_state_map[vertex][action] = next
                next_state_attributes[vertex][action] = {
                    "cost": self.domain.get_transition_value(vertex, action, next).cost,
                    "reward": self.domain.get_transition_value(
                        vertex, action, next
                    ).reward,
                }
                if self.domain.is_goal(next):
                    nb_path += 1
                    targets.add(next)
            if (
                nb_path > self.max_path
                or (nb_nodes > self.max_nodes and nb_path >= 1)
                or (nb_edges > self.max_edges and nb_path >= 1)
            ):
                break
        return GraphDomain(
            next_state_map,
            next_state_attributes,
            targets=targets,
            attribute_weight="cost",
        )


def reachable_states(self, s0: Any):
    """Computes all states reachable from s0."""
    result = {s0}
    stack = [s0]
    domain = self._domain
    while len(stack) > 0:
        if not len(result) % 100:
            print("Expanded {} states.".format(len(result)))
        s = stack.pop()
        if domain.is_terminal(s):
            continue
        # Add successors
        actions = domain.get_applicable_actions(s).get_elements()
        for action in actions:
            successors = domain.get_next_state_distribution(s, action).get_values()
            for succ, prob in successors:
                if prob != 0 and succ not in result:
                    result.add(succ)
                    stack.append(succ)
    return result
