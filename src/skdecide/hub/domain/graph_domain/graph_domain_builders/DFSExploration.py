# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

from skdecide import DeterministicPlanningDomain, Memory
from skdecide.hub.domain.graph_domain.graph_domain_builders.GraphExploration import (
    GraphExploration,
)
from skdecide.hub.domain.graph_domain.GraphDomain import GraphDomain


class DFSExploration(GraphExploration):
    """Depth first search based method storing graph search into a GraphDomain object,
    Main interest of using DFS is that it is possible to limit number of edges and paths to goal if necessary.
    """

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

    def build_graph_domain(
        self, init_state: Any = None, transition_extractor=None, verbose=True
    ) -> GraphDomain:
        if transition_extractor is None:
            transition_extractor = lambda s, a, s_prime: {
                "cost": self.domain.get_transition_value(s, a, s_prime).cost
            }
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
        paths_dict = {}
        targets = set()
        while stack:
            (vertex, path) = stack.pop()
            actions = self.domain.get_applicable_actions(vertex).get_elements()
            for action in actions:
                next = self.domain.get_next_state(vertex, action)
                if action not in next_state_map[vertex]:
                    nb_edges += 1
                else:
                    continue
                next_state_map[vertex][action] = next
                next_state_attributes[vertex][action] = transition_extractor(
                    vertex, action, next
                )
                if self.domain.is_goal(next):
                    targets.add(next)
                    nb_path += 1
                    if verbose:
                        print(nb_path, " / ", self.max_path)
                        print("nodes  ", nb_nodes, " / ", self.max_nodes)
                        print("edges  ", nb_edges, " / ", self.max_edges)
                else:
                    if next not in next_state_map:
                        stack.append((next, path + [next]))
                        paths_dict[next] = set(tuple(path + [next]))
                if next not in next_state_map:
                    next_state_map[next] = {}
                    next_state_attributes[next] = {}
                    nb_nodes += 1
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
