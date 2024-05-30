# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Tuple

import networkx as nx


class Graph:
    def __init__(
        self,
        nodes: List[Tuple[Hashable, Dict[str, Any]]],
        edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]],
        undirected=True,
    ):
        self.nodes = nodes
        self.edges = edges
        self.undirected = undirected
        self.neighbors_dict = {}
        self.predecessors_dict = {}
        self.edges_infos_dict = {}
        self.nodes_infos_dict = {}
        self.build_nodes_infos_dict()
        self.build_edges()
        self.nodes_name = sorted(self.nodes_infos_dict)
        self.graph_nx = self.to_networkx()

    def get_edges(self):
        return self.edges_infos_dict.keys()

    def get_nodes(self):
        return self.nodes_name

    def build_nodes_infos_dict(self):
        for n, d in self.nodes:
            self.nodes_infos_dict[n] = d

    def build_edges(self):
        for n1, n2, d in self.edges:
            self.edges_infos_dict[(n1, n2)] = d
            if n2 not in self.predecessors_dict:
                self.predecessors_dict[n2] = set()
            if n1 not in self.neighbors_dict:
                self.neighbors_dict[n1] = set()
            self.predecessors_dict[n2].add(n1)
            self.neighbors_dict[n1].add(n2)
            if self.undirected:
                if n1 not in self.predecessors_dict:
                    self.predecessors_dict[n1] = set()
                if n2 not in self.neighbors_dict:
                    self.neighbors_dict[n2] = set()
                self.predecessors_dict[n1].add(n2)
                self.neighbors_dict[n2].add(n1)
                self.edges_infos_dict[(n2, n1)] = d

    def get_neighbors(self, node):
        return self.neighbors_dict.get(node, [])

    def get_predecessors(self, node):
        return self.predecessors_dict.get(node, [])

    def get_attr_node(self, node, attr):
        return self.nodes_infos_dict.get(node, {}).get(attr, None)

    def get_attr_edge(self, node1, node2, attr):
        return self.edges_infos_dict.get((node1, node2), {}).get(attr, None)

    def to_networkx(self):
        graph_nx = nx.DiGraph() if not self.undirected else nx.Graph()
        graph_nx.add_nodes_from(self.nodes)
        graph_nx.add_edges_from(self.edges)
        return graph_nx

    def check_loop(self):
        try:
            cycles = nx.find_cycle(self.graph_nx, orientation="original")
        except:
            cycles = None
        return cycles

    def precedessors_nodes(self, n):
        return nx.algorithms.ancestors(self.graph_nx, n)

    def ancestors_map(self):
        return {
            n: nx.algorithms.ancestors(self.graph_nx, n) for n in self.graph_nx.nodes()
        }

    def descendants_map(self):
        return {
            n: nx.algorithms.descendants(self.graph_nx, n)
            for n in self.graph_nx.nodes()
        }

    def successors_map(self):
        return {n: list(nx.neighbors(self.graph_nx, n)) for n in self.graph_nx.nodes()}

    def predecessors_map(self):
        return {n: list(self.graph_nx.predecessors(n)) for n in self.graph_nx.nodes()}
