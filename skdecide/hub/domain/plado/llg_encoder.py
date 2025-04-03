import logging
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional, Union

import gymnasium as gym
import networkx as nx
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import plado
except ImportError:
    plado_available = False
    logger.warning(
        "You need to install plado library to use PladoPddlDomain or PladoPPddlDomain!"
    )
    from fractions import Fraction

    Float = Fraction
else:
    plado_available = True
    from plado.semantics.task import (
        AddEffect,
        Atom,
        DelEffect,
        SimpleCondition,
        State,
        Task,
    )


class EdgeLabel(Enum):
    NU = "nu"
    GAMMA = "gamma"
    PRE = "pre"
    ADD = "add"
    DEL = "del"


map_edgelabel2int = {label: idx for idx, label in enumerate(EdgeLabel)}
map_int2edgelabel = list(EdgeLabel)


class NodeLabel(Enum):
    PREDICATE = "predicate"
    OBJECT = "object"
    ACTION = "action"
    STATE = "state"
    GOAL = "goal"
    STATIC = "static"


map_nodelabel2int = {label: idx for idx, label in enumerate(NodeLabel)}
map_int2nodelabel = list(NodeLabel)


class IndexFunctionType(Enum):
    ONEHOT = "one_hot_encoding"
    RANDSPHERE = "random_injection_to_sphere"


class LLGEncoder:
    """Lifted Learning Graph encoder for plado state.

    This encodes the current state of a PDDL domain/problem pair into a LLG similar to what is presented in

    Chen, D. Z., Thiébaux, S., & Trevizan, F. (2024).
    Learning Domain-Independent Heuristics for Grounded and Lifted Planning.
    Proceedings of the AAAI Conference on Artificial Intelligence, 38(18), 20078-20086.
    https://doi.org/10.1609/aaai.v38i18.29986

    """

    # color used for nodes and edges when plotting
    map_edgelabel2color = {
        EdgeLabel.NU: "grey",
        EdgeLabel.ADD: "b",
        EdgeLabel.DEL: "r",
        EdgeLabel.PRE: "k",
        EdgeLabel.GAMMA: "g",
    }
    map_nodelabel2color = {
        NodeLabel.ACTION: "#ff8000",
        NodeLabel.GOAL: "#ffed00",
        NodeLabel.STATE: "#00cc99",
        NodeLabel.PREDICATE: "#f5baff",
        NodeLabel.OBJECT: "#8acff0",
        None: "#b1b0ae",
    }

    edges_dtype = np.int8

    def __init__(
        self,
        task: Task,
        index_function_type: IndexFunctionType = IndexFunctionType.ONEHOT,
        index_function_default_dim: int = 2,
        cost_functions: Optional[set[int]] = None,
    ):
        self.index_function_default_dim = index_function_default_dim
        self.index_function_type = index_function_type
        self.task = task
        if cost_functions is None:
            total_cost: Optional[int] = None
            for i, f in enumerate(self.task.functions):
                if f.name == "total-cost":
                    self.total_cost = i
                    break
            if total_cost is None:
                self.cost_functions = set()
            else:
                self.cost_functions = {total_cost}
        else:
            self.cost_functions = cost_functions

        self._init_graph()

    def encode(self, state: State) -> gym.spaces.GraphInstance:
        """Encode plado state into an LLG."""
        self._encode_state(state=state)
        return self._add_endoded_state_to_gym_graph()

    def decode(self, graph: gym.spaces.GraphInstance) -> State:
        """Decode an LLG into a plado state.

        Here we use the fact that the graph from this LLGEncoder.encode().
        So
        - we know that nodes are well ordered:
            - predicate node before atom arg nodes
            - first atom arg before second atom arg
        - we now already the correspondance between nodes and predicates/objects

        If these hypotheses are not valid, rather use `decode_llg()` which makes less hypotheses,
        but could be less efficient.

        """
        state = State(
            num_predicates=self.task.num_fluent_predicates,
            num_functions=len(self.task.functions),
        )
        state_nodes = (
            graph.nodes[
                self._state_node_start :, map_nodelabel2int[NodeLabel.STATE]
            ].nonzero()[0]
            + self._state_node_start
        )
        state_encoding_edge_links = graph.edge_links[self._state_edge_start :, :]
        for state_node in state_nodes:
            state_node_neighbours = [
                e[1] for e in state_encoding_edge_links if e[0] == state_node
            ]
            state_node_neighbours.sort()  # to be sure to have atom predicate + args in right order
            atom_args = []
            predicate = None
            for node in state_node_neighbours:  # predicate then arg0, arg1, ...
                if predicate is None:
                    # atom predicate
                    predicate = self._map_predicate_node2id[node]
                else:
                    # atom arg
                    object_node = [
                        e[1]
                        for e in graph.edge_links
                        if e[0] == node and e[1] != state_node
                    ][0]
                    atom_args.append(self._map_object_node2id[object_node])
            state.atoms[predicate].add(tuple(atom_args))
        return state

    @property
    def graph_space(self) -> gym.spaces.Graph:
        node_low = np.array(
            [0] * len(NodeLabel) + self._index_lows, dtype=self.nodes_dtype
        )
        node_high = np.array(
            [1] * len(NodeLabel) + self._index_highs, dtype=self.nodes_dtype
        )
        edge_low = np.zeros((len(EdgeLabel),), dtype=self.edges_dtype)
        edge_high = np.ones((len(EdgeLabel),), dtype=self.edges_dtype)
        return gym.spaces.Graph(
            node_space=gym.spaces.Box(
                low=node_low, high=node_high, dtype=self.nodes_dtype
            ),
            edge_space=gym.spaces.Box(
                low=edge_low, high=edge_high, dtype=self.edges_dtype
            ),
        )

    def plot(
        self,
        graph: gym.spaces.GraphInstance,
        subgraph: Optional[Any] = None,
        subgraphs: Optional[Iterable[Any]] = None,
        ax: Optional[Any] = None,
    ) -> None:
        """Plot the encoding graph (or a subgraph of it)

        Args:
            graph: encoding llg
            subgraph: subgraph id (action or predicate)
            ax: matplotlib axes in which plot the graph

        Returns:

        """
        node_labels, node_color, edge_color = self._prepare_for_plot(graph)
        G = nx.Graph()
        if subgraph is None and subgraphs is None:
            edge_links = graph.edge_links
        else:
            if subgraphs is None:
                subgraphs = [subgraph]
            elif subgraph is not None:
                subgraphs = list(subgraphs) + [subgraph]
            subgraphs_edges_idx = set(
                idx for subgraph in subgraphs for idx in self._subgraphs_edges[subgraph]
            )
            edge_links = [
                e for i, e in enumerate(graph.edge_links) if i in subgraphs_edges_idx
            ]
        G.add_edges_from(edge_links)
        nx.draw_networkx(
            G,
            labels={i: lab for i, lab in node_labels.items() if i in G.nodes},
            edge_color=[edge_color[e] for e in G.edges],
            node_color=[node_color[n] for n in G.nodes],
            ax=ax,
        )

    def _init_graph(self):
        # hyp:
        #   - no numeric fluents except for total-cost
        #   - goal: no numeric constraints, no negative atoms
        #   - actions
        #      - precondition: no numeric constraints, no negative atoms
        #      - effect: no probabilistic effect, no conditional effect, no numerical effect
        assert (
            len(self.task.functions) == len(self.cost_functions)
            and len(self.task.goal.condition.constraints) == 0
            and len(self.task.goal.condition.negated_atoms) == 0
            and all(
                len(action.precondition.constraints) == 0
                and len(action.precondition.negated_atoms) == 0
                for action in self.task.actions
            )
            # no probabilitic effects
            and all(
                [
                    all([e.num_effects == 1 for e in action.effect.effects])
                    for action in self.task.actions
                ]
            )
            # no conditional effects
            and all(
                [
                    all(
                        [
                            len(e.outcomes[0][1][0].condition.negated_atoms) == 0
                            for e in action.effect.effects
                        ]
                    )
                    for action in self.task.actions
                ]
            )
            and all(
                [
                    all(
                        [
                            len(e.outcomes[0][1][0].condition.atoms) == 0
                            for e in action.effect.effects
                        ]
                    )
                    for action in self.task.actions
                ]
            )
            and all(
                [
                    all(
                        [
                            len(e.outcomes[0][1][0].condition.constraints) == 0
                            for e in action.effect.effects
                        ]
                    )
                    for action in self.task.actions
                ]
            )
            # no numerical effect
            and all(
                [
                    all(
                        [
                            isinstance(
                                e.outcomes[0][1][0].effect, (DelEffect, AddEffect)
                            )
                            for e in action.effect.effects
                        ]
                    )
                    for action in self.task.actions
                ]
            )
        )

        self._reset_graph()

        # add only one edges (and the other way will be added at the end systematically)

        # objects
        self._object_nodes = self._get_new_node_ids(len(self.task.objects))
        self._map_object_node2id = {
            node: i for i, node in enumerate(self._object_nodes)
        }
        # predicates
        self._predicate_nodes = self._get_new_node_ids(len(self.task.predicates))
        self._map_predicate_node2id = {
            node: i for i, node in enumerate(self._predicate_nodes)
        }
        for node_pred, predicate in zip(self._predicate_nodes, self.task.predicates):
            for node_obj in self._object_nodes:
                self._add_edge(node_obj, node_pred, EdgeLabel.NU, subgraph=predicate)

        # goals
        self._goal_nodes = self._encode_condition(
            condition=self.task.goal.condition,
            edge_label=EdgeLabel.GAMMA,
            grounded=True,
            variable_nodes=self._object_nodes,
        )

        # actions
        self._action_nodes = []
        for action in self.task.actions:
            # action node
            action_node = self._get_new_node_id()
            self._action_nodes.append(action_node)
            # action arg nodes
            action_arg_nodes = self._get_new_node_ids(action.parameters)
            for action_arg_node in action_arg_nodes:
                self._add_edge(
                    action_node, action_arg_node, EdgeLabel.NU, subgraph=action
                )
            # precondition
            self._encode_condition(
                condition=action.precondition,
                edge_label=EdgeLabel.PRE,
                grounded=False,
                variable_nodes=action_arg_nodes,
                action_node=action_node,
                subgraph=action,
            )
            # add and del effects
            for e in action.effect.effects:
                atomic_effect: Union[AddEffect, DelEffect] = e.outcomes[0][1][0].effect
                atom = atomic_effect.atom
                if isinstance(atomic_effect, AddEffect):
                    edge_label = EdgeLabel.ADD
                elif isinstance(atomic_effect, DelEffect):
                    edge_label = EdgeLabel.DEL
                else:
                    raise NotImplementedError()
                self._encode_condition_atom(
                    atom=atom,
                    edge_label=edge_label,
                    grounded=False,
                    variable_nodes=action_arg_nodes,
                    action_node=action_node,
                    subgraph=action,
                )

        # static facts
        for static_predicate, predicate_atoms in enumerate(self.task.static_facts):
            predicate = (
                self.task.num_fluent_predicates
                + self.task.num_derived_predicates
                + static_predicate
            )
            for atom_args in predicate_atoms:
                self._static_state_nodes.append(
                    self._encode_condition_atom(
                        atom=Atom(
                            predicate=predicate, args=atom_args, variables=tuple()
                        ),
                        edge_label=EdgeLabel.GAMMA,
                        grounded=True,
                        variable_nodes=self._object_nodes,
                    )
                )

        # graph w/o current state encoded
        self._initial_graph = self._compute_gym_graph()

        # ready to encode state from there
        self._get_ready_for_state_encoding()

    def _compute_gym_graph(self) -> gym.spaces.GraphInstance:
        nodes = np.zeros((self._get_n_nodes(), self.nodes_dim), dtype=self.nodes_dtype)
        nodes[self._predicate_nodes, map_nodelabel2int[NodeLabel.PREDICATE]] = 1
        nodes[
            self._predicate_nodes[
                self.task.num_fluent_predicates + self.task.num_derived_predicates :
            ],
            map_nodelabel2int[NodeLabel.STATIC],
        ] = 1
        nodes[self._object_nodes, map_nodelabel2int[NodeLabel.OBJECT]] = 1
        nodes[self._action_nodes, map_nodelabel2int[NodeLabel.ACTION]] = 1
        nodes[self._goal_nodes, map_nodelabel2int[NodeLabel.GOAL]] = 1
        nodes[self._static_state_nodes, map_nodelabel2int[NodeLabel.STATE]] = 1
        nodes[self._state_nodes, map_nodelabel2int[NodeLabel.STATE]] = 1
        for node, index in self._map_node2argindex.items():
            nodes[node, -self.index_function_dim :] = self.index_function(index)
        edge_int_labels = [map_edgelabel2int[label] for label in self._edge_labels]
        edges = np.eye(len(EdgeLabel), dtype=self.edges_dtype)[edge_int_labels]
        edge_links = np.array(self._edge_links)
        return gym.spaces.GraphInstance(
            nodes=nodes,
            edges=edges,
            edge_links=edge_links,
        )

    def _add_endoded_state_to_gym_graph(self) -> gym.spaces.GraphInstance:
        nodes_state = np.zeros(
            (self._get_n_nodes() - self._state_node_start, self.nodes_dim),
            dtype=self.nodes_dtype,
        )
        nodes_state[
            np.array(self._state_nodes) - self._state_node_start,
            map_nodelabel2int[NodeLabel.STATE],
        ] = 1
        for node, index in self._map_node2argindex.items():
            if node >= self._state_node_start:
                nodes_state[
                    node - self._state_node_start, -self.index_function_dim :
                ] = self.index_function(index)
        edge_int_labels_state = [
            map_edgelabel2int[label]
            for label in self._edge_labels[self._state_edge_start :]
        ]
        edges_state = np.eye(len(EdgeLabel), dtype=self.edges_dtype)[
            edge_int_labels_state
        ]
        edge_links_state = np.array(self._edge_links[self._state_edge_start :])
        nodes = np.vstack(
            (self._initial_graph.nodes, nodes_state),
        )
        edges = np.vstack(
            (self._initial_graph.edges, edges_state),
        )
        edge_links = np.vstack(
            (self._initial_graph.edge_links, edge_links_state),
        )
        return gym.spaces.GraphInstance(
            nodes=nodes,
            edges=edges,
            edge_links=edge_links,
        )

    def index_function(
        self, x: Union[int, npt.NDArray[np.int_]]
    ) -> npt.NDArray[Union[np.int_, np.float_]]:
        """Maps an index into sphere S^T."""
        if self.index_function_type == IndexFunctionType.ONEHOT:
            return np.eye(self.index_function_dim, dtype=self.nodes_dtype)[
                x
            ]  # shape: x_shape + (if_dim,)
        else:
            raise NotImplementedError()

    def index_function_inverse(
        self, y: npt.NDArray[Union[np.int_, np.float_]]
    ) -> Union[int, npt.NDArray[np.int_]]:
        """Inverse of index function."""
        if self.index_function_type == IndexFunctionType.ONEHOT:
            assert y.shape[-1] == self.index_function_dim
            return np.argmax(y, -1)
        else:
            raise NotImplementedError()

    def _get_new_node_id(self) -> int:
        self._last_node += 1
        return self._last_node

    def _get_new_node_ids(self, n_nodes: int) -> list[int]:
        node_ids = [self._last_node + i_node + 1 for i_node in range(n_nodes)]
        self._last_node += n_nodes
        return node_ids

    def _get_n_nodes(self) -> int:
        return self._last_node + 1

    def _add_edge(
        self, node1: int, node2: int, label: EdgeLabel, subgraph: Optional[Any] = None
    ) -> None:
        self._edge_links.append((node1, node2))
        self._edge_labels.append(label)
        self._edge_links.append((node2, node1))
        self._edge_labels.append(label)
        if subgraph is not None:
            self._subgraphs_edges[subgraph].extend(
                [len(self._edge_links) - 2, len(self._edge_links) - 1]
            )

    def _reset_graph(self) -> None:
        self._last_node = -1
        self._edge_links = []
        self._edge_labels = []
        self._map_node2argindex = {}
        self._goal_nodes = []
        self._action_nodes = []
        self._predicate_nodes = []
        self._object_nodes = []
        self._state_nodes = []
        self._static_state_nodes = []
        self._subgraphs_edges: dict[Any, list[int]] = defaultdict(list)
        self._reset_index_function()
        self._ready_for_state_encoding = False

    def _get_ready_for_state_encoding(self):
        assert len(self._state_nodes) == 0
        self._state_node_start = self._get_n_nodes()
        self._state_edge_start = len(self._edge_links)
        self._state_subgraph_edge_start = defaultdict(
            lambda: 0,
            {subgraph: len(edges) for subgraph, edges in self._subgraphs_edges.items()},
        )
        self._ready_for_state_encoding = True

    def _reset_state(self):
        assert self._ready_for_state_encoding
        self._state_nodes = []
        self._last_node = self._state_node_start - 1
        self._edge_links = self._edge_links[: self._state_edge_start]
        self._edge_labels = self._edge_labels[: self._state_edge_start]
        for subgraph, edges in self._subgraphs_edges.items():
            self._subgraphs_edges[subgraph] = edges[
                : self._state_subgraph_edge_start[subgraph]
            ]
        self._map_node2argindex = {
            n: idx
            for n, idx in self._map_node2argindex.items()
            if n < self._state_node_start
        }

    def _encode_state(self, state: State) -> None:
        # forget previous state encoding
        self._reset_state()
        # encode
        for predicate, predicate_atoms in enumerate(state.atoms):
            for atom_args in predicate_atoms:
                self._state_nodes.append(
                    self._encode_condition_atom(
                        atom=Atom(
                            predicate=predicate, args=atom_args, variables=tuple()
                        ),
                        edge_label=EdgeLabel.GAMMA,
                        grounded=True,
                        variable_nodes=self._object_nodes,
                    )
                )

    def _reset_index_function(self):
        # Index Function dim
        if self.index_function_type == IndexFunctionType.ONEHOT:
            # max arity of predicates
            self.index_function_dim = max(
                len(p.parameters) for p in self.task.predicates
            )
            self._index_lows = [0] * self.index_function_dim
            self._index_highs = [1] * self.index_function_dim

        else:
            self.index_function_dim = self.index_function_default_dim
            self._index_lows = [-1.0] * self.index_function_dim
            self._index_highs = [1.0] * self.index_function_dim

    @property
    def nodes_dim(self) -> int:
        return len(NodeLabel) + self.index_function_dim

    @property
    def nodes_dtype(self) -> npt.DTypeLike:
        if self.index_function_type == IndexFunctionType.ONEHOT:
            return np.int_
        else:
            return np.float_

    def _encode_condition(
        self,
        condition: SimpleCondition,
        edge_label: EdgeLabel,
        grounded: bool,
        variable_nodes: list[int],
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> list[int]:
        scheme_pred_nodes = []
        for atom in condition.atoms:
            scheme_pred_nodes.append(
                self._encode_condition_atom(
                    atom=atom,
                    edge_label=edge_label,
                    grounded=grounded,
                    variable_nodes=variable_nodes,
                    action_node=action_node,
                    subgraph=subgraph,
                )
            )
        return scheme_pred_nodes

    def _encode_condition_atom(
        self,
        atom: Atom,
        edge_label: EdgeLabel,
        variable_nodes: list[int],
        grounded: bool,
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> int:
        if subgraph is None:
            subgraph = self.task.predicates[atom.predicate]
        scheme_pred_node = self._get_new_node_id()
        self._add_edge(
            self._predicate_nodes[atom.predicate],
            scheme_pred_node,
            edge_label,
            subgraph=subgraph,
        )
        if grounded:
            list_atom_pos_arg = list(enumerate(atom.args))
        else:
            list_atom_pos_arg = [(pos, arg) for arg, pos in atom.variables]
        if action_node is not None and len(list_atom_pos_arg) == 0:  # 0-ary predicate
            self._add_edge(scheme_pred_node, action_node, edge_label, subgraph=subgraph)
        for pos, arg in list_atom_pos_arg:
            pred_arg_node = self._get_new_node_id()
            self._map_node2argindex[pred_arg_node] = pos
            self._add_edge(
                scheme_pred_node, pred_arg_node, edge_label, subgraph=subgraph
            )
            self._add_edge(
                pred_arg_node, variable_nodes[arg], edge_label, subgraph=subgraph
            )
        return scheme_pred_node

    def _prepare_for_plot(
        self, graph: gym.spaces.GraphInstance
    ) -> tuple[dict[int, str], dict[int, str], dict[tuple[int, int], str],]:
        edge_color = {
            tuple(e): self.map_edgelabel2color[map_int2edgelabel[int(i_lab)]]
            for e, i_lab in zip(graph.edge_links, np.argmax(graph.edges, -1))
        }
        node_color = {
            node: self.map_nodelabel2color[map_int2nodelabel[int(i_lab)]]
            if has_lab
            else self.map_nodelabel2color[None]
            for node, (i_lab, has_lab) in enumerate(
                zip(
                    np.argmax(graph.nodes[:, : len(NodeLabel)], -1),
                    np.sum(graph.nodes[:, : len(NodeLabel)], -1),
                )
            )
        }
        node_labels_np = np.full(shape=(len(graph.nodes),), fill_value="", dtype=object)
        node_labels_np[
            graph.nodes[:, map_nodelabel2int[NodeLabel.GOAL]].nonzero()[0]
        ] = "goal"
        node_labels_np[
            graph.nodes[:, map_nodelabel2int[NodeLabel.STATE]].nonzero()[0]
        ] = "state"
        try:
            argindex_nodes_features = graph.nodes[:, -self.index_function_dim :]
            argindex_rows = np.sum(argindex_nodes_features, -1).nonzero()[0]
            node_labels_np[argindex_rows] = self.index_function_inverse(
                argindex_nodes_features[argindex_rows]
            )
        except NotImplementedError:
            for node, idx in self._map_node2argindex.items():
                node_labels_np[node] = idx
        for i, node in enumerate(self._action_nodes):
            node_labels_np[node] = self.task.actions[i].name
            node_color[node] = self.map_nodelabel2color[NodeLabel.ACTION]
        for i, node in enumerate(self._predicate_nodes):
            node_labels_np[node] = self.task.predicates[i].name
            node_color[node] = self.map_nodelabel2color[NodeLabel.PREDICATE]
        for i, node in enumerate(self._object_nodes):
            node_labels_np[node] = self.task.objects[i]
            node_color[node] = self.map_nodelabel2color[NodeLabel.OBJECT]
        node_labels: dict[int, str] = dict(enumerate(node_labels_np))

        return node_labels, node_color, edge_color

    def _prepare_for_plot_intern_encoding(
        self,
    ) -> tuple[dict[int, str], dict[int, str], dict[tuple[int, int], str],]:
        edge_color = {
            e: self.map_edgelabel2color[lab]
            for e, lab in zip(self._edge_links, self._edge_labels)
        }
        node_labels = {}
        node_color = defaultdict(lambda: self.map_nodelabel2color[None])
        for i, node in enumerate(self._action_nodes):
            node_labels[node] = self.task.actions[i].name
            node_color[node] = self.map_nodelabel2color[NodeLabel.ACTION]
        for i, node in enumerate(self._predicate_nodes):
            node_labels[node] = self.task.predicates[i].name
            node_color[node] = self.map_nodelabel2color[NodeLabel.PREDICATE]
        for i, node in enumerate(self._object_nodes):
            node_labels[node] = self.task.objects[i]
            node_color[node] = self.map_nodelabel2color[NodeLabel.OBJECT]
        for node in self._goal_nodes:
            node_labels[node] = "goal"
            node_color[node] = self.map_nodelabel2color[NodeLabel.GOAL]
        for node in self._state_nodes + self._static_state_nodes:
            node_labels[node] = "state"
            node_color[node] = self.map_nodelabel2color[NodeLabel.STATE]
        for node, idx in self._map_node2argindex.items():
            node_labels[node] = idx

        return node_labels, node_color, edge_color

    def plot_intern_encoding(
        self, subgraph: Optional[Any] = None, ax: Optional[Any] = None
    ):
        node_labels, node_color, edge_color = self._prepare_for_plot_intern_encoding()
        G = nx.Graph()
        if subgraph is None:
            edge_links = self._edge_links
        else:
            edge_links = [
                e
                for i, e in enumerate(self._edge_links)
                if i in self._subgraphs_edges[subgraph]
            ]
        G.add_edges_from(edge_links)
        nx.draw_networkx(
            G,
            labels={i: lab for i, lab in node_labels.items() if i in G.nodes},
            edge_color=[edge_color[e] for e in G.edges],
            node_color=[node_color[n] for n in G.nodes],
            ax=ax,
        )


def decode_llg(graph: gym.spaces.GraphInstance) -> State:
    """Decode a llg graph into a plado state without information about the plado.Task.

    This may be less efficient than LLGEncoder.decode() but is self-sufficient.

    """
    object_nodes = graph.nodes[:, map_nodelabel2int[NodeLabel.OBJECT]].nonzero()[0]
    map_object_node2id = {node: i for i, node in enumerate(object_nodes)}
    predicate_nodes = (
        graph.nodes[:, map_nodelabel2int[NodeLabel.PREDICATE]]  # predicate
        * (1 - graph.nodes[:, map_nodelabel2int[NodeLabel.STATIC]])  # not static
    ).nonzero()[0]
    map_predicate_node2id = {node: i for i, node in enumerate(predicate_nodes)}
    state_nodes = graph.nodes[:, map_nodelabel2int[NodeLabel.STATE]].nonzero()[0]
    state = State(num_predicates=len(predicate_nodes), num_functions=0)
    for state_node in state_nodes:
        state_node_neighbours = [e[1] for e in graph.edge_links if e[0] == state_node]
        state_node_neighbours.sort()
        atom_args = []
        predicate = None
        for node in state_node_neighbours:
            try:
                # atom predicate
                predicate = map_predicate_node2id[node]
            except KeyError:
                # atom arg
                object_node = [
                    e[1]
                    for e in graph.edge_links
                    if e[0] == node and e[1] != state_node
                ][0]
                atom_args.append(map_object_node2id[object_node])
        if predicate is not None:  # can be None if static
            state.atoms[predicate].add(tuple(atom_args))
    return state
