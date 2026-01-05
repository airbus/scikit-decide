from __future__ import annotations

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
    from plado.datalog.numeric import (
        Addition,
        BinaryOperation,
        Constant,
        Division,
        Fluent,
        Multiplication,
        NumericConstraint,
        NumericExpression,
        Subtraction,
    )
    from plado.semantics.task import (
        Action,
        AddEffect,
        Atom,
        DelEffect,
        Float,
        NumericEffect,
        SimpleCondition,
        State,
        Task,
    )
except ImportError:
    plado_available = False
    logger.warning(
        "You need to install plado library to use PladoPddlDomain or PladoPPddlDomain!"
    )
    from fractions import Fraction

    Float = Fraction
else:
    plado_available = True


class EdgeLabel(Enum):
    NU = "nu"  # link between predicate/fluent/action and its variables
    GAMMA = "gamma"  # state or goal encoding
    PRE = "pre"  # action precondition encoding
    EFFECT = "effect"  # action effect encoding


map_edgelabel2int = {label: idx for idx, label in enumerate(EdgeLabel)}
map_int2edgelabel = list(EdgeLabel)


class NodeLabel(Enum):
    PREDICATE = "predicate"  # predicate type node
    FLUENT = "fluent"  # fluent type node
    OBJECT = "object"  # object node
    ACTION = "action"  # action type node
    STATE = "state"  # current state atom/fluent
    GOAL = "goal"  # goal atom/constraint on fluent to reach
    STATIC = "static"  # not updated by an action
    NUMERIC = "numeric"  # contains a numeric value
    NEGATED = "negated"  # negation of an atom
    LESS = "less"  # comparator encoding for < and <=
    EQUAL = "equal"  # comparator encoding for <=, == et >=
    GREATER = "greater"  # comparator encoding for > and >=
    PLUS = "+"
    MINUS = "-"
    DIVIDE = "/"
    MULTIPLY = "*"
    ASSIGN = "assign"  # operator assigning arg 1 (numeric value) to arg 0 (numeric predicate)


map_int2nodelabel = list(NodeLabel)
map_nodelabel2int = {label: idx for idx, label in enumerate(map_int2nodelabel)}
value_index = len(map_int2nodelabel)


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
        EdgeLabel.EFFECT: "b",
        EdgeLabel.PRE: "k",
        EdgeLabel.GAMMA: "g",
    }
    map_nodelabel2color = defaultdict(
        lambda: "#b1b0ae",
        {
            NodeLabel.ACTION: "#ff8000",
            NodeLabel.GOAL: "#ffed00",
            NodeLabel.STATE: "#00cc99",
            NodeLabel.PREDICATE: "#f5baff",
            NodeLabel.FLUENT: "#f5baff",
            NodeLabel.OBJECT: "#8acff0",
        },
    )

    edges_dtype = np.int8

    def __init__(
        self,
        task: Task,
        index_function_type: IndexFunctionType = IndexFunctionType.ONEHOT,
        index_function_default_dim: int = 2,
        cost_functions: Optional[set[int]] = None,
        encode_actions: bool = False,
        simplify_encoding: bool = True,
        encode_static_facts: bool = True,
    ):
        """

        Args:
            task:
            index_function_type:
            index_function_default_dim:
            cost_functions:
            encode_actions: decide whether to generate the subgraph dedicated to action encoding
            simplify_encoding: decrease node and edge features dimension by dropping labels never used by the graph
                (e.g.: if no action encoding, drop EdgeLabel.EFFECT and EdgeLabel.PRE, NodeLabel.ACTION, NodeLabel.ASSIGN, ...,
                 if no fluents defined, drop NodeLabel.NUMERIC, NodeLabel.GREATER, NodeLabel.PLUS, ...)
                NB: the encoding will be domain dependent.
            encode_static_facts: whether to encode the static facts or not.

        """
        if encode_actions:
            assert encode_static_facts, (
                "encode_static_facts must be True if encode_actions is True."
            )
        self.encode_static_facts = encode_static_facts
        self.encode_actions = encode_actions
        self.simplify_encoding = simplify_encoding
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

    @property
    def predicates_to_encode(self):
        """Subset of the task.predicates sufficient to encode."""
        if self.encode_static_facts:
            return self.task.predicates
        else:
            return self.task.predicates[
                : self.task.num_fluent_predicates + self.task.num_derived_predicates
            ]

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
                self._state_node_start :, self.map_nodelabel2int[NodeLabel.STATE]
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
            function_id = None
            if self.has_value_column:
                value = Float(graph.nodes[state_node, self.value_column_index])
            else:
                value = Float(0)
            for (
                node
            ) in state_node_neighbours:  # predicate/function then arg0, arg1, ...
                if predicate is None and function_id is None:
                    try:
                        # atom predicate
                        predicate = self._map_predicate_node2id[node]
                    except KeyError:
                        # fluent function
                        function_id = self._map_function_node2id[node]
                else:
                    # atom arg
                    object_node = [
                        e[1]
                        for e in graph.edge_links
                        if e[0] == node and e[1] != state_node
                    ][0]
                    atom_args.append(self._map_object_node2id[object_node])
            if predicate is not None:
                state.atoms[predicate].add(tuple(atom_args))
            else:
                state.fluents[function_id][tuple(atom_args)] = value
        # add total-cost = 0.
        for function_id in self.cost_functions:
            state.fluents[function_id][tuple()] = Float(0)
        return state

    @property
    def graph_space(self) -> gym.spaces.Graph:
        node_low_list = [0] * len(self.map_int2nodelabel)
        node_high_list = [1] * len(self.map_int2nodelabel)
        if self.has_value_column:
            if self.has_fluents():
                value_low = -np.inf
                value_high = np.inf
            else:
                value_low = 0
                value_high = 0
            node_low_list.append(value_low)
            node_high_list.append(value_high)
        node_low_list += self._index_lows
        node_high_list += self._index_highs
        node_low = np.array(
            node_low_list,
            dtype=self.nodes_dtype,
        )
        node_high = np.array(
            node_high_list,
            dtype=self.nodes_dtype,
        )
        edge_low = np.zeros((self.edges_dim,), dtype=self.edges_dtype)
        edge_high = np.ones((self.edges_dim,), dtype=self.edges_dtype)
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

    def _check_actions_hypotheses(self):
        # hyp:
        #   - actions
        #      - effect: no probabilistic effect, no conditional effect
        assert (
            # no probabilistic effects
            all(
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
        )

    def _init_graph(self):
        self._check_actions_hypotheses()
        self._reset_graph()

        # add only one edges (and the other way will be added at the end systematically)

        # objects
        self._object_nodes = self._get_new_node_ids(len(self.task.objects))
        self._map_object_node2id = {
            node: i for i, node in enumerate(self._object_nodes)
        }
        # predicates
        self._predicate_nodes = self._get_new_node_ids(len(self.predicates_to_encode))
        self._map_predicate_node2id = {
            node: i for i, node in enumerate(self._predicate_nodes)
        }
        for node_pred, predicate in zip(
            self._predicate_nodes, self.predicates_to_encode
        ):
            for node_obj in self._object_nodes:
                self._add_edge(node_obj, node_pred, EdgeLabel.NU, subgraph=predicate)

        # fluents
        self._function_nodes = self._get_new_node_ids(len(self.task.functions))
        self._map_function_node2id = {
            node: i for i, node in enumerate(self._function_nodes)
        }
        for node_fluent, fluent in zip(self._function_nodes, self.task.functions):
            for node_obj in self._object_nodes:
                self._add_edge(node_obj, node_fluent, EdgeLabel.NU, subgraph=fluent)

        # goals
        self._goal_nodes = self._encode_condition(
            condition=self.task.goal.condition,
            edge_label=EdgeLabel.GAMMA,
            grounded=True,
            variable_nodes=self._object_nodes,
        )

        if self.encode_actions:
            # actions
            for action in self.task.actions:
                self._encode_action(action)

        if self.encode_static_facts:
            # static facts
            for static_predicate, predicate_atoms in enumerate(self.task.static_facts):
                predicate = (
                    self.task.num_fluent_predicates
                    + self.task.num_derived_predicates
                    + static_predicate
                )
                for atom_args in predicate_atoms:
                    self._static_state_nodes.append(
                        self._encode_atom(
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
        nodes[self._predicate_nodes, self.map_nodelabel2int[NodeLabel.PREDICATE]] = 1
        nodes[self._function_nodes, self.map_nodelabel2int[NodeLabel.FLUENT]] = 1
        for node, comparators in self._map_node2comparator.items():
            for comparator in comparators:
                nodes[node, self.map_nodelabel2int[comparator]] = 1
        for node, value in self._map_node2value.items():
            nodes[node, self.value_column_index] = value
            nodes[node, self.map_nodelabel2int[NodeLabel.NUMERIC]] = 1
        for node, operator in self._map_node2operator.items():
            nodes[node, self.map_nodelabel2int[operator]] = 1
        nodes[
            self._predicate_nodes[
                self.task.num_fluent_predicates + self.task.num_derived_predicates :
            ],
            self.map_nodelabel2int[NodeLabel.STATIC],
        ] = 1
        nodes[self._negated_atom_nodes, self.map_nodelabel2int[NodeLabel.NEGATED]] = 1
        nodes[self._object_nodes, self.map_nodelabel2int[NodeLabel.OBJECT]] = 1
        nodes[self._action_nodes, self.map_nodelabel2int[NodeLabel.ACTION]] = 1
        nodes[self._goal_nodes, self.map_nodelabel2int[NodeLabel.GOAL]] = 1
        nodes[self._static_state_nodes, self.map_nodelabel2int[NodeLabel.STATE]] = 1
        nodes[self._state_nodes, self.map_nodelabel2int[NodeLabel.STATE]] = 1
        for node, index in self._map_node2argindex.items():
            nodes[node, -self.index_function_dim :] = self.index_function(index)
        edges = self._encode_edge_features(self._edge_labels)
        edge_links = np.array(self._edge_links)
        if self.simplify_encoding:
            # keep only necessary node features
            necessary_node_labels = {
                self.map_int2nodelabel[i_label]
                for i_label in nodes[:, : self._n_node_labels].sum(axis=0).nonzero()[0]
            }
            # do not forget labels necessary for encoding state
            necessary_node_labels.add(NodeLabel.STATE)
            if self.has_fluents():
                necessary_node_labels.add(NodeLabel.NUMERIC)
            # columns to keep (old indices)
            nodes_features_dim_old = (
                self.nodes_dim
            )  # will change with has_value_column and map_int2nodelabel
            indices_nodes_columns_to_keep = [
                i_label
                for i_label, label in enumerate(self.map_int2nodelabel)
                if label in necessary_node_labels
            ]
            # keep value column?
            self.has_value_column = NodeLabel.NUMERIC in necessary_node_labels
            if self.has_value_column:
                indices_nodes_columns_to_keep.append(self.value_column_index)
            # index function
            indices_nodes_columns_to_keep += list(
                range(
                    nodes_features_dim_old - self.index_function_dim,
                    nodes_features_dim_old,
                )
            )

            # update nodes by dropping not used features
            nodes = nodes[:, indices_nodes_columns_to_keep]

            # update maps idx<->feature (keep same order as before)
            self.map_int2nodelabel = [
                label
                for label in self.map_int2nodelabel
                if label in necessary_node_labels
            ]
            self.map_nodelabel2int = {
                label: idx for idx, label in enumerate(self.map_int2nodelabel)
            }

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
            self.map_nodelabel2int[NodeLabel.STATE],
        ] = 1
        for node, value in self._map_node2value.items():
            if node >= self._state_node_start:
                nodes_state[node - self._state_node_start, self.value_column_index] = (
                    value
                )
                nodes_state[
                    node - self._state_node_start,
                    self.map_nodelabel2int[NodeLabel.NUMERIC],
                ] = 1
        for node, index in self._map_node2argindex.items():
            if node >= self._state_node_start:
                nodes_state[
                    node - self._state_node_start, -self.index_function_dim :
                ] = self.index_function(index)
        edges_state = self._encode_edge_features(
            self._edge_labels[self._state_edge_start :]
        )
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

    @property
    def edges_dim(self) -> int:
        if self.encode_actions or not self.simplify_encoding:
            return len(EdgeLabel)
        else:
            return 1

    def _encode_edge_features(self, edge_labels: list[EdgeLabel]) -> np.ndarray:
        if self.encode_actions or not self.simplify_encoding:
            edge_int_labels_state = [map_edgelabel2int[label] for label in edge_labels]
            return np.eye(self.edges_dim, dtype=self.edges_dtype)[edge_int_labels_state]
        else:
            return np.array(
                tuple(1 if label == EdgeLabel.GAMMA else 0 for label in edge_labels),
                dtype=self.edges_dtype,
            ).reshape((len(edge_labels), self.edges_dim))

    def index_function(
        self, x: Union[int, npt.NDArray[np.int_]]
    ) -> npt.NDArray[Union[np.int_, np.float64]]:
        """Maps an index into sphere S^T."""
        if self.index_function_type == IndexFunctionType.ONEHOT:
            return np.eye(self.index_function_dim, dtype=self.nodes_dtype)[
                x
            ]  # shape: x_shape + (if_dim,)
        else:
            raise NotImplementedError()

    def index_function_inverse(
        self, y: npt.NDArray[Union[np.int_, np.float64]]
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
        self._map_node2operator: dict[int, NodeLabel] = {}
        self._map_node2comparator: dict[int, list[NodeLabel]] = {}
        self._goal_nodes = []
        self._action_nodes = []
        self._predicate_nodes = []
        self._function_nodes = []
        self._object_nodes = []
        self._state_nodes = []
        self._map_node2value: dict[int, float] = {}
        self._static_state_nodes = []
        self._negated_atom_nodes = []
        self._subgraphs_edges: dict[Any, list[int]] = defaultdict(list)
        self._reset_index_function()
        self._ready_for_state_encoding = False
        self.has_value_column = True
        self.map_nodelabel2int = map_nodelabel2int
        self.map_int2nodelabel = map_int2nodelabel

    @property
    def value_column_index(self):
        """Index of node value among node features"""
        return self._n_node_labels

    @property
    def _n_node_labels(self):
        """Nb of node labels used by the graph."""
        return len(self.map_int2nodelabel)

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
        self._map_node2value = {
            n: idx
            for n, idx in self._map_node2value.items()
            if n < self._state_node_start
        }

    def _encode_state(self, state: State) -> None:
        # forget previous state encoding
        self._reset_state()
        # encode atoms
        for predicate, predicate_atoms in enumerate(state.atoms):
            for atom_args in predicate_atoms:
                self._state_nodes.append(
                    self._encode_atom(
                        atom=Atom(
                            predicate=predicate, args=atom_args, variables=tuple()
                        ),
                        edge_label=EdgeLabel.GAMMA,
                        grounded=True,
                        variable_nodes=self._object_nodes,
                    )
                )
        # encode fluents
        for function_id, function_fluents in enumerate(state.fluents):
            if function_id in self.cost_functions:
                # skip total-cost
                continue
            for fluent_args, value in function_fluents.items():
                state_node = self._encode_fluent(
                    fluent=Fluent(
                        function_id=function_id, args=fluent_args, variables=tuple()
                    ),
                    edge_label=EdgeLabel.GAMMA,
                    grounded=True,
                    variable_nodes=self._object_nodes,
                )
                self._state_nodes.append(state_node)
                self._map_node2value[state_node] = float(value)

    def _reset_index_function(self):
        # Index Function dim
        if self.index_function_type == IndexFunctionType.ONEHOT:
            # max arity of predicates (min 2 because of binary operators in numeric expressions)
            self.index_function_dim = max(
                2, max(len(p.parameters) for p in self.predicates_to_encode)
            )
            self._index_lows = [0] * self.index_function_dim
            self._index_highs = [1] * self.index_function_dim

        else:
            self.index_function_dim = max(2, self.index_function_default_dim)
            self._index_lows = [-1.0] * self.index_function_dim
            self._index_highs = [1.0] * self.index_function_dim

    @property
    def nodes_dim(self) -> int:
        # flags (ie node labels) + value + index function
        return (
            len(self.map_int2nodelabel)
            + int(self.has_value_column)
            + self.index_function_dim
        )

    def has_fluents(self) -> bool:
        return len(self.task.functions) > 0

    @property
    def nodes_dtype(self) -> npt.DTypeLike:
        if (
            self.index_function_type == IndexFunctionType.ONEHOT
            and not self.has_fluents()
        ):
            return np.int_
        else:
            return np.float64

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
            scheme_pred_node = self._encode_atom(
                atom=atom,
                edge_label=edge_label,
                grounded=grounded,
                variable_nodes=variable_nodes,
                action_node=action_node,
                subgraph=subgraph,
            )
            scheme_pred_nodes.append(scheme_pred_node)
        for atom in condition.negated_atoms:
            scheme_pred_node = self._encode_atom(
                atom=atom,
                edge_label=edge_label,
                grounded=grounded,
                variable_nodes=variable_nodes,
                action_node=action_node,
                subgraph=subgraph,
            )
            scheme_pred_nodes.append(scheme_pred_node)
            self._negated_atom_nodes.append(scheme_pred_node)
        for constraint in condition.constraints:
            scheme_pred_node = self._encode_constraint(
                constraint=constraint,
                edge_label=edge_label,
                grounded=grounded,
                variable_nodes=variable_nodes,
                action_node=action_node,
                subgraph=subgraph,
            )
            scheme_pred_nodes.append(scheme_pred_node)

        return scheme_pred_nodes

    def _encode_action(self, action: Action) -> None:
        subgraph = action
        # action node
        action_node = self._get_new_node_id()
        self._action_nodes.append(action_node)
        # action arg nodes
        action_arg_nodes = self._get_new_node_ids(action.parameters)
        for action_arg_node in action_arg_nodes:
            self._add_edge(action_node, action_arg_node, EdgeLabel.NU, subgraph=action)
        variable_nodes = action_arg_nodes
        # precondition
        self._encode_condition(
            condition=action.precondition,
            edge_label=EdgeLabel.PRE,
            grounded=False,
            variable_nodes=variable_nodes,
            action_node=action_node,
            subgraph=subgraph,
        )
        # add and del effects
        for e in action.effect.effects:
            atomic_effect = e.outcomes[0][1][0].effect
            edge_label = EdgeLabel.EFFECT
            if isinstance(atomic_effect, (AddEffect, DelEffect)):
                atom = atomic_effect.atom
                scheme_pred_node = self._encode_atom(
                    atom=atom,
                    edge_label=edge_label,
                    grounded=False,
                    variable_nodes=variable_nodes,
                    action_node=action_node,
                    subgraph=subgraph,
                )
                if isinstance(atomic_effect, DelEffect):
                    self._negated_atom_nodes.append(scheme_pred_node)
            elif isinstance(atomic_effect, NumericEffect):
                fluent = atomic_effect.fluent
                scheme_fluent_node = self._encode_fluent(
                    fluent=fluent,
                    edge_label=edge_label,
                    grounded=False,
                    variable_nodes=variable_nodes,
                    action_node=action_node,
                    subgraph=subgraph,
                )
                expr_node = self._encode_numeric_expression(
                    expr=atomic_effect.expr,
                    edge_label=edge_label,
                    variable_nodes=variable_nodes,
                    grounded=False,
                    action_node=action_node,
                    subgraph=subgraph,
                )
                assign_node = self._encode_binary_operation(
                    operator_label=NodeLabel.ASSIGN,
                    lhs_node=scheme_fluent_node,
                    rhs_node=expr_node,
                    edge_label=edge_label,
                    subgraph=subgraph,
                )
            else:
                raise NotImplementedError()

    def _encode_binary_operation(
        self,
        operator_label: NodeLabel,
        lhs_node: int,
        rhs_node: int,
        edge_label: EdgeLabel,
        subgraph: Any,
    ) -> int:
        operator_node = self._get_new_node_id()
        self._map_node2operator[operator_node] = operator_label
        self._add_edge(operator_node, lhs_node, label=edge_label, subgraph=subgraph)
        self._add_edge(operator_node, rhs_node, label=edge_label, subgraph=subgraph)
        self._map_node2argindex[lhs_node] = 0
        self._map_node2argindex[rhs_node] = 1
        return operator_node

    def _encode_constraint(
        self,
        constraint: NumericConstraint,
        edge_label: EdgeLabel,
        variable_nodes: list[int],
        grounded: bool,
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> int:
        if subgraph is None:
            fluents = _extract_fluents_from_numeric_expression(constraint.expr)
            if len(fluents) > 0:
                fluent = fluents[0]
                fluent_id = fluent.function_id
                subgraph = self.task.functions[fluent_id]
        comparator_labels = _convert_comparator2nodelabels(constraint.comparator)
        expr_node = self._encode_numeric_expression(
            expr=constraint.expr,
            edge_label=edge_label,
            variable_nodes=variable_nodes,
            grounded=grounded,
            action_node=action_node,
            subgraph=subgraph,
        )
        self._map_node2comparator[expr_node] = comparator_labels
        return expr_node

    def _encode_numeric_expression(
        self,
        expr: NumericExpression,
        edge_label: EdgeLabel,
        variable_nodes: list[int],
        grounded: bool,
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> int:
        if isinstance(expr, BinaryOperation):
            lhs_node = self._encode_numeric_expression(
                expr=expr.lhs,
                edge_label=edge_label,
                variable_nodes=variable_nodes,
                grounded=grounded,
                action_node=action_node,
                subgraph=subgraph,
            )
            rhs_node = self._encode_numeric_expression(
                expr=expr.rhs,
                edge_label=edge_label,
                variable_nodes=variable_nodes,
                grounded=grounded,
                action_node=action_node,
                subgraph=subgraph,
            )
            if isinstance(expr, Addition):
                operator_label = NodeLabel.PLUS
            elif isinstance(expr, Subtraction):
                operator_label = NodeLabel.MINUS
            elif isinstance(expr, Division):
                operator_label = NodeLabel.DIVIDE
            elif isinstance(expr, Multiplication):
                operator_label = NodeLabel.MULTIPLY
            else:
                raise NotImplementedError()
            expr_node = self._encode_binary_operation(
                operator_label=operator_label,
                lhs_node=lhs_node,
                rhs_node=rhs_node,
                edge_label=edge_label,
                subgraph=subgraph,
            )
        elif isinstance(expr, Fluent):
            expr_node = self._encode_fluent(
                fluent=expr,
                edge_label=edge_label,
                variable_nodes=variable_nodes,
                grounded=grounded,
                action_node=action_node,
                subgraph=subgraph,
            )
        elif isinstance(expr, Constant):
            expr_node = self._get_new_node_id()
            self._map_node2value[expr_node] = float(expr.value)
        else:
            raise NotImplementedError()
        return expr_node

    def _encode_fluent(
        self,
        fluent: Fluent,
        edge_label: EdgeLabel,
        variable_nodes: list[int],
        grounded: bool,
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> int:
        fluent_id = fluent.function_id
        if subgraph is None:
            subgraph = self.task.functions[fluent_id]
        function_node = self._function_nodes[fluent_id]
        return self._encode_atom_or_fluent(
            atom_or_fluent=fluent,
            predicate_or_function_node=function_node,
            edge_label=edge_label,
            variable_nodes=variable_nodes,
            grounded=grounded,
            action_node=action_node,
            subgraph=subgraph,
        )

    def _encode_atom(
        self,
        atom: Atom,
        edge_label: EdgeLabel,
        variable_nodes: list[int],
        grounded: bool,
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> int:
        if subgraph is None:
            subgraph = self.predicates_to_encode[atom.predicate]
        predicate_node = self._predicate_nodes[atom.predicate]
        return self._encode_atom_or_fluent(
            atom_or_fluent=atom,
            predicate_or_function_node=predicate_node,
            edge_label=edge_label,
            variable_nodes=variable_nodes,
            grounded=grounded,
            action_node=action_node,
            subgraph=subgraph,
        )

    def _encode_atom_or_fluent(
        self,
        atom_or_fluent: Union[Atom, Fluent],
        predicate_or_function_node: int,
        edge_label: EdgeLabel,
        variable_nodes: list[int],
        grounded: bool,
        action_node: Optional[int] = None,
        subgraph: Optional[Any] = None,
    ) -> int:
        scheme_pred_node = self._get_new_node_id()
        self._add_edge(
            predicate_or_function_node,
            scheme_pred_node,
            edge_label,
            subgraph=subgraph,
        )
        if grounded:
            list_atom_pos_arg = list(enumerate(atom_or_fluent.args))
        else:
            list_atom_pos_arg = [(pos, arg) for arg, pos in atom_or_fluent.variables]
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

    def _get_edges_color(self, graph: gym.spaces.GraphInstance):
        if self.edges_dim > 1:
            return {
                tuple(e): self.map_edgelabel2color[map_int2edgelabel[int(i_lab)]]
                for e, i_lab in zip(graph.edge_links, np.argmax(graph.edges, -1))
            }
        else:
            return {
                tuple(e): self.map_edgelabel2color[EdgeLabel.GAMMA]
                if is_gamma
                else self.map_edgelabel2color[EdgeLabel.NU]
                for e, is_gamma in zip(graph.edge_links, graph.edges.ravel())
            }

    def _prepare_for_plot(
        self, graph: gym.spaces.GraphInstance
    ) -> tuple[
        dict[int, str],
        dict[int, str],
        dict[tuple[int, int], str],
    ]:
        edge_color = self._get_edges_color(graph)
        node_color = {
            node: self.map_nodelabel2color[self.map_int2nodelabel[int(i_lab)]]
            if has_lab
            else self.map_nodelabel2color[None]
            for node, (i_lab, has_lab) in enumerate(
                zip(
                    np.argmax(graph.nodes[:, : self._n_node_labels], -1),
                    np.sum(graph.nodes[:, : self._n_node_labels], -1),
                )
            )
        }
        node_labels_np = np.full(shape=(len(graph.nodes),), fill_value="", dtype=object)
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.NEGATED)
        ] = "not"
        try:
            argindex_nodes_features = graph.nodes[:, -self.index_function_dim :]
            argindex_rows = np.sum(argindex_nodes_features, -1).nonzero()[0]
            for node in argindex_rows:
                node_labels_np[node] = str(
                    self.index_function_inverse(argindex_nodes_features[node])
                )
        except NotImplementedError:
            for node, idx in self._map_node2argindex.items():
                node_labels_np[node] = str(idx)

        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.LESS)
        ] += "<"
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.GREATER)
        ] += ">"
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.EQUAL)
        ] += "="
        ind = self._get_node_indices_from_label(graph, label=NodeLabel.NUMERIC)
        node_labels_np[ind] = [
            node_labels_np[i] + ", " + str(graph.nodes[i, self.value_column_index])
            for i in ind
        ]
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.PLUS)
        ] += ", +"
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.MINUS)
        ] += ", -"
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.MULTIPLY)
        ] += ", *"
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.DIVIDE)
        ] += ", /"
        node_labels_np[
            self._get_node_indices_from_label(graph, label=NodeLabel.ASSIGN)
        ] += ":="
        for i, node in enumerate(self._action_nodes):
            node_labels_np[node] = self.task.actions[i].name
        for i, node in enumerate(self._predicate_nodes):
            node_labels_np[node] = self.predicates_to_encode[i].name
        for i, node in enumerate(self._function_nodes):
            node_labels_np[node] = self.task.functions[i].name
        for i, node in enumerate(self._object_nodes):
            node_labels_np[node] = self.task.objects[i]

        node_labels: dict[int, str] = dict(enumerate(node_labels_np))

        return node_labels, node_color, edge_color

    def _get_node_indices_from_label(
        self, graph: gym.spaces.GraphInstance, label: NodeLabel
    ) -> np.ndarray:
        if label in self.map_int2nodelabel:
            return graph.nodes[:, self.map_nodelabel2int[label]].nonzero()[0]
        else:
            return np.array([], dtype=int)


def decode_llg(
    graph: gym.spaces.GraphInstance, cost_functions: Optional[set[int]] = None
) -> State:
    """Decode a llg graph into a plado state without information about the plado.Task.

    This may be less efficient than LLGEncoder.decode() but is self-sufficient.

    Works only if LLGEncoder was used with options encode_actions=True and simplify_encoding=False.

    """
    if cost_functions is None:
        cost_functions = set()
    object_nodes = graph.nodes[:, map_nodelabel2int[NodeLabel.OBJECT]].nonzero()[0]
    map_object_node2id = {node: i for i, node in enumerate(object_nodes)}
    predicate_nodes = (
        graph.nodes[:, map_nodelabel2int[NodeLabel.PREDICATE]]  # predicate
        * (1 - graph.nodes[:, map_nodelabel2int[NodeLabel.STATIC]])  # not static
    ).nonzero()[0]
    function_nodes = (graph.nodes[:, map_nodelabel2int[NodeLabel.FLUENT]]).nonzero()[0]
    map_predicate_node2id = {node: i for i, node in enumerate(predicate_nodes)}
    map_function_node2id = {node: i for i, node in enumerate(function_nodes)}
    state_nodes = graph.nodes[:, map_nodelabel2int[NodeLabel.STATE]].nonzero()[0]
    state = State(
        num_predicates=len(predicate_nodes), num_functions=len(function_nodes)
    )
    for state_node in state_nodes:
        state_node_neighbours = [e[1] for e in graph.edge_links if e[0] == state_node]
        state_node_neighbours.sort()
        atom_args = []
        predicate = None
        function_id = None
        value = Float(graph.nodes[state_node, value_index])
        for node in state_node_neighbours:
            try:
                # atom predicate
                predicate = map_predicate_node2id[node]
            except KeyError:
                try:
                    # fluent function
                    function_id = map_function_node2id[node]
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
        if function_id is not None:
            state.fluents[function_id][tuple(atom_args)] = value
    # add total-cost = 0.
    for function_id in cost_functions:
        state.fluents[function_id][tuple()] = Float(0)
    return state


def _extract_fluents_from_numeric_expression(expr: NumericExpression) -> list[Fluent]:
    if isinstance(expr, BinaryOperation):
        return _extract_fluents_from_numeric_expression(
            expr.lhs
        ) + _extract_fluents_from_numeric_expression(expr.rhs)
    elif isinstance(expr, Fluent):
        return [expr]
    else:
        return []


def _convert_comparator2nodelabels(comparator: int) -> list[NodeLabel]:
    comparator_labels: list[NodeLabel] = []
    if comparator in (NumericConstraint.LESS, NumericConstraint.LESS_EQUAL):
        comparator_labels.append(NodeLabel.LESS)
    if comparator in (NumericConstraint.GREATER, NumericConstraint.GREATER_EQUAL):
        comparator_labels.append(NodeLabel.GREATER)
    if comparator in (
        NumericConstraint.LESS_EQUAL,
        NumericConstraint.EQUAL,
        NumericConstraint.GREATER_EQUAL,
    ):
        comparator_labels.append(NodeLabel.EQUAL)
    return comparator_labels
