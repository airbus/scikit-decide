"""Fixtures to be reused by several test files."""
from enum import Enum
from typing import Any, Optional, SupportsFloat, Union

import gymnasium as gym
import numpy as np
from numpy import typing as npt
from pytest_cases import fixture

from skdecide import Domain, Space, TransitionOutcome, Value
from skdecide.builders.domain import (
    FullyObservable,
    Initializable,
    Markovian,
    Rewards,
    Sequential,
    SingleAgent,
)
from skdecide.hub.space.gym import GymSpace, ListSpace, MaskableMultiDiscreteSpace


class StateEncoding(Enum):
    NATIVE = "native"
    GRAPH = "graph"


class ActionEncoding(Enum):
    NEXT_NODE = "next-node"  # action specifies only action type + next node
    BOTH_NODES = (
        "both-nodes"  # action specifies action type + starting or next node + next node
    )
    # (purely for unit testing purposes)


class GraphWalkEnv(gym.Env[int, int]):
    """Custom env.

    actions:
    - (0, -1) => do not move
    - (1, x) => move to node x (impossible if no direct edge)

    obs = state:
        - native encoding: node id
        - graph encoding: graph with node feature 1 on current node else 0

    hard-coded graph:

    0 -> 1 -> 2 --> 4
      \         /
       -> 3 ---



    """

    N_STATES = 5
    state = 0

    edges_end_nodes = [[1, 3], [2], [4], [4], []]  # start node 0 -> 1,3

    def __init__(
        self,
        state_encoding: StateEncoding = StateEncoding.NATIVE,
        action_encoding: ActionEncoding = ActionEncoding.BOTH_NODES,
    ):
        self.state_encoding = state_encoding
        self.action_encoding = action_encoding
        if self.state_encoding == StateEncoding.NATIVE:
            self.observation_space = gym.spaces.Discrete(self.N_STATES)
        else:
            self.observation_space = gym.spaces.Graph(
                node_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                edge_space=None,
            )
        if self.action_encoding == ActionEncoding.NEXT_NODE:
            self.action_space = gym.spaces.MultiDiscrete((2, self.N_STATES))
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                (2, self.N_STATES, self.N_STATES)
            )

    def node2state(self, node: int):
        if self.state_encoding == StateEncoding.NATIVE:
            return node
        else:
            node_features = np.zeros((self.N_STATES, 1), dtype=np.int8)
            node_features[node] = 1
            edge_links = np.concatenate(
                tuple(
                    tuple((start_node, end_node) for end_node in end_nodes)
                    for start_node, end_nodes in enumerate(self.edges_end_nodes)
                    if len(end_nodes) > 0
                )
            )
            edge_features = None
            return gym.spaces.GraphInstance(
                nodes=node_features, edges=edge_features, edge_links=edge_links
            )

    def state2node(self, state: Union[int, gym.spaces.GraphInstance]) -> int:
        if self.state_encoding == StateEncoding.NATIVE:
            return state
        else:
            return int(state.nodes.nonzero()[0][0])

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.state = self.node2state(0)
        return self.state, {}

    def step(
        self, action: npt.NDArray[int]
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        current_node = self.state2node(self.state)
        if self.action_encoding == ActionEncoding.NEXT_NODE:
            action_id, node = action
        else:
            action_id, start_node, node = action
            if action_id == 1 and start_node != current_node and start_node != node:
                raise RuntimeError()
        if action_id == 1:
            if node not in self.edges_end_nodes[current_node]:
                raise RuntimeError()
            else:
                current_node = node
                self.state = self.node2state(node)
        terminated = current_node == self.N_STATES - 1
        if terminated:
            reward = 1000.0
        else:
            reward = -100.0
        return self.state, reward, terminated, False, {}

    def action_masks(self) -> npt.NDArray[int]:
        """Actually returns the applicable actions as a whole numpy array."""
        return np.array(self.applicable_actions())

    def applicable_actions(self) -> list[npt.NDArray[int]]:
        current_node = self.state2node(self.state)
        if self.action_encoding == ActionEncoding.NEXT_NODE:
            return [np.array((0, -1))] + [
                np.array((1, node)) for node in self.edges_end_nodes[current_node]
            ]
        else:
            return (
                [np.array((0, -1, -1))]
                + [
                    np.array((1, current_node, node))
                    for node in self.edges_end_nodes[current_node]
                ]
                + [
                    np.array((1, node, node))
                    for node in self.edges_end_nodes[current_node]
                ]
            )


class HeteroGraphWalkEnv(gym.Env[int, int]):
    """Custom env with "hetero" graph.

    Some nodes are position and other represent the distance between positions.

    actions:
    - (0, -1) => do not move
    - (1, x) => move to node x (impossible if no couple edges to go there)

    obs = state:
    graph with 2 type of nodes separated by feature POSITION (1: yes, 0:no) and DISTANCE
    (only if POSITION =0, corresponds to distance between the neighbours of this node)

    graphe codé en dur:

    0 -> 1 -> 2 --> 4
      \         /
       -> 3 ---



    """

    N_STATES = 5
    POSITION = 0
    DISTANCE = 1
    CURRENT = 2
    node_features_dim = 3

    action_space = gym.spaces.MultiDiscrete((2, N_STATES))

    state = 0

    edges_end_nodes_with_distance = [
        [(1, 1), (3, 5)],
        [(2, 1)],
        [(4, 1)],
        [(4, 5)],
        [],
    ]  # start node 0 -> 1,3 with dist 1 and 5 resp.

    def __init__(self):
        self.observation_space = gym.spaces.Graph(
            node_space=gym.spaces.Box(
                low=np.array([0, 0, 0]), high=np.array([1, 10, 1]), dtype=np.int8
            ),
            edge_space=None,
        )

    @property
    def n_nodes(self):
        return 2 * sum(
            len(l_by_start) for l_by_start in self.edges_end_nodes_with_distance
        )

    def node2state(self, node: int):
        node_features = np.zeros((self.n_nodes, self.node_features_dim), dtype=np.int8)
        node_features[: len(self.edges_end_nodes_with_distance), self.POSITION] = 1
        node_features[node, self.CURRENT] = 1
        edge_links = []
        i_node = len(self.edges_end_nodes_with_distance)
        for start_node, end_nodes_n_dist in enumerate(
            self.edges_end_nodes_with_distance
        ):
            for end_node, dist in end_nodes_n_dist:
                edge_links.append((start_node, i_node))
                edge_links.append((i_node, end_node))
                node_features[i_node, self.DISTANCE] = dist
                i_node += 1
        edge_links_np = np.array(edge_links)
        edge_features = None
        return gym.spaces.GraphInstance(
            nodes=node_features, edges=edge_features, edge_links=edge_links_np
        )

    def state2node(self, state: Union[int, gym.spaces.GraphInstance]) -> int:
        return int(state.nodes[:, self.CURRENT].nonzero()[0][0])

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.state = self.node2state(0)
        return self.state, {}

    def step(
        self, action: npt.NDArray[int]
    ) -> tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        current_node = self.state2node(self.state)
        action_id, node = action
        if action_id == 1:
            ok = False
            for node2, dist in self.edges_end_nodes_with_distance[current_node]:
                if node2 == node:
                    ok = True
                    break
            if ok:
                current_node = node
                self.state = self.node2state(node)
            else:
                raise RuntimeError()
        else:
            dist = 1

        terminated = current_node == self.N_STATES - 1
        if terminated:
            reward = 0
        else:
            reward = -dist
        return self.state, reward, terminated, False, {}

    def action_masks(self) -> npt.NDArray[int]:
        """Actually returns the applicable actions as a whole numpy array."""
        return np.array(self.applicable_actions())

    def applicable_actions(self) -> list[npt.NDArray[int]]:
        current_node = self.state2node(self.state)
        return [np.array((0, -1))] + [
            np.array((1, node))
            for node, dist in self.edges_end_nodes_with_distance[current_node]
        ]


class D(
    Domain,
    SingleAgent,
    Sequential,
    Initializable,
    Markovian,
    FullyObservable,
    Rewards,
):
    T_state = Union[int, gym.spaces.GraphInstance]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = npt.NDArray[int]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = dict[str, Any]  # Type of additional information in environment outcome


class GraphWalkDomain(D):
    def __init__(self, state_encoding: StateEncoding = StateEncoding.NATIVE):
        self._gym_env = GraphWalkEnv(state_encoding=state_encoding)

    def _state_reset(self) -> D.T_state:
        return self._gym_env.reset()[0]

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        state, reward, terminated, truncated, info = self._gym_env.step(action)
        return TransitionOutcome(
            state=state, value=Value(reward=reward), termination=terminated, info=info
        )

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(self._gym_env.applicable_actions())

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return MaskableMultiDiscreteSpace(nvec=self._gym_env.action_space.nvec)

    def _get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
        return GymSpace(gym_space=self._gym_env.observation_space)


class HeteroGraphWalkDomain(D):
    def __init__(self):
        self._gym_env = HeteroGraphWalkEnv()

    def _state_reset(self) -> D.T_state:
        return self._gym_env.reset()[0]

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        state, reward, terminated, truncated, info = self._gym_env.step(action)
        return TransitionOutcome(
            state=state, value=Value(reward=reward), termination=terminated, info=info
        )

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(self._gym_env.applicable_actions())

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return MaskableMultiDiscreteSpace(nvec=self._gym_env.action_space.nvec)

    def _get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
        return GymSpace(gym_space=self._gym_env.observation_space)


@fixture
def graph_walk_env():
    return GraphWalkEnv()


@fixture
def graph_walk_with_graph_obs_env():
    return GraphWalkEnv(state_encoding=StateEncoding.GRAPH)


@fixture
def graph_walk_with_heterograph_obs_env():
    return HeteroGraphWalkEnv()


@fixture
def action_components_node_flag_indices():
    return [None, HeteroGraphWalkEnv.POSITION]


@fixture
def graph_walk_domain_factory():
    return GraphWalkDomain


@fixture
def graph_walk_with_graph_obs_domain_factory():
    return lambda: GraphWalkDomain(state_encoding=StateEncoding.GRAPH)


@fixture
def graph_walk_with_heterograph_obs_domain_factory():
    return HeteroGraphWalkDomain
