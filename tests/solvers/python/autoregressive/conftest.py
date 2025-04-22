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
    - (0, -1) => rester sur place
    - (1, x) => aller au noeud x (impossible si pas de edge directe)

    obs = state:
        - native encoding: noeud d'1 graphe
        - graph encoding: graphe avec node feature 1 sur le current node et 0 sinon

    graphe codé en dur:

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


class D(
    Domain,
    SingleAgent,
    Sequential,
    Initializable,
    Markovian,
    FullyObservable,
    Rewards,
):
    T_state = int  # Type of states
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


@fixture
def graph_walk_env():
    return GraphWalkEnv()


@fixture
def graph_walk_with_graph_obs_env():
    return GraphWalkEnv(state_encoding=StateEncoding.GRAPH)


@fixture
def graph_walk_domain_factory():
    return GraphWalkDomain


@fixture
def graph_walk_with_graph_obs_domain_factory():
    return lambda: GraphWalkDomain(state_encoding=StateEncoding.GRAPH)
