"""Fixtures to be reused by several test files."""

from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
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
from skdecide.hub.space.gym import DiscreteSpace, ListSpace, MultiDiscreteSpace


class GraphWalkEnv(gym.Env[int, int]):
    """Custom env.

    actions:
    - (0, -1) => rester sur place
    - (1, x) => aller au noeud x (impossible si pas de edge directe)

    obs = state: noeud d'1 graphe

    graphe codé en dur:

    0 -> 1 -> 2 --> 4
      \         /
       -> 3 ---



    """

    N_STATES = 5

    action_space = gym.spaces.MultiDiscrete((2, N_STATES))
    observation_space = gym.spaces.Discrete(N_STATES)

    state = 0

    edges_end_nodes = [[1, 3], [2], [4], [4], []]  # start node 0 -> 1,3

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.state = 0
        return self.state, {}

    def step(
        self, action: npt.NDArray[int]
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        action_id, node = action
        if action_id == 1:
            if node not in self.edges_end_nodes[self.state]:
                raise RuntimeError()
            else:
                self.state = node
        terminated = self.state == self.N_STATES - 1
        if terminated:
            reward = 1000.0
        else:
            reward = -100.0
        return self.state, reward, terminated, False, {}

    def action_masks(self) -> npt.NDArray[int]:
        """Actually returns the applicable actions as a whole numpy array."""
        return np.array(self.applicable_actions())

    def applicable_actions(self) -> list[npt.NDArray[int]]:
        return [np.array((0, -1))] + [
            np.array((1, node)) for node in self.edges_end_nodes[self.state]
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
    T_state = int  # Type of states
    T_observation = T_state  # Type of observations
    T_event = npt.NDArray[int]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = dict[str, Any]  # Type of additional information in environment outcome


class GraphWalkDomain(D):
    def __init__(self):
        self._gym_env = GraphWalkEnv()

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
        return MultiDiscreteSpace(nvec=self._gym_env.action_space.nvec)

    def _get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
        return DiscreteSpace(n=self._gym_env.observation_space.n)


@fixture
def graph_walk_env():
    return GraphWalkEnv()


@fixture
def graph_walk_domain_factory():
    return GraphWalkDomain
