from typing import Any

import gymnasium as gym
import numpy as np
import scipy
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.spaces import Box, Graph, GraphInstance

from skdecide.builders.domain import (
    FullyObservable,
    Initializable,
    Markovian,
    Renderable,
    Rewards,
    Sequential,
    SingleAgent,
)
from skdecide.core import Space, TransitionOutcome, Value
from skdecide.domains import Domain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn.ppo.ppo import Graph2GraphPPO
from skdecide.hub.space.gym import GymSpace
from skdecide.utils import rollout


class D(
    Domain,
    SingleAgent,
    Sequential,
    Initializable,
    Markovian,
    FullyObservable,
    Renderable,
    Rewards,
):
    T_state = GraphInstance  # Type of states
    T_observation = T_state  # Type of observations
    T_event = GraphInstance  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class GraphJspDomain(D):
    _gym_env: DisjunctiveGraphJspEnv

    def __init__(self, gym_env, deterministic=False):
        self._gym_env = gym_env
        if self._gym_env.normalize_observation_space:
            self.n_nodes_features = gym_env.n_machines + 1
        else:
            self.n_nodes_features = 2
        self.deterministic = deterministic

    def _state_reset(self) -> D.T_state:
        return self._np_state2graph_state(self._gym_env.reset()[0])

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        env_state, reward, terminated, truncated, info = self._gym_env.step(
            self._graph_action2env_action(action, deterministic=self.deterministic)
        )
        state = self._np_state2graph_state(env_state)
        if truncated:
            info["TimeLimit.truncated"] = True
        return TransitionOutcome(
            state=state, value=Value(reward=reward), termination=terminated, info=info
        )

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return NotImplementedError(
            "`get_applicable_actions()` is not applicable :) here as the graph action space is continuous"
        )

    def _is_applicable_action_from(
        self, action: D.T_agent[D.T_event], memory: D.T_memory[D.T_state]
    ) -> bool:
        return self._graph_action2env_action(action) in self._gym_env.valid_actions()

    def _get_observation_space_(self) -> Space[D.T_observation]:
        if self._gym_env.normalize_observation_space:
            original_graph_space = Graph(
                node_space=Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.n_nodes_features,),
                    dtype=np.float_,
                ),
                edge_space=Box(low=0, high=1.0, dtype=np.float_),
            )

        else:
            original_graph_space = Graph(
                node_space=Box(
                    low=np.array([0, 0]),
                    high=np.array(
                        [
                            self._gym_env.n_machines,
                            self._gym_env.longest_processing_time,
                        ]
                    ),
                    dtype=np.int_,
                ),
                edge_space=Box(
                    low=0, high=self._gym_env.longest_processing_time, dtype=np.int_
                ),
            )
        return GymSpace(original_graph_space)

    def _get_action_space(self) -> Space[D.T_observation]:
        if self._gym_env.normalize_observation_space:
            edge_space = Box(low=0, high=1.0, dtype=np.float_)
        else:
            edge_space = Box(
                low=0, high=self._gym_env.longest_processing_time, dtype=np.int_
            )
        original_graph_space = Graph(
            node_space=Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float_,
            ),
            edge_space=edge_space,
        )
        return GymSpace(original_graph_space)

    def _np_state2graph_state(self, np_state: np.array) -> GraphInstance:
        if not self._gym_env.normalize_observation_space:
            np_state = np_state.astype(np.int_)

        nodes = np_state[:, -self.n_nodes_features :]
        adj = np_state[:, : -self.n_nodes_features]
        edge_starts_ends = adj.nonzero()
        edge_links = np.transpose(edge_starts_ends)
        edges = adj[edge_starts_ends][:, None]

        return GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)

    def _graph_action2env_action(
        self, graph_action: gym.spaces.GraphInstance, deterministic=False
    ) -> int:
        logits = graph_action.nodes
        if deterministic:
            node_idx = np.argmax(logits)
        else:
            probs = scipy.special.softmax(logits, axis=0).flatten()
            node_idx = np.random.choice(a=len(probs), p=probs)
        return int(node_idx)

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        return self._gym_env.render(**kwargs)


jsp = np.array(
    [
        [
            [0, 1, 2],  # machines for job 0
            [0, 2, 1],  # machines for job 1
            [0, 1, 2],  # machines for job 2
        ],
        [
            [3, 2, 2],  # task durations of job 0
            [2, 1, 4],  # task durations of job 1
            [0, 4, 3],  # task durations of job 2
        ],
    ]
)


domain_factory = lambda: GraphJspDomain(
    gym_env=DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        normalize_observation_space=False,
        flat_observation_space=False,
        action_mode="task",
    )
)


with StableBaseline(
    domain_factory=domain_factory,
    algo_class=Graph2GraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={
        "total_timesteps": 10000,
    },
    # n_steps=512,
) as solver:
    solver.solve()
    rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=30,
        num_episodes=1,
        render=True,
    )
