from typing import Any

import numpy as np
import torch as th
import torch_geometric as thg
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
from skdecide.hub.solver.stable_baselines.gnn.ppo.ppo import Graph2NodePPO
from skdecide.hub.space.gym import GymSpace, ListSpace
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
    T_event = int  # Type of events
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
        env_state, reward, terminated, truncated, info = self._gym_env.step(action)
        state = self._np_state2graph_state(env_state)
        if truncated:
            info["TimeLimit.truncated"] = True
        return TransitionOutcome(
            state=state, value=Value(reward=reward), termination=terminated, info=info
        )

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(np.nonzero(self._gym_env.valid_action_mask())[0])

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

    def _get_action_space_(self) -> Space[D.T_observation]:
        return GymSpace(self._gym_env.action_space)

    def _np_state2graph_state(self, np_state: np.array) -> GraphInstance:
        if not self._gym_env.normalize_observation_space:
            np_state = np_state.astype(np.int_)

        nodes = np_state[:, -self.n_nodes_features :]
        adj = np_state[:, : -self.n_nodes_features]
        edge_starts_ends = adj.nonzero()
        edge_links = np.transpose(edge_starts_ends)
        edges = adj[edge_starts_ends][:, None]

        return GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)

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


def test_gnn_graph2node_jsp_sb3():

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
        algo_class=Graph2NodePPO,
        baselines_policy="GraphInputPolicy",
        learn_config={
            "total_timesteps": 200,
        },
        n_steps=100,
    ) as solver:
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=30,
            num_episodes=1,
            render=False,
        )

        # check batch handling by policy
        domain = domain_factory()
        obs = domain.reset()
        policy = solver._algo.policy
        obs_thg_data, _ = policy.obs_to_tensor(obs)
        # set training=False to be deterministic
        policy.train(False)
        # batch with same obs => same logits
        obs_batch = thg.data.Batch.from_data_list([obs_thg_data, obs_thg_data])
        batched_logits = policy.get_distribution(obs_batch).distribution.logits
        assert th.allclose(batched_logits[0], batched_logits[1])
        # batch with an obs with less node => last logit ~= -inf
        x = obs_thg_data.x[:-1, :]
        edge_index, edge_attr = thg.utils.subgraph(
            subset=list(range(len(obs_thg_data.x) - 1)),
            edge_index=obs_thg_data.edge_index,
            edge_attr=obs_thg_data.edge_attr,
        )
        obs_thg_data2 = thg.data.Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        obs_batch = thg.data.Batch.from_data_list([obs_thg_data, obs_thg_data2])
        last_prob = th.exp(policy.get_distribution(obs_batch).distribution.logits)[
            -1, -1
        ]
        assert last_prob == 0.0
