from typing import Any

import numpy as np
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.spaces import Box, Graph, GraphInstance

from skdecide.core import Space, TransitionOutcome, Value
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn import GraphPPO
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask import MaskableGraphPPO
from skdecide.hub.space.gym import DiscreteSpace, GymSpace, ListSpace
from skdecide.utils import rollout

# JSP graph env


class D(GymDomain):
    T_state = GraphInstance  # Type of states
    T_observation = T_state  # Type of observations
    T_event = int  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class GraphJspDomain(D):
    _gym_env: DisjunctiveGraphJspEnv

    def __init__(self, gym_env):
        GymDomain.__init__(self, gym_env=gym_env)
        if self._gym_env.normalize_observation_space:
            self.n_nodes_features = gym_env.n_machines + 1
        else:
            self.n_nodes_features = 2

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        outcome = super()._state_step(action=action)
        outcome.state = self._np_state2graph_state(outcome.state)
        return outcome

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(np.nonzero(self._gym_env.valid_action_mask())[0])

    def _state_reset(self) -> D.T_state:
        return self._np_state2graph_state(super()._state_reset())

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        # overriden to get an enumerable space
        return DiscreteSpace(n=self._gym_env.action_space.n)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        if self._gym_env.normalize_observation_space:
            original_graph_space = Graph(
                node_space=Box(
                    low=0.0, high=1.0, shape=(self.n_nodes_features,), dtype=np.float_
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


jsp_env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=False,
    flat_observation_space=False,
    action_mode="task",
)

# random rollout
domain = GraphJspDomain(gym_env=jsp_env)
rollout(domain=domain, max_steps=jsp_env.total_tasks_without_dummies, num_episodes=1)

# solve with sb3-GraphPPO
domain_factory = lambda: GraphJspDomain(gym_env=jsp_env)
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=GraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={"total_timesteps": 100},
) as solver:

    solver.solve()
    rollout(domain=domain_factory(), solver=solver, max_steps=100, num_episodes=1)

# solver with sb3-MaskableGraphPPO
domain_factory = lambda: GraphJspDomain(gym_env=jsp_env)
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=MaskableGraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={"total_timesteps": 100},
    use_action_masking=True,
) as solver:

    solver.solve()
    rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=100,
        num_episodes=1,
        use_applicable_actions=True,
    )
