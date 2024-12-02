import sys
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pytest
import torch as th
import torch_geometric as thg
from gymnasium.spaces import Box, Discrete, Graph, GraphInstance
from pytest_cases import fixture, fixture_union, param_fixture
from torch_geometric.nn import global_add_pool

from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.core import Space, TransitionOutcome, Value
from skdecide.domains import DeterministicPlanningDomain, Domain
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.domain.maze import Maze
from skdecide.hub.domain.maze.maze import DEFAULT_MAZE, Action, State
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn import GraphPPO
from skdecide.hub.solver.stable_baselines.gnn.common.torch_layers import (
    GraphFeaturesExtractor,
)
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask import MaskableGraphPPO
from skdecide.hub.space.gym import DictSpace, GymSpace, ListSpace
from skdecide.utils import rollout

if not sys.platform.startswith("win"):
    try:
        from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
    except ImportError:
        graph_jsp_env_available = False
    else:
        graph_jsp_env_available = True
else:
    # import not working on windows because of the banner
    graph_jsp_env_available = False


if graph_jsp_env_available:
    # JSP graph env

    class D(Domain):
        T_state = GraphInstance  # Type of states
        T_observation = T_state  # Type of observations
        T_event = int  # Type of events
        T_value = float  # Type of transition values (rewards or costs)
        T_info = None  # Type of additional information in environment outcome

    class GraphJspDomain(GymDomain, D):
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

        def action_masks(self) -> npt.NDArray[bool]:
            return np.array(self._gym_env.valid_action_mask())

        def _get_applicable_actions_from(
            self, memory: D.T_memory[D.T_state]
        ) -> D.T_agent[Space[D.T_event]]:
            return ListSpace(np.nonzero(self._gym_env.valid_action_mask())[0])

        def _is_applicable_action_from(
            self, action: D.T_agent[D.T_event], memory: D.T_memory[D.T_state]
        ) -> bool:
            return self._gym_env.valid_action_mask()[action]

        def _state_reset(self) -> D.T_state:
            return self._np_state2graph_state(super()._state_reset())

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

    class DD(D):
        T_state = dict[str, Any]

    class MultiInputGraphJspDomain(GraphJspDomain, DD):
        def _get_observation_space_(self) -> Space[D.T_observation]:
            return DictSpace(
                spaces=dict(
                    graph=super()._get_observation_space_(),
                    static=Box(low=0.0, high=1.0),
                )
            )

        def _state_step(
            self, action: DD.T_event
        ) -> TransitionOutcome[
            DD.T_state, Value[DD.T_value], DD.T_predicate, DD.T_info
        ]:
            transition = super()._state_step(action)
            transition.state = dict(graph=transition.state, static=np.array([0.5]))
            return transition

        def _state_reset(self) -> DD.T_state:
            return dict(graph=super()._state_reset(), static=np.array([0.5]))

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


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = GraphInstance  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class GraphMaze(D):
    def __init__(self, maze_str: str = DEFAULT_MAZE, discrete_features: bool = False):
        self.discrete_features = discrete_features
        self.maze_domain = Maze(maze_str=maze_str)
        np_wall = np.array(self.maze_domain._maze)
        np_y = np.array(
            [
                [(i) for j in range(self.maze_domain._num_cols)]
                for i in range(self.maze_domain._num_rows)
            ]
        )
        np_x = np.array(
            [
                [(j) for j in range(self.maze_domain._num_cols)]
                for i in range(self.maze_domain._num_rows)
            ]
        )
        walls = np_wall.ravel()
        coords = [i for i in zip(np_y.ravel(), np_x.ravel())]
        np_node_id = np.reshape(range(len(walls)), np_wall.shape)
        edge_links = []
        edges = []
        for i in range(self.maze_domain._num_rows):
            for j in range(self.maze_domain._num_cols):
                current_coord = (i, j)
                if i > 0:
                    next_coord = (i - 1, j)
                    edge_links.append(
                        (np_node_id[current_coord], np_node_id[next_coord])
                    )
                    edges.append(np_wall[current_coord] * np_wall[next_coord])
                if i < self.maze_domain._num_rows - 1:
                    next_coord = (i + 1, j)
                    edge_links.append(
                        (np_node_id[current_coord], np_node_id[next_coord])
                    )
                    edges.append(np_wall[current_coord] * np_wall[next_coord])
                if j > 0:
                    next_coord = (i, j - 1)
                    edge_links.append(
                        (np_node_id[current_coord], np_node_id[next_coord])
                    )
                    edges.append(np_wall[current_coord] * np_wall[next_coord])
                if j < self.maze_domain._num_cols - 1:
                    next_coord = (i, j + 1)
                    edge_links.append(
                        (np_node_id[current_coord], np_node_id[next_coord])
                    )
                    edges.append(np_wall[current_coord] * np_wall[next_coord])
        self.edges = np.array(edges)
        self.edge_links = np.array(edge_links)
        self.walls = walls
        self.node_ids = np_node_id
        self.coords = coords

    def _mazestate2graph(self, state: State) -> GraphInstance:
        x, y = state
        agent_presence = np.zeros(self.walls.shape, dtype=self.walls.dtype)
        agent_presence[self.node_ids[y, x]] = 1
        nodes = np.stack([self.walls, agent_presence], axis=-1)
        if self.discrete_features:
            return GraphInstance(
                nodes=nodes, edges=self.edges, edge_links=self.edge_links
            )
        else:
            return GraphInstance(
                nodes=nodes, edges=self.edges[:, None], edge_links=self.edge_links
            )

    def _graph2mazestate(self, graph: GraphInstance) -> State:
        y, x = self.coords[graph.nodes[:, 1].nonzero()[0][0]]
        return State(x=x, y=y)

    def _is_terminal(self, state: D.T_state) -> D.T_predicate:
        return self.maze_domain._is_terminal(self._graph2mazestate(state))

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        maze_memory = self._graph2mazestate(memory)
        maze_next_state = self.maze_domain._get_next_state(
            memory=maze_memory, action=action
        )
        return self._mazestate2graph(maze_next_state)

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        maze_memory = self._graph2mazestate(memory)
        if next_state is None:
            maze_next_state = None
        else:
            maze_next_state = self._graph2mazestate(next_state)
        return self.maze_domain._get_transition_value(
            memory=maze_memory, action=action, next_state=maze_next_state
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        return self.maze_domain._get_action_space_()

    def _get_goals_(self) -> Space[D.T_observation]:
        return ListSpace([self._mazestate2graph(self.maze_domain._goal)])

    def _is_goal(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_predicate]:
        return self.maze_domain._is_goal(self._graph2mazestate(observation))

    def _get_initial_state_(self) -> D.T_state:
        return self._mazestate2graph(self.maze_domain._get_initial_state_())

    def _get_observation_space_(self) -> Space[D.T_observation]:
        if self.discrete_features:
            return GymSpace(
                Graph(
                    node_space=Box(low=0, high=1, shape=(2,), dtype=self.walls.dtype),
                    edge_space=Discrete(2),
                )
            )
        else:
            return GymSpace(
                Graph(
                    node_space=Box(low=0, high=1, shape=(2,), dtype=self.walls.dtype),
                    edge_space=Box(low=0, high=1, shape=(1,), dtype=self.edges.dtype),
                )
            )

    def _render_from(self, memory: D.T_state, **kwargs: Any) -> Any:
        maze_memory = self._graph2mazestate(memory)
        self.maze_domain._render_from(memory=maze_memory, **kwargs)

    def action_masks(self) -> npt.NDArray[bool]:
        mazestate_memory = self._graph2mazestate(self._memory)
        return np.array(
            [
                self._graph2mazestate(
                    self._get_next_state(action=action, memory=self._memory)
                )
                != mazestate_memory
                for action in self._get_action_space().get_elements()
            ]
        )


discrete_features = param_fixture("discrete_features", [False, True])


@fixture
def maze_domain_factory(discrete_features):
    return lambda: GraphMaze(discrete_features=discrete_features)


@fixture
def jsp_domain_factory():
    if sys.platform.startswith("win"):
        pytest.skip("jsp-graph-env not importable on windows")
    if not graph_jsp_env_available:
        pytest.skip("jsp-graph-env not available")
    return lambda: GraphJspDomain(
        gym_env=DisjunctiveGraphJspEnv(
            jps_instance=jsp,
            perform_left_shift_if_possible=True,
            normalize_observation_space=False,
            flat_observation_space=False,
            action_mode="task",
        )
    )


@fixture
def jsp_dict_domain_factory():
    if sys.platform.startswith("win"):
        pytest.skip("jsp-graph-env not importable on windows")
    if not graph_jsp_env_available:
        pytest.skip("jsp-graph-env not available")
    return lambda: MultiInputGraphJspDomain(
        gym_env=DisjunctiveGraphJspEnv(
            jps_instance=jsp,
            perform_left_shift_if_possible=True,
            normalize_observation_space=False,
            flat_observation_space=False,
            action_mode="task",
        )
    )


domain_factory = fixture_union(
    "domain_factory", [maze_domain_factory, jsp_domain_factory]
)


def test_observation_space(domain_factory):
    domain = domain_factory()
    assert domain.reset() in domain.get_observation_space()


def test_ppo(domain_factory):
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_ppo_user_gnn(domain_factory):
    domain = domain_factory()
    node_features_dim = int(
        np.prod(domain.get_observation_space().unwrapped().node_space.shape)
    )
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
        policy_kwargs=dict(
            features_extractor_class=GraphFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_class=thg.nn.models.GAT,
                gnn_kwargs=dict(
                    in_channels=node_features_dim,
                    hidden_channels=64,
                    num_layers=2,
                    dropout=0.2,
                ),
                gnn_out_dim=64,
                features_dim=64,
            ),
        ),
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


class MyReductionLayer(th.nn.Module):
    def __init__(self, gnn_out_dim: int, features_dim: int):
        super().__init__()
        self.gnn_out_dim = gnn_out_dim
        self.features_dim = features_dim
        self.linear_layer = th.nn.Linear(gnn_out_dim, features_dim)

    def forward(self, embedded_observations: thg.data.Data) -> th.Tensor:
        x, edge_index, batch = (
            embedded_observations.x,
            embedded_observations.edge_index,
            embedded_observations.batch,
        )
        h = global_add_pool(x, batch)
        h = self.linear_layer(h).relu()
        return h


def test_ppo_user_reduction_layer(domain_factory):
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="GraphInputPolicy",
        learn_config={"total_timesteps": 100},
        policy_kwargs=dict(
            features_extractor_class=GraphFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_out_dim=128,
                features_dim=64,
                reduction_layer_class=MyReductionLayer,
                reduction_layer_kwargs=dict(
                    gnn_out_dim=128,
                    features_dim=64,
                ),
            ),
        ),
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_maskable_ppo(domain_factory):
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
            render=False,
            use_action_masking=True,
        )


def test_dict_ppo(jsp_dict_domain_factory):
    domain_factory = jsp_dict_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=GraphPPO,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_dict_maskable_ppo(jsp_dict_domain_factory):
    domain_factory = jsp_dict_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=MaskableGraphPPO,
        baselines_policy="MultiInputPolicy",
        learn_config={"total_timesteps": 100},
        use_action_masking=True,
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
            use_action_masking=True,
        )
