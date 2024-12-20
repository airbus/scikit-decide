import logging
import os
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import ray
import torch as th
import torch_geometric as thg
from gymnasium.spaces import Box, Discrete, Graph, GraphInstance
from pytest_cases import fixture, param_fixture
from torch_geometric.nn import global_add_pool

from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.core import Mask, Space, Value
from skdecide.domains import DeterministicPlanningDomain
from skdecide.hub.domain.maze import Maze
from skdecide.hub.domain.maze.maze import DEFAULT_MAZE, Action, State
from skdecide.hub.solver.ray_rllib import RayRLlib
from skdecide.hub.solver.ray_rllib.gnn.algorithms import GraphPPO
from skdecide.hub.space.gym import DictSpace, GymSpace, ListSpace
from skdecide.utils import rollout


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

    def _mazestate2graph(self, state: State) -> D.T_state:
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

    def _graph2mazestate(self, graph: D.T_state) -> State:
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


class D(DeterministicPlanningDomain, Renderable):
    T_state = GraphInstance  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MaskedGraphMaze(D):
    def __init__(self, maze_str: str = DEFAULT_MAZE, discrete_features: bool = False):
        self.graph_maze = GraphMaze(
            maze_str=maze_str, discrete_features=discrete_features
        )

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        return self.graph_maze._get_next_state(memory=memory, action=action)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return self.graph_maze._get_transition_value(
            memory=memory, action=action, next_state=next_state
        )

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self.graph_maze._is_terminal(state=state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return self.graph_maze._get_action_space_()

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return self.graph_maze._get_goals_()

    def _is_goal(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_predicate]:
        return self.graph_maze._is_goal(observation=observation)

    def _get_initial_state_(self) -> D.T_state:
        return self.graph_maze._get_initial_state_()

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return self.graph_maze._get_observation_space_()

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        return self.graph_maze._render_from(memory=memory, **kwargs)

    def _get_action_mask(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> D.T_agent[Mask]:
        # a different way to display applicable actions
        # we could also override only _get_applicable_action() but it will be more computationally efficient to
        # implement directly get_action_mask()
        if memory is None:
            memory = self._memory
        mazestate_memory = self.graph_maze._graph2mazestate(memory)
        return np.array(
            [
                self.graph_maze._graph2mazestate(
                    self._get_next_state(action=action, memory=memory)
                )
                != mazestate_memory
                for action in self._get_action_space().get_elements()
            ]
        )

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(
            [
                action
                for action, mask in zip(
                    self._get_action_space().get_elements(),
                    self._get_action_mask(memory=memory),
                )
                if mask
            ]
        )


class D(GraphMaze):
    T_state = dict[str, Any]


class DictGraphMaze(D):
    def _get_observation_space_(self) -> Space[D.T_observation]:
        return DictSpace(
            spaces=dict(
                graph=super()._get_observation_space_(),
                static=Box(low=0.0, high=1.0, dtype=np.float_),
            )
        )

    def _mazestate2graph(self, state: State) -> D.T_state:
        return dict(
            graph=super()._mazestate2graph(state),
            static=np.array([0.5], dtype=np.float_),
        )

    def _graph2mazestate(self, graph: D.T_state) -> State:
        return super()._graph2mazestate(graph["graph"])


class D(DeterministicPlanningDomain, Renderable):
    T_state = dict[str, Any]
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MaskedDictGraphMaze(D):
    def __init__(self, maze_str: str = DEFAULT_MAZE, discrete_features: bool = False):
        self.graph_maze = DictGraphMaze(
            maze_str=maze_str, discrete_features=discrete_features
        )

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        return self.graph_maze._get_next_state(memory=memory, action=action)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        return self.graph_maze._get_transition_value(
            memory=memory, action=action, next_state=next_state
        )

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self.graph_maze._is_terminal(state=state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return self.graph_maze._get_action_space_()

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return self.graph_maze._get_goals_()

    def _is_goal(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_predicate]:
        return self.graph_maze._is_goal(observation=observation)

    def _get_initial_state_(self) -> D.T_state:
        return self.graph_maze._get_initial_state_()

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return self.graph_maze._get_observation_space_()

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        return self.graph_maze._render_from(memory=memory, **kwargs)

    def _get_action_mask(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> D.T_agent[Mask]:
        # a different way to display applicable actions
        # we could also override only _get_applicable_action() but it will be more computationally efficient to
        # implement directly get_action_mask()
        if memory is None:
            memory = self._memory
        mazestate_memory = self.graph_maze._graph2mazestate(memory)
        return np.array(
            [
                self.graph_maze._graph2mazestate(
                    self._get_next_state(action=action, memory=memory)
                )
                != mazestate_memory
                for action in self._get_action_space().get_elements()
            ]
        )

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return ListSpace(
            [
                action
                for action, mask in zip(
                    self._get_action_space().get_elements(),
                    self._get_action_mask(memory=memory),
                )
                if mask
            ]
        )


discrete_features = param_fixture("discrete_features", [False, True])


@fixture
def domain_factory(discrete_features):
    return lambda: GraphMaze(discrete_features=discrete_features)


def test_observation_space(domain_factory):
    domain = domain_factory()
    assert domain.reset() in domain.get_observation_space()
    rollout(domain=domain, num_episodes=1, max_steps=3, render=False, verbose=False)


def test_dict_observation_space():
    domain = DictGraphMaze()
    assert domain.reset() in domain.get_observation_space()
    rollout(domain=domain, num_episodes=1, max_steps=3, render=False, verbose=False)


@fixture
def ray_init():
    # add module test_gnn_ray_rllib and thus GraphMaze to ray runtimeenv
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"working_dir": os.path.dirname(__file__)},
        # local_mode=True,  # uncomment this line and comment the one above to debug more easily
    )


@fixture
def graphppo_config():
    return (
        GraphPPO.get_default_config()
        # set num of CPU<1 to avoid hanging for ever in github actions on macos 11
        .resources(
            num_cpus_per_worker=0.5,
        )
        # small number to increase speed of the unit test
        .training(train_batch_size=256)
    )


def test_ppo(domain_factory, graphppo_config, ray_init):
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


class MyGNN(thg.nn.models.GAT):

    LOG_SENTENCE = "Using custom GNN."

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout,
            act,
            act_first,
            act_kwargs,
            norm,
            norm_kwargs,
            jk,
            **kwargs
        )
        logging.warning(MyGNN.LOG_SENTENCE)


def test_ppo_user_gnn(domain_factory, graphppo_config, ray_init, caplog):
    domain = domain_factory()
    node_features_dim = int(
        np.prod(domain.get_observation_space().unwrapped().node_space.shape)
    )

    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_class=MyGNN,
            gnn_kwargs=dict(
                in_channels=node_features_dim,
                hidden_channels=64,
                num_layers=2,
                dropout=0.2,
            ),
            gnn_out_dim=64,
            features_dim=64,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        with caplog.at_level(logging.WARNING):
            solver.solve()
            rollout(
                domain=domain_factory(),
                solver=solver,
                max_steps=100,
                num_episodes=1,
                render=False,
            )

    assert MyGNN.LOG_SENTENCE in caplog.text


class MyReductionLayer(th.nn.Module):
    LOG_SENTENCE = "Using custom reduction layer."

    def __init__(self, gnn_out_dim: int, features_dim: int):
        super().__init__()
        self.gnn_out_dim = gnn_out_dim
        self.features_dim = features_dim
        self.linear_layer = th.nn.Linear(gnn_out_dim, features_dim)
        logging.warning(MyReductionLayer.LOG_SENTENCE)

    def forward(self, embedded_observations: thg.data.Data) -> th.Tensor:
        x, edge_index, batch = (
            embedded_observations.x,
            embedded_observations.edge_index,
            embedded_observations.batch,
        )
        h = global_add_pool(x, batch)
        h = self.linear_layer(h).relu()
        return h


def test_ppo_user_reduction_layer(domain_factory, graphppo_config, ray_init, caplog):
    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_out_dim=128,
            features_dim=64,
            reduction_layer_class=MyReductionLayer,
            reduction_layer_kwargs=dict(
                gnn_out_dim=128,
                features_dim=64,
            ),
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        with caplog.at_level(logging.WARNING):
            solver.solve()
            rollout(
                domain=domain_factory(),
                solver=solver,
                max_steps=100,
                num_episodes=1,
                render=False,
            )

    assert MyReductionLayer.LOG_SENTENCE in caplog.text


def test_dict_ppo(graphppo_config, ray_init):
    domain_factory = DictGraphMaze
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:

        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_ppo_masked(discrete_features, graphppo_config, ray_init):
    domain_factory = lambda: MaskedGraphMaze(discrete_features=discrete_features)
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_dict_ppo_masked(discrete_features, graphppo_config, ray_init):
    domain_factory = lambda: MaskedDictGraphMaze(discrete_features=discrete_features)
    solver_kwargs = dict(
        algo_class=GraphPPO, train_iterations=1  # , gamma=0.95, train_batch_size_log2=8
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_multiinput_obs
        solver.solve()
        rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=100,
            num_episodes=1,
            render=False,
        )


def test_ppo_masked_user_gnn(graphppo_config, ray_init, caplog):
    domain_factory = MaskedGraphMaze
    node_features_dim = int(
        np.prod(domain_factory().get_observation_space().unwrapped().node_space.shape)
    )
    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_class=MyGNN,
            gnn_kwargs=dict(
                in_channels=node_features_dim,
                hidden_channels=64,
                num_layers=2,
                dropout=0.2,
            ),
            gnn_out_dim=64,
            features_dim=64,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_obs
        with caplog.at_level(logging.WARNING):
            solver.solve()
    assert MyGNN.LOG_SENTENCE in caplog.text


def test_dict_ppo_masked_user_gnn(graphppo_config, ray_init, caplog):
    domain_factory = MaskedDictGraphMaze
    node_features_dim = int(
        np.prod(
            domain_factory()
            .get_observation_space()
            .unwrapped()["graph"]
            .node_space.shape
        )
    )
    solver_kwargs = dict(
        algo_class=GraphPPO,
        train_iterations=1,
        graph_feature_extractors_kwargs=dict(
            gnn_class=MyGNN,
            gnn_kwargs=dict(
                in_channels=node_features_dim,
                hidden_channels=64,
                num_layers=2,
                dropout=0.2,
            ),
            gnn_out_dim=64,
            features_dim=64,
        ),
    )
    with RayRLlib(
        domain_factory=domain_factory, config=graphppo_config, **solver_kwargs
    ) as solver:
        assert solver._action_masking and solver._is_graph_multiinput_obs
        with caplog.at_level(logging.WARNING):
            solver.solve()
    assert MyGNN.LOG_SENTENCE in caplog.text
