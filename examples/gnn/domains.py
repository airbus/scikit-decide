"""Fixtures to be reused by several test files."""

from typing import Any, Optional

import numpy as np
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from gymnasium.spaces import Box, Discrete, Graph, GraphInstance

from skdecide.builders.domain import (
    FullyObservable,
    Initializable,
    Markovian,
    Renderable,
    Rewards,
    Sequential,
    SingleAgent,
    UnrestrictedActions,
)
from skdecide.core import Mask, Space, TransitionOutcome, Value
from skdecide.domains import DeterministicPlanningDomain, Domain
from skdecide.hub.domain.maze.maze import DEFAULT_MAZE, Action, Maze, State
from skdecide.hub.space.gym import DictSpace, DiscreteSpace, GymSpace, ListSpace


class D(DeterministicPlanningDomain, Renderable):
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

    def _get_action_mask_from(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> D.T_agent[Mask]:
        # a different way to display applicable actions
        # we could also override only _get_applicable_action_from() but it will be more computationally efficient to
        # implement directly _get_action_mask_from()
        if memory is None:
            memory = self._memory
        mazestate_memory = self._graph2mazestate(memory)
        return np.array(
            [
                self._graph2mazestate(
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
                    self._get_action_mask_from(memory=memory),
                )
                if mask
            ]
        )


class UnmaskedGraphMaze(GraphMaze, UnrestrictedActions):
    """Version of GraphMaze with unrestricted actions.

    Useful to test algo without action masking.

    """

    ...


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
    """Domain for JobShopProblem with observations being graphs and actions nodes.

    It wraps the gym environment `DisjunctiveGraphJspEnv` from graph-jsp-env library,
    whose states are a numpy version of the graph.

    """

    _gym_env: DisjunctiveGraphJspEnv

    def __init__(self, gym_env):
        self._gym_env = gym_env
        if self._gym_env.normalize_observation_space:
            self.n_nodes_features = gym_env.n_machines + 1
        else:
            self.n_nodes_features = 2

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
                    dtype=np.float64,
                ),
                edge_space=Box(low=0, high=1.0, dtype=np.float64),
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

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return DiscreteSpace(n=self._gym_env.action_space.n)

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


class UnmaskedGraphJspDomain(UnrestrictedActions, GraphJspDomain):
    """Version of GraphJspDomain with unrestricted actions.

    Useful to test algo without action masking.

    """

    ...


class D(GraphJspDomain):
    T_state = dict[str, Any]


class MultiInputGraphJspDomain(D):
    """Multi-input version of GraphJspDomain.

    This domain adds only a constant features to test algos with multiinputs, some being graphs.

    """

    def _get_observation_space_(self) -> Space[D.T_observation]:
        return DictSpace(
            spaces=dict(
                graph=super()._get_observation_space_(),
                static=Box(low=0.0, high=1.0, dtype=float),
            )
        )

    def _state_step(
        self, action: D.T_event
    ) -> TransitionOutcome[D.T_state, Value[D.T_value], D.T_predicate, D.T_info]:
        transition = super()._state_step(action)
        transition.state = dict(graph=transition.state, static=np.array([0.5]))
        return transition

    def _state_reset(self) -> D.T_state:
        return dict(graph=super()._state_reset(), static=np.array([0.5]))


class UnmaskedMultiInputGraphJspDomain(UnrestrictedActions, GraphJspDomain):
    """Version of UnmaskedMultiInputGraphJspDomain with unrestricted actions.

    Useful to test algo without action masking.

    """

    ...
