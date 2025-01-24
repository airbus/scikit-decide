from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Box, Discrete, Graph, GraphInstance

from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.core import Mask, Space, Value
from skdecide.domains import DeterministicPlanningDomain
from skdecide.hub.domain.maze import Maze
from skdecide.hub.domain.maze.maze import DEFAULT_MAZE, Action, State
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.gnn import GraphPPO
from skdecide.hub.solver.stable_baselines.gnn.ppo_mask import MaskableGraphPPO
from skdecide.hub.space.gym import GymSpace, ListSpace
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

    def _get_action_mask(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> D.T_agent[Mask]:
        # overriden since by default it is only 1's (inheriting from UnrestrictedAction)
        # we could also override only _get_applicable_action() but it will be more computationally efficient to
        # implement directly get_action_mask()
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


MAZE = """
+-+-+-+-+o+-+-+--+-+-+
|   |              | |
+ + + +-+-+-+ +--+ + +
| | |   |   | |  |   |
+ +-+-+ +-+ + + -+ +-+
| |   |   | |    |   |
+ + + + + + + +--+ +-+
|   |   |   | |      |
+-+-+-+-+-+-+-+ -+-+ +
|             |    | |
+ +-+-+-+-+ + +--+-+ +
|   |       |        |
+ + + +-+ +-+ +--+-+-+
| | |   |     |      |
+ +-+-+ + +-+ + -+-+ +
| |     | | | |    | |
+-+ +-+ + + + +--+ + +
|   |   |   |    | | |
+ +-+ +-+-+-+-+ -+ + +
|   |       |      | |
+-+-+-+-+-+x+-+--+-+-+
"""

domain = GraphMaze(maze_str=MAZE, discrete_features=True)
assert domain.reset() in domain.get_observation_space()

# random rollout
rollout(domain=domain, max_steps=50, num_episodes=1)

# solve with sb3-PPO-GNN
domain_factory = lambda: GraphMaze(maze_str=MAZE)
max_steps = domain.maze_domain._num_cols * domain.maze_domain._num_rows
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=GraphPPO,
    baselines_policy="GraphInputPolicy",
    learn_config={"total_timesteps": 100},
) as solver:

    solver.solve()
    rollout(domain=domain_factory(), solver=solver, max_steps=max_steps, num_episodes=1)

# solver with sb3-MaskableGraphPPO
domain_factory = lambda: GraphMaze(maze_str=MAZE)
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
