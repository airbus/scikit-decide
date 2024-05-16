# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator as op
import random as rd
from copy import deepcopy
from enum import Enum
from math import sqrt
from typing import Any, NamedTuple, Optional

import matplotlib.pyplot as plt

from skdecide.builders.domain.agent import MultiAgent, SingleAgent
from skdecide.builders.domain.concurrency import Sequential
from skdecide.builders.domain.dynamics import Simulation, UncertainTransitions
from skdecide.builders.domain.events import Actions
from skdecide.builders.domain.goals import Goals
from skdecide.builders.domain.initialization import (
    DeterministicInitialized,
    Initializable,
)
from skdecide.builders.domain.memory import Markovian
from skdecide.builders.domain.observability import FullyObservable
from skdecide.builders.domain.renderability import Renderable
from skdecide.builders.domain.value import PositiveCosts
from skdecide.core import (
    DiscreteDistribution,
    Distribution,
    SamplableSpace,
    SingleValueDistribution,
    Space,
    StrDict,
    TransitionOutcome,
    Value,
)
from skdecide.domains import Domain
from skdecide.hub.solver.lrtdp import LRTDP
from skdecide.hub.solver.martdp import MARTDP
from skdecide.hub.solver.mcts import HMCTS
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver, rollout

DEFAULT_MAZE = """
+-+-+-+-+-+-+-+-+-+-+
|         |         |
+         +         +
|         |         |
+         +         +
|         |         |
+         +         +
|         |         |
+         +         +
|     +-+-+-+-+     |
+                   +
|     +-+-+-+-+     |
+         +         +
|         |         |
+         +         +
|         |         |
+         +         +
|         |         |
+         +         +
|         |         |
+-+-+-+-+-+-+-+-+-+-+
"""


class AgentState(NamedTuple):
    x: int
    y: int


class AgentAction(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    stay = 4  # only possible in agent's goal state


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))


class D(
    Domain,
    MultiAgent,
    Sequential,
    Simulation,
    Actions,
    DeterministicInitialized,
    Markovian,
    PositiveCosts,
    Goals,
    FullyObservable,
    Renderable,
):
    T_state = StrDict[AgentState]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = StrDict[AgentAction]  # Type of events
    T_value = StrDict[Value]  # Type of transition values (rewards or costs)
    T_predicate = StrDict[bool]  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class MultiAgentMaze(D):
    def __init__(self, maze_str=DEFAULT_MAZE, nb_agents=4, flatten_data=False):
        maze = []
        for y, line in enumerate(maze_str.strip().split("\n")):
            line = line.rstrip()
            row = []
            for x, c in enumerate(line):
                if c == " ":
                    row.append(1)  # spaces are 1s
                else:
                    row.append(0)  # walls are 0s
            maze.append(row)
        self._maze = maze
        self._num_cols = len(maze[0])
        self._num_rows = len(maze)
        self._ax = None
        self._nb_agents = nb_agents
        self._flatten_data = flatten_data
        self._generate_agents()

    def get_agents(self):
        return {"Agent #{}".format(i) for i in range(self._nb_agents)}

    def _generate_agents(self):
        # Randomly place self._nb_agents / 2 in each half of the maze
        quadrant_xdim = (self._num_cols // 2) - 1
        quadrant_ydim = (self._num_rows // 2) - 2
        quadrant = [i for i in range(quadrant_xdim * quadrant_ydim)]
        left_agents_1 = rd.sample(quadrant, self._nb_agents // 4)
        left_agents_1_starts = [
            tuple(map(op.add, divmod(i, quadrant_ydim), (1, 1))) for i in left_agents_1
        ]
        left_agents_2 = rd.sample(quadrant, self._nb_agents // 4)
        left_agents_2_starts = [
            tuple(map(op.add, divmod(i, quadrant_ydim), (1, quadrant_ydim + 4)))
            for i in left_agents_2
        ]
        left_agents_starts = left_agents_1_starts + left_agents_2_starts
        right_agents_1 = rd.sample(quadrant, self._nb_agents // 4)
        right_agents_1_starts = [
            tuple(map(op.add, divmod(i, quadrant_ydim), (quadrant_xdim + 2, 1)))
            for i in right_agents_1
        ]
        right_agents_2 = rd.sample(quadrant, self._nb_agents // 4)
        right_agents_2_starts = [
            tuple(
                map(
                    op.add,
                    divmod(i, quadrant_ydim),
                    (quadrant_xdim + 2, quadrant_ydim + 4),
                )
            )
            for i in right_agents_2
        ]
        right_agents_starts = right_agents_1_starts + right_agents_2_starts

        # Randomly assign left agents' starts to right agents' goals and vice versa
        right_agent_orders = [j for j in range(len(right_agents_starts))]
        rd.shuffle(right_agent_orders)
        left_agents_goals = [right_agents_starts[i] for i in right_agent_orders]
        left_agent_orders = [j for j in range(len(left_agents_starts))]
        rd.shuffle(left_agent_orders)
        right_agents_goals = [left_agents_starts[i] for i in left_agent_orders]

        # Set the agents
        self._agents_starts = left_agents_starts + right_agents_starts
        self._agents_starts = {
            "Agent #{}".format(i): AgentState(x=p[0], y=p[1])
            for i, p in enumerate(self._agents_starts)
        }
        self._agents_goals = left_agents_goals + right_agents_goals
        self._agents_goals = {
            "Agent #{}".format(i): AgentState(x=p[0], y=p[1])
            for i, p in enumerate(self._agents_goals)
        }

        return self._agents_starts

    def _get_initial_state_(self) -> D.T_state:
        return HashableDict(self._agents_starts)

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        next_state = {}
        transition_value = {}
        occupied_cells = {}
        dead_end = {a: False for a in memory}

        for agent, state in memory.items():
            if state == self._agents_goals[agent]:
                next_state[agent] = state
                transition_value[agent] = Value(cost=0)
                continue
            if (
                action[agent] == AgentAction.stay
            ):  # must test after goal check for proper cost setting
                next_state[agent] = state
                transition_value[agent] = Value(cost=1)
                continue
            next_state_1 = next_state_2 = next_state_3 = state
            if action[agent] == AgentAction.left:
                if state.x > 0 and self._maze[state.y][state.x - 1] == 1:
                    next_state_1 = AgentState(x=state.x - 1, y=state.y)
                    if state.y > 0 and self._maze[state.y - 1][state.x - 1] == 1:
                        next_state_2 = AgentState(x=state.x - 1, y=state.y - 1)
                    if (
                        state.y < self._num_rows - 1
                        and self._maze[state.y + 1][state.x - 1] == 1
                    ):
                        next_state_2 = AgentState(x=state.x - 1, y=state.y + 1)
            elif action[agent] == AgentAction.right:
                if (
                    state.x < self._num_cols - 1
                    and self._maze[state.y][state.x + 1] == 1
                ):
                    next_state_1 = AgentState(x=state.x + 1, y=state.y)
                    if state.y > 0 and self._maze[state.y - 1][state.x + 1] == 1:
                        next_state_2 = AgentState(x=state.x + 1, y=state.y - 1)
                    if (
                        state.y < self._num_rows - 1
                        and self._maze[state.y + 1][state.x + 1] == 1
                    ):
                        next_state_2 = AgentState(x=state.x + 1, y=state.y + 1)
            elif action[agent] == AgentAction.up:
                if state.y > 0 and self._maze[state.y - 1][state.x] == 1:
                    next_state_1 = AgentState(x=state.x, y=state.y - 1)
                    if state.x > 0 and self._maze[state.y - 1][state.x - 1] == 1:
                        next_state_2 = AgentState(x=state.x - 1, y=state.y - 1)
                    if (
                        state.x < self._num_cols - 1
                        and self._maze[state.y - 1][state.x + 1] == 1
                    ):
                        next_state_2 = AgentState(x=state.x + 1, y=state.y - 1)
            elif action[agent] == AgentAction.down:
                if (
                    state.y < self._num_rows - 1
                    and self._maze[state.y + 1][state.x] == 1
                ):
                    next_state_1 = AgentState(x=state.x, y=state.y + 1)
                    if state.x > 0 and self._maze[state.y + 1][state.x - 1] == 1:
                        next_state_2 = AgentState(x=state.x - 1, y=state.y + 1)
                    if (
                        state.x < self._num_cols - 1
                        and self._maze[state.y + 1][state.x + 1] == 1
                    ):
                        next_state_2 = AgentState(x=state.x + 1, y=state.y + 1)
            next_state[agent] = rd.choices(
                [next_state_1, next_state_2, next_state_3], [0.8, 0.1, 0.1], k=1
            )[0]
            transition_value[agent] = Value(cost=1)
            if tuple(next_state[agent]) in occupied_cells:
                dead_end[agent] = True
                dead_end[occupied_cells[tuple(next_state[agent])]] = True
                transition_value[agent] = Value(cost=1000)  # for random walk
            else:
                occupied_cells[tuple(next_state[agent])] = agent
        return TransitionOutcome(
            state=HashableDict(next_state),
            value=(
                transition_value
                if not self._flatten_data
                else Value(cost=sum(v.cost for a, v in transition_value.items()))
            ),
            termination=(
                dead_end
                if not self._flatten_data
                else all(t for a, t in dead_end.items())
            ),
            info=None,
        )

    def get_agent_applicable_actions(
        self,
        memory: D.T_memory[D.T_state],
        other_agents_actions: D.T_agent[D.T_concurrency[D.T_event]],
        agent: str,
    ) -> Space[D.T_event]:
        if memory[agent] == self._agents_goals[agent]:
            return ListSpace([AgentAction.stay])
        else:
            # compute other agents' most probably occupied next cells
            occupied_next_cells = set()
            for other_agent, other_agent_action in other_agents_actions.items():
                if other_agent_action == AgentAction.left:
                    occupied_next_cells.add(
                        (memory[other_agent].x - 1, memory[other_agent].y)
                    )
                elif other_agent_action == AgentAction.right:
                    occupied_next_cells.add(
                        (memory[other_agent].x + 1, memory[other_agent].y)
                    )
                elif other_agent_action == AgentAction.up:
                    occupied_next_cells.add(
                        (memory[other_agent].x, memory[other_agent].y - 1)
                    )
                elif other_agent_action == AgentAction.down:
                    occupied_next_cells.add(
                        (memory[other_agent].x, memory[other_agent].y + 1)
                    )
                elif other_agent_action == AgentAction.stay:
                    occupied_next_cells.add(
                        (memory[other_agent].x, memory[other_agent].y)
                    )
            # now, compute application actions
            applicable_actions = [AgentAction.stay]
            if (
                memory[agent].y > 0
                and self._maze[memory[agent].y - 1][memory[agent].x] == 1
                and (memory[agent].x, memory[agent].y - 1) not in occupied_next_cells
            ):
                applicable_actions.append(AgentAction.up)
            if (
                memory[agent].y < self._num_rows - 1
                and self._maze[memory[agent].y + 1][memory[agent].x] == 1
                and (memory[agent].x, memory[agent].y + 1) not in occupied_next_cells
            ):
                applicable_actions.append(AgentAction.down)
            if (
                memory[agent].x > 0
                and self._maze[memory[agent].y][memory[agent].x - 1] == 1
                and (memory[agent].x - 1, memory[agent].y) not in occupied_next_cells
            ):
                applicable_actions.append(AgentAction.left)
            if (
                memory[agent].x < self._num_cols - 1
                and self._maze[memory[agent].y][memory[agent].x + 1] == 1
                and (memory[agent].x + 1, memory[agent].y) not in occupied_next_cells
            ):
                applicable_actions.append(AgentAction.right)
            return ListSpace(applicable_actions)

    class SamplableActionSpace(SamplableSpace):
        def __init__(self, domain: D, memory: D.T_memory[D.T_state]) -> None:
            self._domain = domain
            self._memory = memory

        def items(self) -> D.T_agent[D.T_concurrency[EnumSpace]]:
            # Used by random walk that needs independent agent action spaces.
            # It may lead to infeasible actions in which case _state_sample(...)
            # returns a large cost
            return [(a, EnumSpace(AgentAction)) for a in self._memory.keys()]

        def sample(self) -> D.T_agent[D.T_concurrency[D.T_event]]:
            # Shuffle the agents (not in place, thus don't use rd.shuffle())
            agents_order = rd.sample(self._memory.keys(), k=len(self._memory.keys()))
            agents_actions = {}
            for agent in agents_order:
                agents_actions[agent] = self._domain.get_agent_applicable_actions(
                    self._memory, agents_actions, agent
                ).sample()
            return HashableDict(agents_actions)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return MultiAgentMaze.SamplableActionSpace(domain=self, memory=memory)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return {agent: ListSpace([goal]) for agent, goal in self._agents_goals.items()}

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return StrDict[MultiDiscreteSpace([self._num_cols, self.n_um_rows])]

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        if self._ax is None:
            fig = plt.gcf()
            ax = plt.axes()
            ax.set_aspect("equal")  # set the x and y axes to the same scale
            plt.xticks([])  # remove the tick marks by setting to an empty list
            plt.yticks([])  # remove the tick marks by setting to an empty list
            ax.invert_yaxis()  # invert the y-axis so the first row of data is at the top
            self._ax = ax
            plt.ion()
        maze = deepcopy(self._maze)
        for agent, state in memory.items():
            maze[self._agents_goals[agent].y][self._agents_goals[agent].x] = 0.7
            maze[state.y][state.x] = 0.3
        self._ax.pcolormesh(maze)
        plt.pause(0.0001)


class D(
    Domain,
    SingleAgent,
    Sequential,
    UncertainTransitions,
    Actions,
    Initializable,
    Markovian,
    PositiveCosts,
    Goals,
    FullyObservable,
):
    T_state = AgentState  # Type of states
    T_observation = T_state  # Type of observations
    T_event = AgentAction  # Type of events
    T_value = Value  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of logical checks
    T_info = (
        None  # Type of additional information given as part of an environment outcome
    )


class SingleAgentMaze(D):
    def __init__(self, maze, agent_goal):
        self._maze = maze
        self._num_cols = len(maze[0])
        self._num_rows = len(maze)
        self._goal = agent_goal

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        applicable_actions = []
        if memory.y > 0 and self._maze[memory.y - 1][memory.x] == 1:
            applicable_actions.append(AgentAction.up)
        if memory.y < self._num_rows - 1 and self._maze[memory.y + 1][memory.x] == 1:
            applicable_actions.append(AgentAction.down)
        if memory.x > 0 and self._maze[memory.y][memory.x - 1] == 1:
            applicable_actions.append(AgentAction.left)
        if memory.x < self._num_cols - 1 and self._maze[memory.y][memory.x + 1] == 1:
            applicable_actions.append(AgentAction.right)
        return ListSpace(applicable_actions)

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> Distribution[D.T_state]:
        if memory == self._goal:
            return SingleValueDistribution(memory)
        next_state_1 = next_state_2 = next_state_3 = memory
        if action == AgentAction.left:
            if memory.x > 0 and self._maze[memory.y][memory.x - 1] == 1:
                next_state_1 = AgentState(x=memory.x - 1, y=memory.y)
                if memory.y > 0 and self._maze[memory.y - 1][memory.x - 1] == 1:
                    next_state_2 = AgentState(x=memory.x - 1, y=memory.y - 1)
                if (
                    memory.y < self._num_rows - 1
                    and self._maze[memory.y + 1][memory.x - 1] == 1
                ):
                    next_state_2 = AgentState(x=memory.x - 1, y=memory.y + 1)
        if action == AgentAction.right:
            if (
                memory.x < self._num_cols - 1
                and self._maze[memory.y][memory.x + 1] == 1
            ):
                next_state_1 = AgentState(x=memory.x + 1, y=memory.y)
                if memory.y > 0 and self._maze[memory.y - 1][memory.x + 1] == 1:
                    next_state_2 = AgentState(x=memory.x + 1, y=memory.y - 1)
                if (
                    memory.y < self._num_rows - 1
                    and self._maze[memory.y + 1][memory.x + 1] == 1
                ):
                    next_state_2 = AgentState(x=memory.x + 1, y=memory.y + 1)
        if action == AgentAction.up:
            if memory.y > 0 and self._maze[memory.y - 1][memory.x] == 1:
                next_state_1 = AgentState(x=memory.x, y=memory.y - 1)
                if memory.x > 0 and self._maze[memory.y - 1][memory.x - 1] == 1:
                    next_state_2 = AgentState(x=memory.x - 1, y=memory.y - 1)
                if (
                    memory.x < self._num_cols - 1
                    and self._maze[memory.y - 1][memory.x + 1] == 1
                ):
                    next_state_2 = AgentState(x=memory.x + 1, y=memory.y - 1)
        if action == AgentAction.down:
            if (
                memory.y < self._num_rows - 1
                and self._maze[memory.y + 1][memory.x] == 1
            ):
                next_state_1 = AgentState(x=memory.x, y=memory.y + 1)
                if memory.x > 0 and self._maze[memory.y + 1][memory.x - 1] == 1:
                    next_state_2 = AgentState(x=memory.x - 1, y=memory.y + 1)
                if (
                    memory.x < self._num_cols - 1
                    and self._maze[memory.y + 1][memory.x + 1] == 1
                ):
                    next_state_2 = AgentState(x=memory.x + 1, y=memory.y + 1)
        return DiscreteDistribution(
            [(next_state_1, 0.8), (next_state_2, 0.1), (next_state_3, 0.1)]
        )

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if memory == self._goal:
            return Value(cost=0)
        else:
            return Value(cost=1)

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        return self._is_goal(state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(AgentAction)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ListSpace([self._goal])

    def _get_initial_state_(self) -> D.T_state:
        return self._start

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self._num_cols, self._num_rows])


def martdp_callback(solver):
    print(
        "Best value: {}".format(
            solver.get_utility(solver.get_domain().get_initial_state())
        )
    )
    print("Residual moving average: {}".format(solver.get_residual_moving_average()))
    return solver.get_residual_moving_average() <= 0.1


def mcts_callback(solver, i=None):
    print(
        "Best value: {}".format(
            solver.get_utility(solver.get_domain().get_initial_state())
        )
    )
    print("Residual moving average: {}".format(solver.get_residual_moving_average()))
    return solver.get_residual_moving_average() <= 0.01


if __name__ == "__main__":

    try_solvers = [
        # Multi-agent RTDP
        {
            "name": "Multi-agent RTDP with single-agent LRTDP heuristics",
            "entry": "MAHD",
            "config": {
                "multiagent_solver_class": MARTDP,
                "singleagent_solver_class": LRTDP,
                "multiagent_domain_class": MultiAgentMaze,
                "singleagent_domain_class": SingleAgentMaze,
                "multiagent_domain_factory": lambda: MultiAgentMaze(),
                "singleagent_domain_factory": lambda multiagent_domain, agent: SingleAgentMaze(
                    multiagent_domain._maze, multiagent_domain._agents_goals[agent]
                ),
                "multiagent_solver_kwargs": {
                    "domain_factory": lambda: MultiAgentMaze(),
                    "time_budget": 600000,
                    "max_depth": 50,
                    "residual_moving_average_window": 10,
                    "max_feasibility_trials": 10,
                    "graph_expansion_rate": 0.01,
                    "action_choice_noise": 0.1,
                    "dead_end_cost": 1000,
                    "callback": lambda slv: martdp_callback(slv),
                    "online_node_garbage": True,
                    "continuous_planning": False,
                    "verbose": False,
                },
                "singleagent_solver_kwargs": {
                    "heuristic": lambda d, s: Value(
                        cost=sqrt((d._goal.x - s.x) ** 2 + (d._goal.y - s.y) ** 2)
                    ),
                    "use_labels": True,
                    "time_budget": 1000,
                    "max_depth": 100,
                    "continuous_planning": False,
                    "online_node_garbage": False,
                    "parallel": False,
                    "verbose": False,
                },
            },
        },
        # Multi-agent UCT
        {
            "name": "Multi-agent UCT with single-agent LRTDP heuristics",
            "entry": "MAHD",
            "config": {
                "multiagent_solver_class": HMCTS,
                "singleagent_solver_class": LRTDP,
                "multiagent_domain_class": MultiAgentMaze,
                "singleagent_domain_class": SingleAgentMaze,
                "multiagent_domain_factory": lambda: MultiAgentMaze(),
                "singleagent_domain_factory": lambda multiagent_domain, agent: SingleAgentMaze(
                    multiagent_domain._maze, multiagent_domain._agents_goals[agent]
                ),
                "multiagent_solver_kwargs": {
                    "domain_factory": lambda: MultiAgentMaze(flatten_data=True),
                    "time_budget": 600000,
                    "max_depth": 50,
                    "residual_moving_average_window": 10,
                    "heuristic_confidence": 1000,
                    "action_choice_noise": 0.1,
                    "expander": HMCTS.Expander.PARTIAL,
                    "state_expansion_rate": 0.01,
                    "action_expansion_rate": 0.01,
                    "transition_mode": HMCTS.TransitionMode.SAMPLE,
                    "online_node_garbage": True,
                    "continuous_planning": False,
                    "callback": lambda slv, i=None: mcts_callback(slv),
                    "verbose": False,
                },
                "singleagent_solver_kwargs": {
                    "heuristic": lambda d, s: Value(
                        cost=sqrt((d._goal.x - s.x) ** 2 + (d._goal.y - s.y) ** 2)
                    ),
                    "use_labels": True,
                    "time_budget": 1000,
                    "max_depth": 100,
                    "continuous_planning": False,
                    "online_node_garbage": False,
                    "parallel": False,
                    "verbose": False,
                },
            },
        },
    ]

    # Load solvers (filtering out badly installed ones)
    solvers = map(
        lambda s: dict(s, entry=load_registered_solver(s["entry"])), try_solvers
    )
    solvers = list(filter(lambda s: s["entry"] is not None, solvers))
    solvers.insert(
        0, {"name": "Random Walk", "entry": None}
    )  # Add Random Walk as option

    # Run loop to ask user input
    domain = MultiAgentMaze()
    while True:
        # Ask user input to select solver
        choice = int(
            input(
                "\nChoose a solver:\n{solvers}\n".format(
                    solvers="\n".join(
                        ["0. Quit"]
                        + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers)]
                    )
                )
            )
        )
        if choice == 0:  # the user wants to quit
            break
        else:
            selected_solver = solvers[choice - 1]
            solver_type = selected_solver["entry"]
            # Test solver solution on domain
            print("==================== TEST SOLVER ====================")
            # Check if Random Walk selected or other
            if solver_type is None:
                rollout(
                    domain,
                    solver=None,
                    max_steps=1000,
                    outcome_formatter=lambda o: f"{o.observation} - cost: {sum(o.value[a].cost for a in o.observation):.2f}",
                )
            else:
                # Check that the solver is compatible with the domain
                assert solver_type.check_domain(domain)
                # Solve with selected solver
                with solver_type(**selected_solver["config"]) as solver:
                    MultiAgentMaze.solve_with(solver)
                    rollout(
                        domain,
                        solver,
                        max_steps=1000,
                        max_framerate=5,
                        outcome_formatter=lambda o: f"{o.observation} - cost: {sum(o.value[a].cost for a in o.observation):.2f}",
                        action_formatter=lambda a: f"{a}",
                    )
