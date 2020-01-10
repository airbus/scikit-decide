# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from enum import Enum
from typing import NamedTuple, Optional, Any
from math import sqrt

import matplotlib.pyplot as plt
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from skdecide import DeterministicPlanningDomain, TransitionValue, Space
from skdecide.builders.domain import UnrestrictedActions, Renderable
from skdecide.hub.space.gym import ListSpace, EnumSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver, rollout


DEFAULT_MAZE = '''
+-+-+-+-+o+-+-+-+-+-+
|   |             | |
+ + + +-+-+-+ +-+ + +
| | |   |   | | |   |
+ +-+-+ +-+ + + + +-+
| |   |   | |   |   |
+ + + + + + + +-+ +-+
|   |   |   | |     |
+-+-+-+-+-+-+-+ +-+ +
|             |   | |
+ +-+-+-+-+ + +-+-+ +
|   |       |       |
+ + + +-+ +-+ +-+-+-+
| | |   |     |     |
+ +-+-+ + +-+ + +-+ +
| |     | | | |   | |
+-+ +-+ + + + +-+ + +
|   |   |   |   | | |
+ +-+ +-+-+-+-+ + + +
|   |       |     | |
+-+-+-+-+-+x+-+-+-+-+
'''


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information given as part of an environment outcome


class Maze(D):

    def __init__(self, maze_str = DEFAULT_MAZE):
        maze = []
        for y, line in enumerate(maze_str.strip().split('\n')):
            line = line.rstrip()
            row = []
            for x, c in enumerate(line):
                if c in {' ', 'o', 'x'}:
                    row.append(1)  # spaces are 1s
                    if c == 'o':
                        self._start = State(x, y)
                    if c == 'x':
                        self._goal = State(x, y)
                else:
                    row.append(0)  # walls are 0s
            maze.append(row)
        self._maze = maze
        self._num_cols = len(maze[0])
        self._num_rows = len(maze)
        self._ax = None

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:

        if action == Action.left:
            next_state = State(memory.x - 1, memory.y)
        if action == Action.right:
            next_state = State(memory.x + 1, memory.y)
        if action == Action.up:
            next_state = State(memory.x, memory.y - 1)
        if action == Action.down:
            next_state = State(memory.x, memory.y + 1)

        # If candidate next state is valid
        if 0 <= next_state.x < self._num_cols and 0 <= next_state.y < self._num_rows and self._maze[next_state.y][
                next_state.x] == 1:
            return next_state
        else:
            return memory

    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                              next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:

        if next_state.x == memory.x and next_state.y == memory.y:
            cost = 2  # big penalty when hitting a wall
        else:
            cost = abs(next_state.x - memory.x) + abs(next_state.y - memory.y)  # every move costs 1

        return TransitionValue(cost=cost)

    def _is_terminal(self, state: D.T_state) -> bool:
        return self._is_goal(state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(Action)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ListSpace([self._goal])

    def _get_initial_state_(self) -> D.T_state:
        return self._start

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self._num_cols, self._num_rows])

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        if self._ax is None:
            fig = plt.gcf()
            fig.canvas.set_window_title('Maze')
            ax = plt.axes()
            ax.set_aspect('equal')  # set the x and y axes to the same scale
            plt.xticks([])  # remove the tick marks by setting to an empty list
            plt.yticks([])  # remove the tick marks by setting to an empty list
            ax.invert_yaxis()  # invert the y-axis so the first row of data is at the top
            self._ax = ax
            plt.ion()
        maze = deepcopy(self._maze)
        maze[self._goal.y][self._goal.x] = 0.7
        maze[memory.y][memory.x] = 0.3
        self._ax.pcolormesh(maze)
        plt.pause(0.0001)


if __name__ == '__main__':

    try_solvers = [

        # Lazy A* (planning)
        {'name': 'Lazy A* (planning)',
         'entry': 'LazyAstar',
         'config': {'heuristic': lambda s, d: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'verbose': True}},
        
        # A* (planning)
        {'name': 'A* (planning)',
         'entry': 'Astar',
         'config': {'heuristic': lambda s, d: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'parallel': True, 'debug_logs': False}},
        
         # UCT
        {'name': 'UCT',
         'entry': 'UCT',
         'config': {'time_budget': 3600000, 'rollout_budget': 100000,
                    'max_depth': 1000, 'discount': 1.0, 'ucb_constant': 1.0 / sqrt(2.0),
                    'parallel': False, 'debug_logs': True}},

        # IW (planning)
        {'name': 'IW (planning)',
         'entry': 'IW',
         'config': {'state_features': lambda s, d: [s.x, s.y],
                    'use_state_feature_hash': False,
                    'parallel': True, 'debug_logs': False}},

        # BFWS (planning)
        {'name': 'BFWS (planning) - (num_rows * num_cols) binary encoding (1 binary variable <=> 1 cell)',
         'entry': 'BFWS',
         'config': {'state_binarizer': lambda s, d, f: f(s.x +  s.y * d._num_cols),
                    'heuristic': lambda s, d: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'termination_checker': lambda s, d: d.is_goal(s),
                    'parallel': True, 'debug_logs': False}},
        
        # BFWS (planning)
        {'name': 'BFWS (planning)  - (num_rows + num_cols) binary encoding (1 binary variable <=> 1 dimension value)',
         'entry': 'BFWS',
         'config': {'state_binarizer': lambda s, d, f: [f(s.x), f(s.y + d._num_cols)],
                    'heuristic': lambda s, d: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'termination_checker': lambda s, d: d.is_goal(s),
                    'parallel': True, 'debug_logs': False}},

        # PPO (deep reinforcement learning)
        {'name': 'PPO (deep reinforcement learning)',
         'entry': 'StableBaseline',
         'config': {'algo_class': PPO2, 'baselines_policy': MlpPolicy, 'learn_config': {'total_timesteps': 25000},
                    'verbose': 1}}
    ]

    # Load solvers (filtering out badly installed ones)
    solvers = map(lambda s: dict(s, entry=load_registered_solver(s['entry'])), try_solvers)
    solvers = list(filter(lambda s: s['entry'] is not None, solvers))
    solvers.insert(0, {'name': 'Random Walk', 'entry': None})  # Add Random Walk as option

    # Run loop to ask user input
    domain = Maze()
    while True:
        # Ask user input to select solver
        choice = int(input('\nChoose a solver:\n{solvers}\n'.format(
            solvers='\n'.join(['0. Quit'] + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers)]))))
        if choice == 0:  # the user wants to quit
            break
        else:
            selected_solver = solvers[choice - 1]
            solver_type = selected_solver['entry']
            # Check if Random Walk selected or other
            if solver_type is None:
                solver = None
            else:
                # Check that the solver is compatible with the domain
                assert solver_type.check_domain(domain)
                # Solve with selected solver
                solver = Maze.solve_with(lambda: solver_type(**selected_solver['config']))
            # Test solver solution on domain
            print('==================== TEST SOLVER ====================')
            rollout(domain, solver, max_steps=1000, max_framerate=30,
                    outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
