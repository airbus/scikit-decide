# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from enum import Enum
from skdecide.hub.solver.mcts.mcts import MCTS
from skdecide.hub.solver import mcts
from typing import NamedTuple, Optional, Any
from pathos.helpers import mp
from math import sqrt

import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from skdecide import DeterministicPlanningDomain, TransitionValue, Space, \
                     EnvironmentOutcome, TransitionOutcome, SingleValueDistribution
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


# Shared memory proxy for use with parallel algorithms only
# Not efficient on this tiny domain but provided for illustration
# To activate parallelism, set parallel=True in the algotihms below
class MyShmProxy:

    _register_ = [(State, 2), (Action, 1), (EnumSpace, 1), (SingleValueDistribution, 1),
                  (TransitionValue, 1), (EnvironmentOutcome, 1), (TransitionOutcome, 1),
                  (bool, 1), (int, 2), (float, 1), (list, 2)]

    def __init__(self):
        self._proxies_ = {State: MyShmProxy.StateProxy, Action: MyShmProxy.ActionProxy,
                          EnumSpace: MyShmProxy.EnumSpaceProxy,
                          SingleValueDistribution: MyShmProxy.SingleValueDistributionProxy,
                          TransitionValue: MyShmProxy.TransitionValueProxy,
                          EnvironmentOutcome: MyShmProxy.EnvironmentOutcomeProxy,
                          TransitionOutcome: MyShmProxy.TransitionOutcomeProxy,
                          bool: MyShmProxy.BoolProxy,
                          int: MyShmProxy.IntProxy,
                          float: MyShmProxy.FloatProxy,
                          list: MyShmProxy.ListProxy}
    
    def copy(self):
        p = MyShmProxy()
        p._proxies_ = dict(self._proxies_)
        return p
    
    def register(self):
        return MyShmProxy._register_

    def initialize(self, t):
        return self._proxies_[t].initialize()
    
    def encode(self, value, shm_value):
        self._proxies_[type(value)].encode(value, shm_value)
    
    def decode(self, t, shm_value):
        return self._proxies_[t].decode(shm_value)

    class StateProxy:
        @staticmethod
        def initialize():
            return mp.Array('d', [0, 0], lock=True)
        
        @staticmethod
        def encode(state, shm_state):
            shm_state[0] = state.x
            shm_state[1] = state.y
        
        @staticmethod
        def decode(shm_state):
            return State(int(shm_state[0]), int(shm_state[1]))
    
    class ActionProxy:
        @staticmethod
        def initialize():
            return mp.Value('I', 0, lock=True)
        
        @staticmethod
        def encode(action, shm_action):
            shm_action.value = action.value
        
        @staticmethod
        def decode(shm_action):
            return Action(shm_action.value)
    
    class EnumSpaceProxy:  # Always used with Action as enum class
        @staticmethod
        def initialize():
            return mp.Array('c', b'')
        
        @staticmethod
        def encode(val, shm_val):
            pass
        
        @staticmethod
        def decode(val):
            return EnumSpace(Action)
    
    class SingleValueDistributionProxy:  # Always used with State
        @staticmethod
        def initialize():
            return MyShmProxy.StateProxy.initialize()
        
        @staticmethod
        def encode(svd, shm_svd):
            MyShmProxy.StateProxy.encode(svd._value, shm_svd)
        
        @staticmethod
        def decode(svd):
            return SingleValueDistribution(MyShmProxy.StateProxy.decode(svd))
    
    class TransitionValueProxy:

        @staticmethod
        def initialize():
            return [mp.Value('d', 0), mp.Value('b', False)]
        
        @staticmethod
        def encode(value, shm_value):
            if value.reward is not None:
                shm_value[0] = value.reward
                shm_value[1] = True
            elif value.cost is not None:
                shm_value[0] = value.cost
                shm_value[1] = False
            else:
                shm_value[0] = 0
                shm_value[1] = True
        
        @staticmethod
        def decode(value):
            if value[1].value:
                return TransitionValue(reward=value[0].value)
            else:
                return TransitionValue(cost=value[0].value)
    
    class EnvironmentOutcomeProxy:
        @staticmethod
        def initialize():
            return [MyShmProxy.StateProxy.initialize()] + \
                   MyShmProxy.TransitionValueProxy.initialize() + \
                   [MyShmProxy.BoolProxy.initialize()]
        
        @staticmethod
        def encode(outcome, shm_outcome):
            MyShmProxy.StateProxy.encode(outcome.observation, shm_outcome[0])
            MyShmProxy.TransitionValueProxy.encode(outcome.value, shm_outcome[1:3])
            MyShmProxy.BoolProxy.encode(outcome.termination, shm_outcome[3])
        
        @staticmethod
        def decode(outcome):
            return EnvironmentOutcome(observation=MyShmProxy.StateProxy.decode(outcome[0]),
                                      value=MyShmProxy.TransitionValueProxy.decode(outcome[1:3]),
                                      termination=MyShmProxy.BoolProxy.decode(outcome[3]))
    
    class TransitionOutcomeProxy:
        @staticmethod
        def initialize():
            return [MyShmProxy.StateProxy.initialize()] + \
                   MyShmProxy.TransitionValueProxy.initialize() + \
                   [MyShmProxy.BoolProxy.initialize()]
        
        @staticmethod
        def encode(outcome, shm_outcome):
            MyShmProxy.StateProxy.encode(outcome.state, shm_outcome[0])
            MyShmProxy.TransitionValueProxy.encode(outcome.value, shm_outcome[1:3])
            MyShmProxy.BoolProxy.encode(outcome.termination, shm_outcome[3])
        
        @staticmethod
        def decode(outcome):
            return TransitionOutcome(state=MyShmProxy.StateProxy.decode(outcome[0]),
                                     value=MyShmProxy.TransitionValueProxy.decode(outcome[1:3]),
                                     termination=MyShmProxy.BoolProxy.decode(outcome[3]))
    
    class BoolProxy:
        @staticmethod
        def initialize():
            return mp.Value('b', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return bool(val.value)
    
    class IntProxy:
        @staticmethod
        def initialize():
            return mp.Value('i', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return int(val.value)
    
    class FloatProxy:
        @staticmethod
        def initialize():
            return mp.Value('d', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return float(val.value)
    
    class ListProxy:  # Always used to encode (R)IW state feature vector
        @staticmethod
        def initialize():
            return mp.Array('i', [0, 0], lock=True)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val[0] = val[0]
            shm_val[1] = val[1]
        
        @staticmethod
        def decode(val):
            return [val[0], val[1]]


if __name__ == '__main__':

    try_solvers = [

        # Lazy A* (planning)
        {'name': 'Lazy A* (planning)',
         'entry': 'LazyAstar',
         'config': {'heuristic': lambda d, s: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'verbose': True}},
        
        # A* (planning)
        {'name': 'A* (planning)',
         'entry': 'Astar',
         'config': {'domain_factory': lambda: Maze(),
                    'heuristic': lambda d, s: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(),
                    'debug_logs': False}},
        
         # UCT
        {'name': 'UCT',
         'entry': 'UCT',
         'config': {'domain_factory': lambda: Maze(),
                    'time_budget': 1000, 'rollout_budget': 100,
                    'max_depth': 50, 'discount': 1.0, 'ucb_constant': 1.0 / sqrt(2.0),
                    'heuristic': lambda d, s: (-sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2), 10000),
                    'continuous_planning': True, 'online_node_garbage': True,
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(),
                    'debug_logs': False}},

        # IW (planning)
        {'name': 'IW (planning)',
         'entry': 'IW',
         'config': {'domain_factory': lambda: Maze(),
                    'state_features': lambda d, s: [s.x, s.y],
                    'use_state_feature_hash': False,
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(),
                    'debug_logs': False}},
        
        # Rollout-IW (classical planning)
        {'name': 'Rollout-IW (classical planning)',
         'entry': 'RIW',
         'config': {'domain_factory': lambda: Maze(),
                    'state_features': lambda d, s: [s.x, s.y],
                    'time_budget': 1000, 'rollout_budget': 100,
                    'max_depth': 50, 'exploration': 0.25,
                    'use_simulation_domain': True, 'online_node_garbage': True,
                    'continuous_planning': False,
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(),
                    'debug_logs': False}},

        # BFWS (planning)
        {'name': 'BFWS (planning) - (num_rows * num_cols) binary encoding (1 binary variable <=> 1 cell)',
         'entry': 'BFWS',
         'config': {'domain_factory': lambda: Maze(),
                    'state_features': lambda d, s: [s.x, s.y],
                    'heuristic': lambda d, s: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                    'termination_checker': lambda d, s: d.is_goal(s),
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(),
                    'debug_logs': False}},

        # PPO (deep reinforcement learning)
        {'name': 'PPO (deep reinforcement learning)',
         'entry': 'StableBaseline',
         'config': {'algo_class': PPO, 'baselines_policy': 'MlpPolicy', 'learn_config': {'total_timesteps': 30000},
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
            # Test solver solution on domain
            print('==================== TEST SOLVER ====================')
            # Check if Random Walk selected or other
            if solver_type is None:
                rollout(domain, solver=None, max_steps=1000, max_framerate=30,
                        outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
            else:
                # Check that the solver is compatible with the domain
                assert solver_type.check_domain(domain)
                # Solve with selected solver
                with solver_type(**selected_solver['config']) as solver:
                    Maze.solve_with(solver)
                    rollout(domain, solver, max_steps=1000, max_framerate=30,
                            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
