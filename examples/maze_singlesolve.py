# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from enum import Enum
from typing import NamedTuple, Optional, Any
from math import sqrt
from multiprocessing import Value, Array

import matplotlib.pyplot as plt

from skdecide import DeterministicPlanningDomain, TransitionValue, Space
from skdecide.builders.domain import UnrestrictedActions, Renderable
from skdecide.hub.space.gym import ListSpace, EnumSpace, MultiDiscreteSpace
from skdecide.utils import rollout
from skdecide.hub.solver.astar import Astar


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


class MyShmProxy:

    _register_ = [(State, 2), (Action, 1), (EnumSpace, 1), (TransitionValue, 1), (bool, 1), (float, 1)]

    def __init__(self):
        self._proxies_ = {State: MyShmProxy.StateProxy, Action: MyShmProxy.ActionProxy,
                          EnumSpace: MyShmProxy.EnumSpaceProxy,
                          TransitionValue: MyShmProxy.TransitionValueProxy,
                          bool: MyShmProxy.BoolProxy,
                          float: MyShmProxy.FloatProxy}
    
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
            return Array('d', [0, 0], lock=True)
        
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
            return Value('I', 0, lock=True)
        
        @staticmethod
        def encode(action, shm_action):
            shm_action.value = action.value
        
        @staticmethod
        def decode(shm_action):
            return Action(shm_action.value)
    
    class EnumSpaceProxy:  # Always used with Action as enum class
        @staticmethod
        def initialize():
            return Array('c', b'')
        
        @staticmethod
        def encode(val, shm_val):
            pass
        
        @staticmethod
        def decode(val):
            return EnumSpace(Action)
    
    class TransitionValueProxy:

        @staticmethod
        def initialize():
            return [Value('d', 0), Value('b', False)]
        
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
    
    class BoolProxy:
        @staticmethod
        def initialize():
            return Value('b', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return bool(val.value)
    
    class FloatProxy:
        @staticmethod
        def initialize():
            return Value('d', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return float(val.value)


if __name__ == '__main__':

    domain_factory = lambda: Maze()
    domain = domain_factory()
    domain.reset()

    solver_factory = lambda: Astar(heuristic=lambda d, s: sqrt((d._goal.x - s.x)**2 + (d._goal.y - s.y)**2),
                                   parallel=True,
                                   shared_memory_proxy=MyShmProxy(),
                                   debug_logs=False)
    with solver_factory() as solver:
        Maze.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=1, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    
