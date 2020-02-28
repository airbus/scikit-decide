from enum import Enum
from typing import *

from skdecide.hub.space.gym import ListSpace, EnumSpace

from skdecide import *
from skdecide.builders.domain import *


# Example of State type (adapt to your needs)
class State(NamedTuple):
    x: int
    y: int


# Example of Action type (adapt to your needs)
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
    T_info = None  # Type of additional information in environment outcome


class MyDomain(D):

    def __init__(self, start, end, maze_str):
        self.start = start
        self.end = end
        self.maze_str = maze_str.strip()
        self.maze = self.maze_str.splitlines()

    def _get_transition_value(self, memory: D.T_state, action: D.T_event, next_state: Optional[D.T_state] = None) -> \
            TransitionValue[D.T_value]:
        return TransitionValue(cost=1 if next_state != memory else 2)

    def _is_terminal(self, state: D.T_state) -> bool:
        return self._is_goal(state)

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        next_x, next_y = memory.x, memory.y
        if action == Action.up:
            next_x -= 1
        if action == Action.down:
            next_x += 1
        if action == Action.left:
            next_y -= 1
        if action == Action.right:
            next_y += 1
        return State(next_x, next_y) if self.maze[next_x][next_y] != '.' else memory

    def _get_action_space_(self) -> Space[D.T_event]:
        return EnumSpace(Action)

    def _get_goals_(self) -> Space[D.T_observation]:
        return ListSpace([self.end])

    def _get_initial_state_(self) -> D.T_state:
        return self.start

    def _get_observation_space_(self) -> Space[D.T_observation]:
        pass

    def _render_from(self, memory: D.T_state, **kwargs: Any) -> Any:
        cols = len(self.maze[0]) + 1
        pos = memory.x * cols + memory.y
        render = self.maze_str[:pos] + 'o' + self.maze_str[pos+1:]
        print(render)


# TEST
maze_str = '''
.....................
.   .             . .
. . . ....... ... . .
. . .   .   . . .   .
. ..... ... . . . ...
. .   .   . .   .   .
. . . . . . . ... ...
.   .   .   . .     .
............... ... .
.             .   . .
. ......... . ..... .
.   .       .       .
. . . ... ... .......
. . .   .     .     .
. ..... . ... . ... .
. .     . . . .   . .
... ... . . . ... . .
.   .   .   .   . . .
. ... ......... . . .
.   .       .     . .
.....................
'''

from skdecide.utils import rollout
domain = MyDomain(State(1, 1), State(19, 19), maze_str)
#rollout(domain, max_steps=100)

# SOLVE
from skdecide.hub.solver.lazy_astar import LazyAstar
solution = MyDomain.solve_with(LazyAstar, lambda: MyDomain(State(1, 1), State(19, 19), maze_str))
rollout(domain, solution, max_steps=100, max_framerate=10, verbose=False)
