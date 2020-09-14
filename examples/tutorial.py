# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Example 4: Create a maze domain and solve it"""

# %%
'''
Import modules.
'''

# %%
from enum import Enum
from typing import *

from skdecide import *
from skdecide.builders.domain import *
from skdecide.utils import rollout
from skdecide.hub.space.gym import ListSpace, EnumSpace
from skdecide.hub.solver.lazy_astar import LazyAstar

# %%
'''
Define your state space (agent positions) & action space (agent movements).
'''

# %%
class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3

# %%
'''
Define your domain type from a base template (DeterministicPlanningDomain here) with optional refinements (UnrestrictedActions & Renderable here).
'''

# %%
class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome

# %%
'''
Implement the maze domain by filling all non-implemented methods and adding a constructor to define the maze & start/end positions.
'''

# %%
class MyDomain(D):

    def __init__(self, start, end, maze_str):
        self.start = start
        self.end = end
        self.maze_str = maze_str.strip()
        self.maze = self.maze_str.splitlines()

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        # Move agent according to action (except if bumping into a wall)
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

    def _get_transition_value(self, memory: D.T_state, action: D.T_event, next_state: Optional[D.T_state] = None) -> \
            TransitionValue[D.T_value]:
        # Set cost to 1 when moving (energy cost) and to 2 when bumping into a wall (damage cost)
        return TransitionValue(cost=1 if next_state != memory else 2)

    def _get_initial_state_(self) -> D.T_state:
        # Set the start position as initial state
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        # Set the end position as goal
        return ListSpace([self.end])

    def _is_terminal(self, state: D.T_state) -> bool:
        # Stop an episode only when goal reached
        return self._is_goal(state)

    def _get_action_space_(self) -> Space[D.T_event]:
        # Define action space
        return EnumSpace(Action)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        # Define observation space (not mandatory here)
        pass

    def _render_from(self, memory: D.T_state, **kwargs: Any) -> Any:
        # Print the maze in console with agent represented by 'o'
        cols = len(self.maze[0]) + 1
        pos = memory.x * cols + memory.y
        render = self.maze_str[:pos] + 'o' + self.maze_str[pos+1:]
        print(render)

# %%
'''
Define a maze and test a random walk inside.
'''

# %%
# Maze example ('.' represent walls, ' ' represent free space)
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

# Start top-left, try to reach bottom-right of this maze
domain = MyDomain(State(1, 1), State(19, 19), maze_str)

# Random walk in the maze (may sometimes reach the goal by chance)
rollout(domain, max_steps=100, render=False)

# %%
'''
Pick a solver (lazy A*) and solve the maze optimally.
'''

# %%
# Check solver compatibility with the domain
assert LazyAstar.check_domain(domain)

# Compute solution and visualize it
with LazyAstar() as solver:
    MyDomain.solve_with(solver, lambda: MyDomain(State(1, 1), State(19, 19), maze_str))
    rollout(domain, solver, max_steps=100, max_framerate=10, verbose=False)
