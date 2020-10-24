# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import NamedTuple, Optional

from skdecide import DeterministicPlanningDomain, TransitionValue, Space
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import ListSpace, EnumSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver, rollout, load_registered_domain


class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class D(DeterministicPlanningDomain, UnrestrictedActions):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information given as part of an environment outcome


class MyDomain(D):

    def __init__(self, num_cols=10, num_rows=10):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:

        if action == Action.left:
            next_state = State(max(memory.x - 1, 0), memory.y)
        if action == Action.right:
            next_state = State(min(memory.x + 1, self.num_cols - 1), memory.y)
        if action == Action.up:
            next_state = State(memory.x, max(memory.y - 1, 0))
        if action == Action.down:
            next_state = State(memory.x, min(memory.y + 1, self.num_rows - 1))

        return next_state

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
        return ListSpace([State(x=self.num_cols - 1, y=self.num_rows - 1)])

    # def is_goal(self, state: D.T_state): # try to debug autocast issue in is_goal which appears in lrta*
    #     return self._get_goals_().contains(state)

    def _get_initial_state_(self) -> D.T_state:
        return State(x=0, y=0)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows])


if __name__ == '__main__':
    def heuristic(domain: MyDomain, st: State):
        return abs(st.x - domain._num_cols+1)+abs(st.y - domain._num_rows+1)

    try_solvers = [

        # Lazy A* (classical planning)
        {'name': 'Lazy A* (classical planning)',
         'entry': 'LazyAstar',
         'config': {'verbose': True, "heuristic": heuristic}},

        # LRTA* (classical planning)
        {'name': 'LRTAStar',
         'entry': 'LRTAstar',
         'config': {'verbose': True, "max_depth": 200, "max_iter": 1000, "heuristic": heuristic}},
    ]

    # Load solvers (filtering out badly installed ones)
    solvers = map(lambda s: dict(s, entry=load_registered_solver(s['entry'])), try_solvers)
    solvers = list(filter(lambda s: s['entry'] is not None, solvers))
    solvers.insert(0, {'name': 'Random Walk', 'entry': None})  # Add Random Walk as option

    # Run loop to ask user input
    #domain = MyDomain()  # MyDomain(5,5)

    d = {'name': 'Maze',
         'entry': 'Maze',
         'config': {},
         'rollout': {'max_steps': 1000, 'max_framerate': 30,
                     'outcome_formatter': lambda o: f'{o.observation} - cost: {o.value.cost:.2f}'}}
    domain_type = load_registered_domain(d['entry'])
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
                rollout(domain_type(), solver=None, max_steps=1000, max_framerate=50,
                        outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
            else:
                # Check that the solver is compatible with the domain
                # Solve with selected solver
                with solver_type(**selected_solver['config']) as solver:
                    domain_type.solve_with(solver)  # ,lambda:MyDomain(5,5))
                    rollout(domain_type(), solver, max_steps=1000, max_framerate=50,
                            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
