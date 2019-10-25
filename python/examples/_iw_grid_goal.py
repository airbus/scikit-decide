# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Update this example according to latest changes

from enum import IntEnum
from typing import NamedTuple, Iterable
import getopt, sys
import numpy as np

from airlaps import DeterministicPlanningDomain, Memory, TransitionValue, Domain, Space, ImplicitSpace, EnumerableSpace
from airlaps.builders.domain import EnumerableTransitionDomain, ActionDomain, GoalDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain, FullyObservableDomain
from airlaps.utils import rollout
from airlaps.catalog.solver.iw import IW


class State(NamedTuple):
    x: int
    y: int


class Actions(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3


class ActionSpace(EnumerableSpace):
    def __init__(self, state=None, num_cols=0, num_rows=0):
        self.state = state
        self.num_cols = num_cols
        self.num_rows = num_rows

    def get_elements(self) -> Iterable[int]:
        if self.state is None:
            return [a for a in Actions]
        else:
            l = []
            if self.state.x > 0:
                l.append(Actions.left)
            if self.state.x < (self.num_cols - 1):
                l.append(Actions.right)
            if self.state.y > 0:
                l.append(Actions.up)
            if self.state.y < (self.num_rows - 1):
                l.append(Actions.down)
            return l


T_state = State  # Type of states
T_observation = T_state  # Type of observations
T_event = Actions  # Type of events
T_value = float  # Type of transition values (rewards or costs)
T_info = None  # Type of additional information given as part of an environment outcome


class GridGoalDomain(DeterministicPlanningDomain):

    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _get_next_state(self, memory: Memory[T_state], event: T_event) -> T_state:
        current_state = self.get_last_state(memory)

        if event == Actions.up and current_state.y > 0:
            next_state = State(x=current_state.x, y=current_state.y - 1)
        elif event == Actions.down and current_state.y < self.n_rows - 1:
            next_state = State(x=current_state.x, y=current_state.y + 1)
        elif event == Actions.left and current_state.x > 0:
            next_state = State(x=current_state.x - 1, y=current_state.y)
        elif event == Actions.right and current_state.x < self.n_cols - 1:
            next_state = State(x=current_state.x + 1, y=current_state.y)
        else:
            next_state = current_state

        return next_state

    def _get_transition_value(self, memory: Memory[T_state], event: T_event, next_state: T_state) -> TransitionValue[
        T_value]:
        current_state = self.get_last_state(memory)
        return TransitionValue(cost=1 if not self.is_terminal(current_state) else 0)

    def is_terminal(self, state: T_state) -> bool:
        return self._is_in_top_left_corner(state)

    def is_goal(self, observation: T_observation) -> bool:
        state = observation  # since our domain is fully observed (GoalMDP inherits FullyObservableDomain)
        return self._is_in_top_left_corner(state)

    def _get_initial_state_(self) -> T_state:
        # Set the initial state to bottom right
        return State(x=self.n_cols - 1, y=self.n_rows - 1)

    def _is_in_top_left_corner(self, state):
        return state.x == 0 and state.y == 0
    
    def _get_action_space_(self) -> Space[T_event]:
        return ActionSpace()
    
    def _get_applicable_actions(self, memory: Memory[T_state]) -> Space[T_event]:
        return ActionSpace(self.get_last_state(memory), self.n_cols, self.n_rows)
    
    def _get_goals_(self) -> Space[T_observation]:
        return ImplicitSpace(lambda s: self.is_terminal(s))


if __name__ == '__main__':

    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           "x:y:a:l:p:",
                                           ["rows=", "columns=", "algorithm", "debug_logs=", "parallel="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    rows = 10
    columns = 10
    algorithm = 'bfs'
    debug_logs = False
    parallel = True

    for opt, arg in options:
        if opt in ('-x', '--rows'):
            rows = int(arg)
        elif opt in ('-y', '--columns'):
            columns = int(arg)
        elif opt in ('-a', '--algorithm'):
            algorithm = arg
        elif opt in ('-l', '--debug_logs'):
            debug_logs = True if arg == 'yes' else False
        elif opt in ('-p', '--parallel'):
            parallel = True if arg == 'yes' else False
    
    solver = IW(planner=algorithm,
                state_to_feature_atoms_encoder=lambda s, d: np.array([s.x, s.y], dtype=np.int64),
                num_tracked_atoms=10,
                parallel=parallel,
                debug_logs=debug_logs)
    solver.reset(lambda: GridGoalDomain(rows, columns))
    # Check that the solver is compatible with the domain
    assert solver.check_domain()
    try :
        solver.solve(Memory([solver._domain.get_initial_state()]))
    except Exception as e:
        print("Oops!", e)
    # Test solver solution on domain
    print('==================== TEST SOLVER ====================')
    rollout(GridGoalDomain(rows, columns), solver, max_steps=1000,
            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    