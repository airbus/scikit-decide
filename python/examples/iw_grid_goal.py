# TODO: Update this example according to latest changes

from enum import Enum
from typing import NamedTuple
import getopt, sys
import numpy as np

from airlaps import DeterministicPlanningDomain, Memory, TransitionValue, Domain
from airlaps.builders.domain import EnumerableTransitionDomain, ActionDomain, GoalDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain, FullyObservableDomain
from airlaps.utils import rollout
from airlaps.catalog.solver.iw import IW


class State(NamedTuple):
    x: int
    y: int


class Actions(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


T_state = State  # Type of states
T_observation = T_state  # Type of observations
T_event = Actions  # Type of events
T_value = float  # Type of transition values (rewards or costs)
T_info = None  # Type of additional information given as part of an environment outcome


class GridGoalDomain(DeterministicPlanningDomain):

    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols

    def get_next_state(self, memory: Memory[T_state], event: T_event) -> T_state:
        current_state = self.get_last_state(memory)

        if event == Actions.up and current_state.y > 0:
            next_state = replace(current_state, y=current_state.y - 1)
        elif event == Actions.down and current_state.y < self.n_rows - 1:
            next_state = replace(current_state, y=current_state.y + 1)
        elif event == Actions.left and current_state.x > 0:
            next_state = replace(current_state, x=current_state.x - 1)
        elif event == Actions.right and current_state.x < self.n_cols - 1:
            next_state = replace(current_state, x=current_state.x + 1)
        else:
            next_state = current_state

        return next_state

    def get_transition_value(self, memory: Memory[T_state], event: T_event, next_state: T_state) -> TransitionValue[
        T_value]:
        current_state = self.get_last_state(memory)
        if next_state.x != current_state.x or next_state.y != current_state.y:
            # movement cost
            return TransitionValue(cost=1)
        else:
            # no cost
            return TransitionValue()

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
                parallel=parallel,
                debug_logs=debug_logs)
    solver.reset(lambda: GridGoalDomain(rows, columns))
    # Check that the solver is compatible with the domain
    assert solver.check_domain()
    try :
        solver.solve(Memory([State(x=0, y=0)]))
    except Exception as e:
        print("Oops!", e)
    # Test solver solution on domain
    print('==================== TEST SOLVER ====================')
    rollout(GridGoalDomain(rows, columns), solver, max_steps=1000,
            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')