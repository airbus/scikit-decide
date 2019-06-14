from enum import Enum
from typing import NamedTuple, Iterable
from math import sqrt
import getopt, sys

from airlaps import GoalMDP, Memory, TransitionValue, Space, ImplicitSpace, EnumerableSpace
from airlaps.builders.domain import UnrestrictedActionDomain
from airlaps.core import DiscreteDistribution
from airlaps.utils import rollout
from airlaps.catalog.solver.aostar import AOstar


class State(NamedTuple):
    x: int
    y: int
    t: int


class Actions(Enum):
    up_slow = 0
    up_fast = 1
    down_slow = 2
    down_fast = 3
    left_slow = 4
    left_fast = 5
    right_slow = 6
    right_fast = 7


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
                l.append(Actions.left_fast)
                l.append(Actions.left_slow)
            if self.state.x < (self.num_cols - 1):
                l.append(Actions.right_fast)
                l.append(Actions.right_slow)
            if self.state.y > 0:
                l.append(Actions.up_fast)
                l.append(Actions.up_slow)
            if self.state.y < (self.num_rows - 1):
                l.append(Actions.down_fast)
                l.append(Actions.down_slow)
            return l


T_state = State  # Type of states
T_observation = T_state  # Type of observations
T_event = Actions  # Type of events
T_value = float  # Type of transition values (rewards or costs)
T_info = None  # Type of additional information given as part of an environment outcome


class MyDomain(GoalMDP):

    def __init__(self, num_cols=10, num_rows=10, budget=20):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.budget = budget
    
    def _get_next_state_distribution(self, memory: Memory[T_state], event: T_event) -> DiscreteDistribution[T_state]:
        current_state = self.get_last_state(memory)

        if current_state.t == 0:
            next_state_distribution = DiscreteDistribution([(State(-1, -1, 0), 1.0)])

        if event == Actions.left_slow:
            next_state_distribution = DiscreteDistribution([(State(max(current_state.x - 1, 0), current_state.y, current_state.t - 1), 0.8),
                                                            (State(-1, -1, 0), 0.2)])
        if event == Actions.left_fast:
            next_state_distribution = DiscreteDistribution([(State(max(current_state.x - 1, 0), current_state.y, current_state.t - 1), 0.9),
                                                            (State(-1, -1, 0), 0.1)])
        if event == Actions.right_slow:
            next_state_distribution = DiscreteDistribution([(State(min(current_state.x + 1, self.num_cols - 1), current_state.y, current_state.t - 1), 0.8),
                                                            (State(-1, -1, 0), 0.2)])
        if event == Actions.right_fast:
            next_state_distribution = DiscreteDistribution([(State(min(current_state.x + 1, self.num_cols - 1), current_state.y, current_state.t - 1), 0.9),
                                                            (State(-1, -1, 0), 0.1)])
        if event == Actions.up_slow:
            next_state_distribution = DiscreteDistribution([(State(current_state.x, max(current_state.y - 1, 0), current_state.t - 1), 0.8),
                                                            (State(-1, -1, 0), 0.2)])
        if event == Actions.up_fast:
            next_state_distribution = DiscreteDistribution([(State(current_state.x, max(current_state.y - 1, 0), current_state.t - 1), 0.9),
                                                            (State(-1, -1, 0), 0.1)])
        if event == Actions.down_slow:
            next_state_distribution = DiscreteDistribution([(State(current_state.x, min(current_state.y + 1, self.num_rows - 1), current_state.t - 1), 0.8),
                                                            (State(-1, -1, 0), 0.2)])
        if event == Actions.down_fast:
            next_state_distribution = DiscreteDistribution([(State(current_state.x, min(current_state.y + 1, self.num_rows - 1), current_state.t - 1), 0.9),
                                                            (State(-1, -1, 0), 0.1)])

        return next_state_distribution

    def _get_transition_value(self, memory: Memory[T_state], event: T_event, next_state: T_state) -> TransitionValue[
            T_value]:
        current_state = self.get_last_state(memory)
        if next_state.x == -1 and next_state.y == -1:
            cost = 2 * (self.num_cols + self.num_rows) # dead-end state, penalty higher than optimal goal-reaching paths
        else:
            cost = abs(next_state.x - current_state.x) + abs(next_state.y - current_state.y)  # every move costs 1
        return TransitionValue(cost=cost)

    def is_terminal(self, state: T_state) -> bool:
        return self.is_goal(state) or (state.x == -1 and state.y == -1)

    def _get_action_space_(self) -> Space[T_event]:
        return ActionSpace()
    
    def _get_applicable_actions(self, memory: Memory[T_state]) -> Space[T_event]:
        return ActionSpace(self.get_last_state(memory), self.num_cols, self.num_rows)

    def _get_goals_(self) -> Space[T_observation]:
        # return ImplicitSpace(lambda s: s.x == self.num_cols - 1 and s.y == self.num_rows - 1)
        return ImplicitSpace(lambda s: True if (s.x==self.num_cols - 1 and s.y==self.num_rows - 1) or
                                               (s.x==-1 and s.y==-1) else False) # trick to consider dead-end state as a goal to  avoid modeling cycles

    def _get_initial_state_(self) -> T_state:
        return State(x=0, y=0, t=self.budget)

    def _get_observation_space_(self) -> Space[T_observation]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows, self.budget])


if __name__ == '__main__':

    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           "x:y:b:c:l:p:",
                                           ["rows=", "columns=", "budget", "detect_cycles=", "debug_logs=", "parallel="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    rows = 10
    columns = 10
    budget = 20
    detect_cycles = False
    debug_logs = False
    parallel = True

    for opt, arg in options:
        if opt in ('-x', '--rows'):
            rows = int(arg)
        elif opt in ('-y', '--columns'):
            columns = int(arg)
        elif opt in ('-b', '--budget'):
            budget = int(arg)
        elif opt in ('-c', '--detect_cycles'):
            detect_cycles = True if arg == 'yes' else False
        elif opt in ('-l', '--debug_logs'):
            debug_logs = True if arg == 'yes' else False
        elif opt in ('-p', '--parallel'):
            parallel = True if arg == 'yes' else False
    
    solver = AOstar(heuristic=lambda s, d: sqrt((s.x-(rows-1))*(s.x-(rows-1))+(s.y-(columns-1))*(s.y-(columns-1))),
                    parallel=parallel,
                    detect_cycles=detect_cycles,
                    debug_logs=debug_logs)
    solver.reset(lambda: MyDomain(rows, columns, budget))
    # Check that the solver is compatible with the domain
    assert solver.check_domain()
    solver.solve(Memory([State(x=0, y=0, t=budget)]))
    # Test solver solution on domain
    print('==================== TEST SOLVER ====================')
    rollout(MyDomain(rows, columns, budget), solver, max_steps=2*budget,
            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
