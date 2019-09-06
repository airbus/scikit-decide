from enum import IntEnum
from typing import NamedTuple, Optional, Iterable
from math import sqrt
import getopt, sys

from airlaps import hub, GoalMDPDomain, TransitionValue, \
     Space, EnumerableSpace, ImplicitSpace, DiscreteDistribution
from airlaps.builders.domain import Actions
from airlaps.utils import rollout

MultiDiscreteSpace = hub.load('MultiDiscreteSpace', folder='hub/space/gym')


class MyState(NamedTuple):
    x: int
    y: int
    t: int


class MyActions(IntEnum):
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
            return [a for a in MyActions]
        else:
            l = []
            if self.state.x > 0:
                l.append(MyActions.left_fast)
                l.append(MyActions.left_slow)
            if self.state.x < (self.num_cols - 1):
                l.append(MyActions.right_fast)
                l.append(MyActions.right_slow)
            if self.state.y > 0:
                l.append(MyActions.up_fast)
                l.append(MyActions.up_slow)
            if self.state.y < (self.num_rows - 1):
                l.append(MyActions.down_fast)
                l.append(MyActions.down_slow)
            return l


class D(GoalMDPDomain, Actions):
    T_state = MyState  # Type of states
    T_observation = MyState  # Type of observations
    T_event = MyActions  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information given as part of an environment outcome


class MyDomain(D):

    def __init__(self, num_cols=10, num_rows=10, budget=20):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.budget = budget
    
    def _get_next_state_distribution(self, memory: D.T_memory[D.T_state],
                                     action: D.T_agent[D.T_concurrency[D.T_event]]) -> DiscreteDistribution[D.T_state]:
        if memory.t == 0:
            next_state_distribution = DiscreteDistribution([(MyState(-1, -1, 0), 1.0)])

        if action == MyActions.left_slow:
            next_state_distribution = DiscreteDistribution([(MyState(max(memory.x - 1, 0), memory.y, memory.t - 1), 0.8),
                                                            (MyState(-1, -1, 0), 0.2)])
        if action == MyActions.left_fast:
            next_state_distribution = DiscreteDistribution([(MyState(max(memory.x - 1, 0), memory.y, memory.t - 1), 0.9),
                                                            (MyState(-1, -1, 0), 0.1)])
        if action == MyActions.right_slow:
            next_state_distribution = DiscreteDistribution([(MyState(min(memory.x + 1, self.num_cols - 1), memory.y, memory.t - 1), 0.8),
                                                            (MyState(-1, -1, 0), 0.2)])
        if action == MyActions.right_fast:
            next_state_distribution = DiscreteDistribution([(MyState(min(memory.x + 1, self.num_cols - 1), memory.y, memory.t - 1), 0.9),
                                                            (MyState(-1, -1, 0), 0.1)])
        if action == MyActions.up_slow:
            next_state_distribution = DiscreteDistribution([(MyState(memory.x, max(memory.y - 1, 0), memory.t - 1), 0.8),
                                                            (MyState(-1, -1, 0), 0.2)])
        if action == MyActions.up_fast:
            next_state_distribution = DiscreteDistribution([(MyState(memory.x, max(memory.y - 1, 0), memory.t - 1), 0.9),
                                                            (MyState(-1, -1, 0), 0.1)])
        if action == MyActions.down_slow:
            next_state_distribution = DiscreteDistribution([(MyState(memory.x, min(memory.y + 1, self.num_rows - 1), memory.t - 1), 0.8),
                                                            (MyState(-1, -1, 0), 0.2)])
        if action == MyActions.down_fast:
            next_state_distribution = DiscreteDistribution([(MyState(memory.x, min(memory.y + 1, self.num_rows - 1), memory.t - 1), 0.9),
                                                            (MyState(-1, -1, 0), 0.1)])

        return next_state_distribution

    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                              next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:
        if next_state.x == -1 and next_state.y == -1:
            cost = 2 * (self.num_cols + self.num_rows) # dead-end state, penalty higher than optimal goal-reaching paths
        else:
            cost = abs(next_state.x - memory.x) + abs(next_state.y - memory.y)  # every move costs 1
        return TransitionValue(cost=cost)

    def _is_terminal(self, state: D.T_state) -> bool:
        return self.is_goal(state) or (state.x == -1 and state.y == -1)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return ActionSpace()
    
    def _get_applicable_actions_from(self, memory: D.T_memory[D.T_state]) -> D.T_agent[Space[D.T_event]]:
        return ActionSpace(memory, self.num_cols, self.num_rows)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        # return ImplicitSpace(lambda s: s.x == self.num_cols - 1 and s.y == self.num_rows - 1)
        return ImplicitSpace(lambda s: True if (s.x==self.num_cols - 1 and s.y==self.num_rows - 1) or
                                               (s.x==-1 and s.y==-1) else False) # trick to consider dead-end state as a goal to  avoid modeling cycles

    def _get_initial_state_(self) -> D.T_state:
        return MyState(x=0, y=0, t=self.budget)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
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
    
    try_solvers = [

        # AO* (planning)
        {'name': 'AO* (planning)',
         'type': {'entry': 'AOstar', 'folder': 'hub/solver/aostar'},
         'config': {'parallel': True, 'discount': 1.0, 'max_tip_expanions': 1,
                     'detect_cycles': False, 'debug_logs': False,
                     'heuristic': lambda s, d: sqrt((s.x-(rows-1))*(s.x-(rows-1))+(s.y-(columns-1))*(s.y-(columns-1)))}}
    ]

    # Load solvers (if installed)
    solvers = []
    for s in try_solvers:
        try:
            if s['type'] is not None:
                s['type'] = hub.load(**s['type'])
            solvers.append(s)
        except Exception as e:
            print(rf'/!\ Could not load {s["name"]} from hub: check installation & missing dependencies')
            print('\nOriginal exception was:', e)
    
    # Run loop to ask user input
    domain = MyDomain(rows, columns, budget)
    while True:
        # Ask user input to select solver
        choice = int(input('\nChoose a solver:\n{solvers}\n'.format(
            solvers='\n'.join(['0. Quit'] + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers)]))))
        if choice == 0:  # the user wants to quit
            break
        else:
            selected_solver = solvers[choice - 1]
            solver_type = selected_solver['type']
            if solver_type is None:
                solver = None
            else:
                # Check that the solver is compatible with the domain
                assert solver_type.check_domain(domain)
                # Solve with selected solver
                solver = MyDomain.solve_with(lambda: solver_type(**selected_solver['config']))
            # Test solver solution on domain
            print('==================== TEST SOLVER ====================')
            rollout(domain, solver, max_steps=1000, max_framerate=30,
                    outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
