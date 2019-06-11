# TODO: Update this example according to latest changes

from enum import Enum

from airlaps import GoalMDP, Memory, TransitionValue, Distribution, DiscreteDistribution, nocopy
from airlaps.dataclasses import dataclass, \
    replace  # TODO: replace 'airlaps.dataclasses' by 'dataclasses' once transitioned to Python 3.7


# @nocopy avoids copy to optimize memory since State is recursively immutable
# (i.e. copy.copy(state) or copy.deepcopy(state) will actually return state instead of a copy)
# @dataclass: frozen=True makes State immutable (avoiding potential side effects of modifying it directly) and hashable
# if all his attributes are (useful e.g. for putting it in a Set or as Dict keys)
@nocopy
@dataclass(frozen=True)
class State:
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


class GridGoalMdpDomain(GoalMDP):

    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols

    def get_next_state_distribution(self, memory: Memory[T_state], event: T_event) -> Distribution[T_state]:
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

        return DiscreteDistribution([(next_state, 0.9), (current_state, 0.1)])  # 90% chance of success in moving

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


# Test the domain
test_as_env = True
n_rows = 5
n_cols = 10
gridGoalMdpDomain = GridGoalMdpDomain(n_rows=n_rows, n_cols=n_cols)
if test_as_env:
    print('Initial observation:', gridGoalMdpDomain.reset())
    while True:
        action = input('Please enter action (%s): ' % ', '.join([a.name for a in Actions]))
        try:
            print(gridGoalMdpDomain.step(getattr(Actions, action)))
        except AttributeError:
            print('This action is unknown')
else:
    while True:
        x = -1
        while x not in range(n_cols):
            input_x = input('Please enter x value (between 0 and %s): ' % str(n_cols - 1))
            try:
                x = int(input_x)
            except ValueError:
                print('Invalid value')
        y = -1
        while y not in range(n_rows):
            input_y = input('Please enter y value (between 0 and %s): ' % str(n_rows - 1))
            try:
                y = int(input_y)
            except ValueError:
                print('Invalid value')
        action = input('Please enter action (%s): ' % ', '.join([a.name for a in Actions]))
        print(gridGoalMdpDomain.sample(Memory([{'x': x, 'y': y}]), getattr(Actions, action)))
