from enum import Enum
from typing import NamedTuple, Optional

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from airlaps import DeterministicPlanningDomain, Memory, TransitionValue, Space
from airlaps.builders.domain import UnrestrictedActionDomain
from airlaps.catalog.solver.lazyastar import LazyAstar
from airlaps.catalog.solver.randomwalk import RandomWalk
from airlaps.utils import rollout
from airlaps.wrappers.solver.baselines import BaselinesSolver
from airlaps.wrappers.space.gym import ListSpace, EnumSpace, MultiDiscreteSpace


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


class MyDomain(DeterministicPlanningDomain, UnrestrictedActionDomain):

    def __init__(self, num_cols=10, num_rows=10):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_next_state(self, memory: Memory[T_state], event: T_event) -> T_state:
        current_state = self.get_last_state(memory)

        if event == Actions.left:
            next_state = State(max(current_state.x - 1, 0), current_state.y)
        if event == Actions.right:
            next_state = State(min(current_state.x + 1, self.num_cols - 1), current_state.y)
        if event == Actions.up:
            next_state = State(current_state.x, max(current_state.y - 1, 0))
        if event == Actions.down:
            next_state = State(current_state.x, min(current_state.y + 1, self.num_rows - 1))

        return next_state

    def _get_transition_value(self, memory: Memory[T_state], event: T_event, next_state: Optional[T_state] = None) -> \
            TransitionValue[T_value]:
        current_state = self.get_last_state(memory)
        if next_state.x == current_state.x and next_state.y == current_state.y:
            cost = 2  # big penalty when hitting a wall
        else:
            cost = abs(next_state.x - current_state.x) + abs(next_state.y - current_state.y)  # every move costs 1
        return TransitionValue(cost=cost)

    def is_terminal(self, state: T_state) -> bool:
        return self.is_goal(state)

    def _get_action_space_(self) -> Space[T_event]:
        return EnumSpace(Actions)

    def _get_goals_(self) -> Space[T_observation]:
        # return ImplicitSpace(lambda s: s.x == self.num_cols - 1 and s.y == self.num_rows - 1)
        return ListSpace([State(x=self.num_cols - 1, y=self.num_rows - 1)])

    def _get_initial_state_(self) -> T_state:
        return State(x=0, y=0)

    def _get_observation_space_(self) -> Space[T_observation]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows])


if __name__ == '__main__':

    solvers = [{'name': 'Random walk', 'factory': RandomWalk, 'config': {}},
               {'name': 'Lazy A* (planning)', 'factory': LazyAstar,
                'config': {'from_observation': Memory([State(x=0, y=0)]), 'verbose': True}},
               {'name': 'PPO (deep reinforcement learning)',
                'factory': lambda: BaselinesSolver(PPO2, MlpPolicy, verbose=1), 'config': {'total_timesteps': 25000}}]

    while True:
        # Ask user input to select solver
        choice = int(input('\nChoose a solver:\n{solvers}\n'.format(
            solvers='\n'.join(['0. Quit'] + [f'{i + 1}. {s["name"]}' for i, s in enumerate(solvers)]))))
        if choice == 0:  # the user wants to quit
            break
        else:
            selected_solver = solvers[choice - 1]
            # Initialize solver
            solver = selected_solver['factory']()
            solver.reset(MyDomain)  # example for non-default initialization: solver.reset(lambda: MyDomain(5, 5))
            # Check that the solver is compatible with the domain
            assert solver.check_domain()
            # Solve with the given config parameters
            solver.solve(**selected_solver['config'])
            # Test solver solution on domain
            print('==================== TEST SOLVER ====================')
            rollout(MyDomain(), solver, max_steps=1000,
                    outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
