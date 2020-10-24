# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import NamedTuple, Optional
from pathos.helpers import mp
from math import sqrt

from stable_baselines3 import PPO

from skdecide import DeterministicPlanningDomain, TransitionValue, Space, \
                     EnvironmentOutcome, TransitionOutcome, SingleValueDistribution
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import ListSpace, EnumSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver, rollout


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

    def _get_initial_state_(self) -> D.T_state:
        return State(x=0, y=0)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows])


# Shared memory proxy for use with parallel algorithms only
# Not efficient on this tiny domain but provided for illustration
# To activate parallelism, set parallel=True in the algotihms below
class MyShmProxy:

    _register_ = [(State, 2), (Action, 1), (EnumSpace, 1), (SingleValueDistribution, 1),
                  (TransitionValue, 1), (EnvironmentOutcome, 1), (TransitionOutcome, 1),
                  (bool, 1), (int, 2), (float, 1), (list, 2)]

    def __init__(self):
        self._proxies_ = {State: MyShmProxy.StateProxy, Action: MyShmProxy.ActionProxy,
                          EnumSpace: MyShmProxy.EnumSpaceProxy,
                          SingleValueDistribution: MyShmProxy.SingleValueDistributionProxy,
                          TransitionValue: MyShmProxy.TransitionValueProxy,
                          EnvironmentOutcome: MyShmProxy.EnvironmentOutcomeProxy,
                          TransitionOutcome: MyShmProxy.TransitionOutcomeProxy,
                          bool: MyShmProxy.BoolProxy,
                          int: MyShmProxy.IntProxy,
                          float: MyShmProxy.FloatProxy,
                          list: MyShmProxy.ListProxy}
    
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
            return mp.Array('d', [0, 0], lock=True)
        
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
            return mp.Value('I', 0, lock=True)
        
        @staticmethod
        def encode(action, shm_action):
            shm_action.value = action.value
        
        @staticmethod
        def decode(shm_action):
            return Action(shm_action.value)
    
    class EnumSpaceProxy:  # Always used with Action as enum class
        @staticmethod
        def initialize():
            return mp.Array('c', b'')
        
        @staticmethod
        def encode(val, shm_val):
            pass
        
        @staticmethod
        def decode(val):
            return EnumSpace(Action)
    
    class SingleValueDistributionProxy:  # Always used with State
        @staticmethod
        def initialize():
            return MyShmProxy.StateProxy.initialize()
        
        @staticmethod
        def encode(svd, shm_svd):
            MyShmProxy.StateProxy.encode(svd._value, shm_svd)
        
        @staticmethod
        def decode(svd):
            return SingleValueDistribution(MyShmProxy.StateProxy.decode(svd))
    
    class TransitionValueProxy:

        @staticmethod
        def initialize():
            return [mp.Value('d', 0), mp.Value('b', False)]
        
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
    
    class EnvironmentOutcomeProxy:
        @staticmethod
        def initialize():
            return [MyShmProxy.StateProxy.initialize()] + \
                   MyShmProxy.TransitionValueProxy.initialize() + \
                   [MyShmProxy.BoolProxy.initialize()]
        
        @staticmethod
        def encode(outcome, shm_outcome):
            MyShmProxy.StateProxy.encode(outcome.observation, shm_outcome[0])
            MyShmProxy.TransitionValueProxy.encode(outcome.value, shm_outcome[1:3])
            MyShmProxy.BoolProxy.encode(outcome.termination, shm_outcome[3])
        
        @staticmethod
        def decode(outcome):
            return EnvironmentOutcome(observation=MyShmProxy.StateProxy.decode(outcome[0]),
                                      value=MyShmProxy.TransitionValueProxy.decode(outcome[1:3]),
                                      termination=MyShmProxy.BoolProxy.decode(outcome[3]))
    
    class TransitionOutcomeProxy:
        @staticmethod
        def initialize():
            return [MyShmProxy.StateProxy.initialize()] + \
                   MyShmProxy.TransitionValueProxy.initialize() + \
                   [MyShmProxy.BoolProxy.initialize()]
        
        @staticmethod
        def encode(outcome, shm_outcome):
            MyShmProxy.StateProxy.encode(outcome.state, shm_outcome[0])
            MyShmProxy.TransitionValueProxy.encode(outcome.value, shm_outcome[1:3])
            MyShmProxy.BoolProxy.encode(outcome.termination, shm_outcome[3])
        
        @staticmethod
        def decode(outcome):
            return TransitionOutcome(state=MyShmProxy.StateProxy.decode(outcome[0]),
                                     value=MyShmProxy.TransitionValueProxy.decode(outcome[1:3]),
                                     termination=MyShmProxy.BoolProxy.decode(outcome[3]))
    
    class BoolProxy:
        @staticmethod
        def initialize():
            return mp.Value('b', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return bool(val.value)
    
    class IntProxy:
        @staticmethod
        def initialize():
            return mp.Value('i', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return int(val.value)
    
    class FloatProxy:
        @staticmethod
        def initialize():
            return mp.Value('d', False)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val.value = val
        
        @staticmethod
        def decode(val):
            return float(val.value)
    
    class ListProxy:  # Always used to encode (R)IW state feature vector
        @staticmethod
        def initialize():
            return mp.Array('i', [0, 0], lock=True)
        
        @staticmethod
        def encode(val, shm_val):
            shm_val[0] = val[0]
            shm_val[1] = val[1]
        
        @staticmethod
        def decode(val):
            return [val[0], val[1]]


if __name__ == '__main__':

    try_solvers = [

        # Lazy A* (classical planning)
        {'name': 'Lazy A* (classical planning)',
         'entry': 'LazyAstar',
         'config': {'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                    'verbose': False}},
        
        # A* (planning)
        {'name': 'A* (planning)',
         'entry': 'Astar',
         'config': {'domain_factory': lambda: MyDomain(),
                    'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(), 'debug_logs': False}},

        # UCT (reinforcement learning / search)
        {'name': 'UCT (reinforcement learning / search)',
         'entry': 'UCT',
         'config': {'domain_factory': lambda: MyDomain(),
                    'time_budget': 1000, 'rollout_budget': 100,
                    'heuristic': lambda d, s: (-sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2), 10000),
                    'online_node_garbage': True,
                    'max_depth': 50, 'ucb_constant': 1.0 / sqrt(2.0),
                    'parallel': False, 'shared_memory_proxy': MyShmProxy()}},

        # PPO: Proximal Policy Optimization (deep reinforcement learning)
        {'name': 'PPO: Proximal Policy Optimization (deep reinforcement learning)',
         'entry': 'StableBaseline',
         'config': {'algo_class': PPO, 'baselines_policy': 'MlpPolicy', 'learn_config': {'total_timesteps': 30000},
                    'verbose': 1}},
        
        # Rollout-IW (classical planning)
        {'name': 'Rollout-IW (classical planning)',
         'entry': 'RIW',
         'config': {'domain_factory': lambda: MyDomain(),
                    'state_features': lambda d, s: [s.x, s.y],
                    'time_budget': 1000, 'rollout_budget': 100,
                    'max_depth': 50, 'exploration': 0.25,
                    'use_simulation_domain': True, 'online_node_garbage': True,
                    'continuous_planning': False,
                    'parallel': False, 'shared_memory_proxy': MyShmProxy()}},
        
        # IW (classical planning)
        {'name': 'IW (classical planning)',
         'entry': 'IW',
         'config': {'domain_factory': lambda: MyDomain(),
                    'state_features': lambda d, s: [s.x, s.y],
                    'parallel': False, 'shared_memory_proxy': MyShmProxy()}},
        
        # BFWS (classical planning)
        {'name': 'BFWS (planning) - (num_rows * num_cols) binary encoding (1 binary variable <=> 1 cell)',
         'entry': 'BFWS',
         'config': {'domain_factory': lambda: MyDomain(),
                    'state_features': lambda d, s: [s.x, s.y],
                    'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                    'termination_checker': lambda d, s: d.is_goal(s),
                    'parallel': False, 'shared_memory_proxy': MyShmProxy(),
                    'debug_logs': False}},
    ]

    # Load solvers (filtering out badly installed ones)
    solvers = map(lambda s: dict(s, entry=load_registered_solver(s['entry'])), try_solvers)
    solvers = list(filter(lambda s: s['entry'] is not None, solvers))
    solvers.insert(0, {'name': 'Random Walk', 'entry': None})  # Add Random Walk as option

    # Run loop to ask user input
    domain = MyDomain()  # MyDomain(5,5)
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
                rollout(domain, solver=None, max_steps=1000,
                        outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
            else:
                # Check that the solver is compatible with the domain
                assert solver_type.check_domain(domain)
                # Solve with selected solver
                with solver_type(**selected_solver['config']) as solver:
                    MyDomain.solve_with(solver)  # ,lambda:MyDomain(5,5))
                    rollout(domain, solver, max_steps=1000,
                            outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
