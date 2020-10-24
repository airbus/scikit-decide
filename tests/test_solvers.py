# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import inspect

from enum import Enum
from typing import NamedTuple, Optional
from math import sqrt
from pathos.helpers import mp

from stable_baselines3 import PPO

from skdecide import DeterministicPlanningDomain, TransitionValue, \
                     Space, ImplicitSpace, \
                     EnvironmentOutcome, TransitionOutcome, \
                     SingleValueDistribution
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import EnumSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver


# Must be defined outside the grid_domain() fixture
# so that parallel domains can pickle it
# /!\ Is it worth defining the domain as a fixture?
class State(NamedTuple):
    x: int
    y: int
    s: int  # step => to make the domain cycle-free for algorithms like AO*


# Must be defined outside the grid_domain() fixture
# so that parallel domains can pickle it
# /!\ Is it worth defining the domain as a fixture?
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


class GridDomain(D):

    def __init__(self, num_cols=10, num_rows=10):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:

        if action == Action.left:
            next_state = State(max(memory.x - 1, 0), memory.y, memory.s + 1)
        if action == Action.right:
            next_state = State(min(memory.x + 1, self.num_cols - 1), memory.y, memory.s + 1)
        if action == Action.up:
            next_state = State(memory.x, max(memory.y - 1, 0), memory.s + 1)
        if action == Action.down:
            next_state = State(memory.x, min(memory.y + 1, self.num_rows - 1), memory.s + 1)

        return next_state

    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                            next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:

        if next_state.x == memory.x and next_state.y == memory.y:
            cost = 2  # big penalty when hitting a wall
        else:
            cost = abs(next_state.x - memory.x) + abs(next_state.y - memory.y)  # every move costs 1

        return TransitionValue(cost=cost)

    def _is_terminal(self, state: D.T_state) -> bool:
        return self._is_goal(state) or state.s >= 100

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return EnumSpace(Action)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda state: state.x == (self.num_cols - 1) and state.y == (self.num_rows - 1))

    def _get_initial_state_(self) -> D.T_state:
        return State(x=0, y=0, s=0)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return MultiDiscreteSpace([self.num_cols, self.num_rows, 100])


# FIXTURES

@pytest.fixture(params=[{'entry': 'Astar',
                         'config': {'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                                    'debug_logs': False},
                         'optimal': True},
                        {'entry': 'AOstar',
                         'config': {'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                                    'debug_logs': False},
                          'optimal': True},
                        {'entry': 'BFWS',
                         'config': {'state_features': lambda d, s: (s.x, s.y),
                                    'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                                    'termination_checker': lambda d, s: d.is_goal(s),
                                    'debug_logs': False},
                         'optimal': True},
                        {'entry': 'IW',
                         'config': {'state_features': lambda d, s: (s.x, s.y),
                                    'debug_logs': False},
                         'optimal': True},
                        {'entry': 'RIW',
                         'config': {'state_features': lambda d, s: (s.x, s.y),
                                    'time_budget': 20,
                                    'rollout_budget': 10,
                                    'max_depth': 10,
                                    'exploration': 0.25,
                                    'use_simulation_domain': True,
                                    'online_node_garbage': True,
                                    'continuous_planning': True,
                                    'debug_logs': False},
                         'optimal': False},
                        {'entry': 'UCT',
                         'config': {'time_budget': 20,
                                    'rollout_budget': 10,
                                    'max_depth': 10,
                                    'continuous_planning': True,
                                    'debug_logs': False},
                         'optimal': False},
                         {'entry': 'LRTDP',
                          'config': {'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                                     'use_labels': True,
                                     'time_budget': 60000,
                                     'rollout_budget': 10000,
                                     'max_depth': 500,
                                     'discount': 1.0,
                                     'epsilon': 0.001,
                                     'online_node_garbage': True,
                                     'continuous_planning': False,
                                     'debug_logs': False},
                          'optimal': True},
                          {'entry': 'ILAOstar',
                           'config': {'heuristic': lambda d, s: sqrt((d.num_cols - 1 - s.x)**2 + (d.num_rows - 1 - s.y)**2),
                                      'discount': 1.0,
                                      'epsilon': 0.001,
                                      'debug_logs': False},
                            'optimal': True}])
def solver_cpp(request):
    return request.param


@pytest.fixture(params=[{'entry': 'LazyAstar',
                         'config': {'verbose': False},
                         'optimal': True},
                        {'entry': 'StableBaseline',
                         'config': {'algo_class': PPO,
                                    'baselines_policy': 'MlpPolicy',
                                    'learn_config': {'total_timesteps': 10},
                                    'verbose': 1},
                         'optimal': False}])
def solver_python(request):
    return request.param


@pytest.fixture(params=[False, True])
def parallel(request):
    return request.param


@pytest.fixture(params=[False, True])
def shared_memory(request):
    return request.param


# HELPER FUNCTION

def get_plan(domain, solver):
    plan = []
    cost = 0
    observation = domain.reset()
    nb_steps = 0
    while (not domain.is_goal(observation)) and nb_steps < 20:
        plan.append(solver.sample_action(observation))
        outcome = domain.step(plan[-1])
        cost += outcome.value.cost
        observation = outcome.observation
        nb_steps += 1
    return plan, cost


# SHARED MEMORY PROXY FOR PARALLEL TESTS

class GridShmProxy:

    _register_ = [(State, 2), (Action, 1), (EnumSpace, 1), (SingleValueDistribution, 1),
                  (TransitionValue, 1), (EnvironmentOutcome, 1), (TransitionOutcome, 1),
                  (bool, 1), (float, 1), (int, 2)]

    def __init__(self):
        self._proxies_ = {State: GridShmProxy.StateProxy, Action: GridShmProxy.ActionProxy,
                          EnumSpace: GridShmProxy.EnumSpaceProxy,
                          SingleValueDistribution: GridShmProxy.SingleValueDistributionProxy,
                          TransitionValue: GridShmProxy.TransitionValueProxy,
                          EnvironmentOutcome: GridShmProxy.EnvironmentOutcomeProxy,
                          TransitionOutcome: GridShmProxy.TransitionOutcomeProxy,
                          bool: GridShmProxy.BoolProxy,
                          float: GridShmProxy.FloatProxy,
                          int: GridShmProxy.IntProxy}
    
    def copy(self):
        p = GridShmProxy()
        p._proxies_ = dict(self._proxies_)
        return p
    
    def register(self):
        return GridShmProxy._register_

    def initialize(self, t):
        return self._proxies_[t].initialize()
    
    def encode(self, value, shm_value):
        self._proxies_[type(value)].encode(value, shm_value)
    
    def decode(self, t, shm_value):
        return self._proxies_[t].decode(shm_value)

    class StateProxy:
        @staticmethod
        def initialize():
            return mp.Array('d', [0, 0, 0], lock=True)
        
        @staticmethod
        def encode(state, shm_state):
            shm_state[0] = state.x
            shm_state[1] = state.y
            shm_state[2] = state.s
        
        @staticmethod
        def decode(shm_state):
            return State(int(shm_state[0]), int(shm_state[1]), int(shm_state[2]))
    
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
            return GridShmProxy.StateProxy.initialize()
        
        @staticmethod
        def encode(svd, shm_svd):
            GridShmProxy.StateProxy.encode(svd._value, shm_svd)
        
        @staticmethod
        def decode(svd):
            return SingleValueDistribution(GridShmProxy.StateProxy.decode(svd))
    
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
            return [GridShmProxy.StateProxy.initialize()] + \
                   GridShmProxy.TransitionValueProxy.initialize() + \
                   [GridShmProxy.BoolProxy.initialize()]
        
        @staticmethod
        def encode(outcome, shm_outcome):
            GridShmProxy.StateProxy.encode(outcome.observation, shm_outcome[0])
            GridShmProxy.TransitionValueProxy.encode(outcome.value, shm_outcome[1:3])
            GridShmProxy.BoolProxy.encode(outcome.termination, shm_outcome[3])
        
        @staticmethod
        def decode(outcome):
            return EnvironmentOutcome(observation=GridShmProxy.StateProxy.decode(outcome[0]),
                                      value=GridShmProxy.TransitionValueProxy.decode(outcome[1:3]),
                                      termination=GridShmProxy.BoolProxy.decode(outcome[3]))
    
    class TransitionOutcomeProxy:
        @staticmethod
        def initialize():
            return [GridShmProxy.StateProxy.initialize()] + \
                   GridShmProxy.TransitionValueProxy.initialize() + \
                   [GridShmProxy.BoolProxy.initialize()]
        
        @staticmethod
        def encode(outcome, shm_outcome):
            GridShmProxy.StateProxy.encode(outcome.state, shm_outcome[0])
            GridShmProxy.TransitionValueProxy.encode(outcome.value, shm_outcome[1:3])
            GridShmProxy.BoolProxy.encode(outcome.termination, shm_outcome[3])
        
        @staticmethod
        def decode(outcome):
            return TransitionOutcome(state=GridShmProxy.StateProxy.decode(outcome[0]),
                                     value=GridShmProxy.TransitionValueProxy.decode(outcome[1:3]),
                                     termination=GridShmProxy.BoolProxy.decode(outcome[3]))
    
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

# TESTS

def do_test_cpp(solver_cpp, parallel, shared_memory, result):
    noexcept = True
    
    try:
        dom = GridDomain()
        solver_type = load_registered_solver(solver_cpp['entry'])
        solver_args = solver_cpp['config']
        if 'parallel' in inspect.signature(solver_type.__init__).parameters:
            solver_args['parallel'] = parallel
        if 'shared_memory_proxy' in inspect.signature(solver_type.__init__).parameters and shared_memory:
            solver_args['shared_memory_proxy'] = GridShmProxy()
        solver_args['domain_factory'] = lambda: GridDomain()

        with solver_type(**solver_args) as slv:
            GridDomain.solve_with(slv)
            plan, cost = get_plan(dom, slv)
    except Exception as e:
        print(e)
        noexcept = False
    result.send(solver_type.check_domain(dom) and noexcept and \
                ((not solver_cpp['optimal']) or parallel or (cost == 18 and len(plan) == 18)))
    result.close()

def test_solve_cpp(solver_cpp, parallel, shared_memory):
    # We launch each algorithm in a separate process in order to avoid the various
    # algorithms to initialize different versions of the OpenMP library in the same
    # process (since our C++ hub algorithms and other algorithms like PPO2 - via torch -
    # might link against different OpenMP libraries)
    pparent, pchild = mp.Pipe(duplex=False)
    p = mp.Process(target=do_test_cpp, args=(solver_cpp, parallel, shared_memory, pchild,))
    p.start()
    r = pparent.recv()
    p.join()
    p.close()
    pparent.close()
    assert r


def do_test_python(solver_python, result):
    noexcept = True

    try:
        dom = GridDomain()
        solver_type = load_registered_solver(solver_python['entry'])
        solver_args = solver_python['config']
    
        with solver_type(**solver_args) as slv:
            GridDomain.solve_with(slv)
            plan, cost = get_plan(dom, slv)
    except Exception as e:
        print(e)
        noexcept = False
    result.send(solver_type.check_domain(dom) and noexcept and \
                ((not solver_python['optimal']) or (cost == 18 and len(plan) == 18)))
    result.close()

def test_solve_python(solver_python):
    # We launch each algorithm in a separate process in order to avoid the various
    # algorithms to initialize different versions of the OpenMP library in the same
    # process (since our C++ hub algorithms and other algorithms like PPO2 - via torch -
    # might link against different OpenMP libraries)
    pparent, pchild = mp.Pipe(duplex=False)
    p = mp.Process(target=do_test_python, args=(solver_python, pchild,))
    p.start()
    r = pparent.recv()
    p.join()
    p.close()
    pparent.close()
    assert r
