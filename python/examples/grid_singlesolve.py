# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import NamedTuple, Optional
from math import sqrt
from multiprocessing import Value, Array
from ctypes import Structure, c_double, c_bool

from skdecide import DeterministicPlanningDomain, TransitionValue, \
                     EnvironmentOutcome, Space, SingleValueDistribution
from skdecide.builders.domain import UnrestrictedActions
from skdecide.hub.space.gym import ListSpace, EnumSpace, MultiDiscreteSpace
from skdecide.hub.solver.riw import RIW
from skdecide.hub.solver.mcts import UCT
from skdecide.utils import rollout


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


class MyShmProxy:

    _register_ = [(State, 2), (Action, 1), (EnumSpace, 1), (SingleValueDistribution, 1),
                  (TransitionValue, 1), (EnvironmentOutcome, 1), (bool, 1)]

    def __init__(self):
        self._proxies_ = {State: MyShmProxy.StateProxy, Action: MyShmProxy.ActionProxy,
                          EnumSpace: MyShmProxy.EnumSpaceProxy,
                          SingleValueDistribution: MyShmProxy.SingleValueDistributionProxy,
                          TransitionValue: MyShmProxy.TransitionValueProxy,
                          EnvironmentOutcome: MyShmProxy.EnvironmentOutcomeProxy,
                          bool: MyShmProxy.BoolProxy}
    
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
            return Array('d', [0, 0], lock=True)
        
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
            return Value('I', 0, lock=True)
        
        def encode(action, shm_action):
            shm_action.value = action.value
        
        @staticmethod
        def decode(shm_action):
            return Action(shm_action.value)
    
    class EnumSpaceProxy:  # Always used with Action as enum class
        @staticmethod
        def initialize():
            return Array('c', b'')
        
        def encode(val, shm_val):
            pass
        
        @staticmethod
        def decode(val):
            return EnumSpace(Action)
    
    class SingleValueDistributionProxy:  # Always used with State
        @staticmethod
        def initialize():
            return MyShmProxy.StateProxy.initialize()
        
        def encode(svd, shm_svd):
            MyShmProxy.StateProxy.encode(svd._value, shm_svd)
        
        @staticmethod
        def decode(svd):
            return SingleValueDistribution(MyShmProxy.StateProxy.decode(svd))
    
    class TransitionValueProxy:

        class MyStructure(Structure):
            _fields_ = [('value', c_double), ('reward', c_bool)]

        @staticmethod
        def initialize():
            return Array(MyShmProxy.TransitionValueProxy.MyStructure, [(0, True)])
        
        def encode(value, shm_value):
            if value.reward is not None:
                shm_value[0].value = value.reward
                shm_value[0].reward = True
            elif value.cost is not None:
                shm_value[0].value = value.cost
                shm_value[0].reward = False
            else:
                shm_value[0].value = 0
                shm_value[0].reward = True
        
        @staticmethod
        def decode(value):
            if value[0].reward:
                return TransitionValue(reward=value[0].value)
            else:
                return TransitionValue(cost=value[0].value)
    
    class EnvironmentOutcomeProxy:
        @staticmethod
        def initialize():
            class MyStructure(Structure):
                _fields_ = []
    
    class BoolProxy:
        @staticmethod
        def initialize():
            return Value('b', False)
        
        def encode(val, shm_val):
            shm_val = val
        
        @staticmethod
        def decode(val):
            return bool(val)


if __name__ == '__main__':

    domain_factory = lambda: MyDomain(10, 10)
    domain = domain_factory()
    domain.reset()

    if RIW.check_domain(domain):
        # solver_factory = lambda: RIW(state_features=lambda d, s: (s.x, s.y),
        #                              use_state_feature_hash=False,
        #                              use_simulation_domain=False,
        #                              time_budget=1000,
        #                              rollout_budget=100,
        #                              max_depth=200,
        #                              exploration=0.25,
        #                              parallel=True,
        #                              debug_logs=False)
        solver_factory = lambda: UCT(time_budget=1000,
                                     rollout_budget=100,
                                     max_depth=200,
                                     transition_mode=UCT.Options.TransitionMode.Distribution,
                                     parallel=True,
                                     shared_memory_proxy=MyShmProxy(),
                                     debug_logs=True)
        solver = MyDomain.solve_with(solver_factory, domain_factory)
        rollout(domain, solver, num_episodes=1, max_steps=2, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
