# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: support OpenAI GoalEnv
from __future__ import annotations

import random
import struct
import bisect
from copy import deepcopy
from typing import Any, Optional, Callable, List
from collections import namedtuple  # TODO: replace with `from typing import NamedTuple`?

import gym
import numpy as np

from skdecide import Domain, Space, TransitionValue, TransitionOutcome, ImplicitSpace
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, UnrestrictedActions, \
    Initializable, DeterministicInitialized, Markovian, Memoryless, FullyObservable, Renderable, Rewards, \
    PositiveCosts, Goals
from skdecide.hub.space.gym import GymSpace, ListSpace


class D(Domain, SingleAgent, Sequential, UnrestrictedActions, Initializable, Memoryless, FullyObservable, Renderable,
        Rewards):
    pass


# TODO: update with latest Gym Env spec (with seed)
class GymDomain(D):
    """This class wraps an OpenAI Gym environment (gym.env) as a scikit-decide domain.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env) -> None:
        """Initialize GymDomain.

        # Parameters
        gym_env: The Gym environment (gym.env) to wrap.
        """
        self._gym_env = gym_env

    def _state_reset(self) -> D.T_state:
        return self._gym_env.reset()

    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        obs, reward, done, info = self._gym_env.step(action)
        return TransitionOutcome(state=obs, value=TransitionValue(reward=reward), termination=done, info=info)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return GymSpace(self._gym_env.action_space)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return GymSpace(self._gym_env.observation_space)

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        if 'mode' in kwargs:
            return self._gym_env.render(mode=kwargs['mode'])
        else:
            return self._gym_env.render()

    def close(self):
        return self._gym_env.close()

    def unwrapped(self):
        """Unwrap the Gym environment (gym.env) and return it.

        # Returns
        The original Gym environment.
        """
        return self._gym_env


class GymDomainStateProxy :
    def __init__(self, state, context=None):
        self._state = state
        self._context = context
    
    def __hash__(self):
        return hash(tuple(self.flatten(self._state)))
    
    def __eq__(self, other):
        return self.flatten(self._state) == self.flatten(other._state)
    
    def __str__(self):
        return self._state.__str__()
    
    def __repr__(self):
        return self._state.__repr__()
    
    def flatten(self, e):
        if isinstance(e, np.ndarray):
            return [e.item(c) for c in range(e.size)]
        elif isinstance(e, tuple):
            return [tt for t in e for tt in self.flatten(t)]
        elif isinstance(e, dict):
            return [tt for k, v in e.items() for tt in self.flatten(v)]
        else:
            return [e]



# class GymDomainActionProxy :
#     def __hash__(self):
#         return str(self).__hash__()
    
#     def __eq__(self, other):
#         return str(self).__eq__(str(other))
    
#     def __str__(self):
#         return self.__str__()


class GymDomainHashable(GymDomain):
    """This class wraps an OpenAI Gym environment (gym.env) as a scikit-decide domain
       using hashable states and actions.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env) -> None:
        """Initialize GymDomain.

        # Parameters
        gym_env: The Gym environment (gym.env) to wrap.
        """
        super().__init__(gym_env)

    def _state_reset(self) -> D.T_state:
        return GymDomainStateProxy(super()._state_reset())

    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        outcome = super()._state_step(action)
        outcome.state = GymDomainStateProxy(outcome.state)
        return outcome


class D(Domain, SingleAgent, Sequential, UnrestrictedActions, DeterministicInitialized,
        Memoryless, FullyObservable, Renderable, Rewards):
    pass


class DeterministicInitializedGymDomain(D):
    """This class wraps an OpenAI Gym environment (gym.env) as a scikit-decide domain
       with a deterministic initial state (i.e. reset the domain to the initial
       state returned by the first reset)

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None) -> None:
        """Initialize GymDomain.

        # Parameters
        gym_env: The Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        """
        self._gym_env = gym_env
        self._set_state = set_state
        self._get_state = get_state
        self._init_env = None
        self._initial_state = None
        self._initial_env_state = None
    
    def set_memory(self, state: D.T_state) -> None:
        self._initial_state = state
        self._memory = self._init_memory(self._initial_state)
        if self._set_state is not None and self._get_state is not None:
            self._initial_env_state = state._context
            self._set_state(self._gym_env, self._initial_env_state)
        else:
            self._init_env = state._context
            self._gym_env = deepcopy(self._init_env)
    
    def _state_reset(self) -> D.T_state:
        if self._initial_state is None:
            self._initial_state = GymDomainStateProxy(state=self._gym_env.reset(), context=None)
            if self._set_state is not None and self._get_state is not None:
                self._initial_env_state = self._get_state(self._gym_env)
                self._initial_state._context = self._initial_env_state
            else:
                self._init_env = deepcopy(self._gym_env)
                self._initial_state._context = self._init_env
        else:
            if self._set_state is not None and self._get_state is not None:
                self._set_state(self._gym_env, self._initial_env_state)
            else:
                self._gym_env = deepcopy(self._init_env)
        return self._initial_state
    
    def _state_step(self, action: D.T_agent[D.T_concurrency[D.T_event]]) -> TransitionOutcome[
            D.T_state, D.T_agent[TransitionValue[D.T_value]], D.T_agent[D.T_info]]:
        obs, reward, done, info = self._gym_env.step(action)
        if self._set_state is not None and self._get_state is not None:
            state = GymDomainStateProxy(state=obs, context=self._initial_env_state)
        else:
            state = GymDomainStateProxy(state=obs, context=self._init_env)
        return TransitionOutcome(state=state, value=TransitionValue(reward=reward), termination=done, info=info)
    
    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return GymSpace(self._gym_env.action_space)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return GymSpace(self._gym_env.observation_space)

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        if 'mode' in kwargs:
            render =  self._gym_env.render(mode=kwargs['mode'])
        else:
            render =  self._gym_env.render()
        if self._set_state is None or self._get_state is None:
            self._gym_env.close()  # avoid deepcopy errors
        return render

    def close(self):
        return self._gym_env.close()

    def unwrapped(self):
        """Unwrap the Gym environment (gym.env) and return it.

        # Returns
        The original Gym environment.
        """
        return self._gym_env


class GymWidthDomain:
    """This class wraps an OpenAI Gym environment as a domain
        usable by width-based solving algorithm (e.g. IW)

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, continuous_feature_fidelity: int = 1) -> None:
        """Initialize GymWidthDomain.

        # Parameters
        continuous_feature_fidelity: Number of integers to represent a continuous feature
                                     in the interval-based feature abstraction (higher is more precise)
        """
        self._continuous_feature_fidelity = continuous_feature_fidelity
        self._feature_increments = []
        self._init_continuous_state_variables = []
        self._elliptical_features = None
    
    def _reset_features(self):
        self._init_continuous_state_variables = []
        self._feature_increments = []
    
    class BEE1Node:
        def __init__(self, ref):
            self.reference = ref
            self.increments = [0]
            self.children = []

    def _init_bee1_features(self, space, state):
        if isinstance(space, gym.spaces.box.Box):
            for cell_id in range(state.size):
                cell = state.item(cell_id)
                self._init_continuous_state_variables.append(cell)
                # positive increments list for each fidelity level
                self._feature_increments.append(GymWidthDomain.BEE1Node(cell))
                # negative increments list for each fidelity level
                self._feature_increments.append(GymWidthDomain.BEE1Node(cell))
        elif isinstance(space, gym.spaces.tuple.Tuple):
            for s in range(len(space.spaces)):
                self._init_bee1_features(space.spaces[s], state[s])
        elif isinstance(space, gym.spaces.dict.Dict):
            for k, s in space.spaces:
                self._init_bee1_features(s, state[k])
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))

    def bee1_features(self, state):
        """Return a numpy vector of ints representing the current 'cumulated layer' of each state variable
        """
        state = state._state if isinstance(state, GymDomainStateProxy) else state
        if len(self._feature_increments) == 0:
            self._init_bee1_features(self._gym_env.observation_space, state)
        return self._bee1_features(self._gym_env.observation_space, state, 0)[1]
    
    def _bee1_features(self, space, element, start):
        if isinstance(space, gym.spaces.box.Box):
            features = []
            index = start
            for cell_id in range(element.size):
                cell = element.item(cell_id)
                cf = []
                if cell > self._init_continuous_state_variables[index]:
                    node = self._feature_increments[2*index]
                    sign = 1
                else:
                    node = self._feature_increments[2*index+1]
                    sign = -1
                for f in range(self._continuous_feature_fidelity):
                    i = bisect.bisect_left(node.increments, sign * (cell - node.reference))
                    cf.append(sign * i)
                    if i >= len(node.increments):
                        node.increments.append(sign * (cell - node.reference))
                        node.children.append(GymWidthDomain.BEE1Node(node.reference + (sign * node.increments[i-1])))
                        for ff in range(f+1, self._continuous_feature_fidelity):
                            cf.append(0)
                        break
                    elif i > 0:
                        node = node.children[i-1]
                features += cf
                index += 1
            return index, features
        elif isinstance(space, gym.spaces.discrete.Discrete):
            return start, [element]
        elif isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            return start, [e for e in element]
        elif isinstance(space, gym.spaces.multi_binary.MultiBinary):
            return start, [e for e in element]
        elif isinstance(space, gym.spaces.tuple.Tuple):
            index = start
            features = []
            for i in range(len(space.spaces)):
                index, l = self._bee1_features(space.spaces[i], element[i], index)
                features += l
            return index, features
        elif isinstance(space, gym.spaces.dict.Dict):
            index = start
            features = []
            for k in space.spaces.keys():
                index, l = self._bee1_features(space.spaces[k], element[k], index)
                features += l
            return index, features
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))
    
    class BEE2Node:
        def __init__(self):
            self.I = []
            self.llb = None
            self.gub = None
            self.children = []

        def eval(self, x, level=0):
            if len(self.I) == 0:
                self.I = [(x, x)]
                self.children = [GymWidthDomain.BEE2Node()]
                self.llb = x
                self.gub = x
                return tuple([0] + [-1 for k in range(level)])
            if x < self.llb:
                self.I += [(x, self.llb)]
                self.children += [GymWidthDomain.BEE2Node()]
                self.llb = x
                return tuple([len(self.I) - 1] + [-1 for k in range(level)])
            elif x > self.gub:
                self.I += [(self.gub, x)]
                self.children += [GymWidthDomain.BEE2Node()]
                self.gub = x
                return tuple([len(self.I) - 1] + [-1 for k in range(level)])
            else:
                # we need to search
                for k, i_k in enumerate(self.I):
                    if i_k[0] <= x <= i_k[1]:
                        sub = []
                        if level > 0:
                            sub = self.children[k].eval(x, level - 1)
                        return tuple([k] + list(sub))
                        #return k
            raise RuntimeError("Should never get here!")
        
        def __repr__(self):
            return '[I={}, llb={}, gub={}]'.format(self.I, self.llb, self.gub)
    
    def _init_bee2_features(self, space, state):
        if isinstance(space, gym.spaces.box.Box):
            for cell_id in range(state.size):
                cell = state.item(cell_id)
                self._init_continuous_state_variables.append(cell)
                self._feature_increments.append(GymWidthDomain.BEE2Node())
                self._feature_increments[-1].eval(cell)
        elif isinstance(space, gym.spaces.tuple.Tuple):
            for s in range(len(space.spaces)):
                self._init_bee2_features(space.spaces[s], state[s])
        elif isinstance(space, gym.spaces.dict.Dict):
            for k, s in space.spaces:
                self._init_bee2_features(s, state[k])
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))

    def bee2_features(self, state):
        """Return a numpy vector of ints representing the current 'cumulated layer' of each state variable
        """
        state = state._state if isinstance(state, GymDomainStateProxy) else state
        if len(self._feature_increments) == 0:
            self._init_bee2_features(self._gym_env.observation_space, state)
        return self._bee2_features(self._gym_env.observation_space, state, 0)[1]
    
    def _bee2_features(self, space, element, start):
        if isinstance(space, gym.spaces.box.Box):
            features = []
            index = start
            for cell_id in range(element.size):
                cell = element.item(cell_id)
                features += self._feature_increments[index].eval(cell, self._continuous_feature_fidelity - 1)
                index += 1
            return index, features
        elif isinstance(space, gym.spaces.discrete.Discrete):
            return start, [element]
        elif isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            return start, [e for e in element]
        elif isinstance(space, gym.spaces.multi_binary.MultiBinary):
            return start, [e for e in element]
        elif isinstance(space, gym.spaces.tuple.Tuple):
            index = start
            features = []
            for i in range(len(space.spaces)):
                index, l = self._bee2_features(space.spaces[i], element[i], index)
                features += l
            return index, features
        elif isinstance(space, gym.spaces.dict.Dict):
            index = start
            features = []
            for k in space.spaces.keys():
                index, l = self._bee2_features(space.spaces[k], element[k], index)
                features += l
            return index, features
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))
    
    def nb_of_binary_features(self) -> int:
        """Return the size of the bit vector encoding an observation
        """
        return self._binary_features(self._gym_env.observation_space,
                                                self._gym_env.observation_space.sample(),
                                                0,
                                                lambda i: None)

    def binary_features(self, memory: D.T_memory[D.T_state]):
        """Transform state in a bit vector and call f on each true value of this vector

        # Parameters
        memory: The Gym state (in observation_space) to binarize

        Return a list of booleans representing the binary representation of each state variable
        """
        memory = memory._state if isinstance(memory, GymDomainStateProxy) else memory
        return self._binary_features(self._gym_env.observation_space, memory)
    
    def _binary_features(self, space: gym.spaces.Space,
                                          element: Any) -> int:
        if isinstance(space, gym.spaces.box.Box):
            features = []
            # compute the size of the bit vector encoding the largest float
            maxlen = len(bin(struct.unpack('!i', struct.pack('!f', float('inf')))[0])[2:])
            for cell in np.nditer(element):
                # convert float to string of 1s and 0s
                float_bin = bin(struct.unpack('!i', struct.pack('!f', cell))[0])
                # the sign of the float is encoded in the first bit in our translation
                if float_bin[0] == '-':
                    features.append(True)
                    float_bin = float_bin[3:]
                else:
                    features.append(False)
                    float_bin = float_bin[2:]
                # add 0s at the beginning of the string so that all elements are encoded with the same number of bits
                # that depends on the size of the bit vector encoding the largest float
                float_bin = ('0' * (maxlen - len(float_bin))) + float_bin
                for b in float_bin:
                    features.append(b == '1')
            return features
        elif isinstance(space, gym.spaces.discrete.Discrete):
            features = []
            # convert int to string of 1s and 0s
            int_bin = bin(element)[2:]
            # add 0s at the beginning of the string so that all elements are encoded with the same number of bits
            # that depends on the discrete space's highest element which is space.n - 1
            int_bin = ('0' * (len(bin(space.n - 1)[2:]) - len(int_bin))) + int_bin
            for b in int_bin:
                features.append(b == '1')
            return features
        elif isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            features = []
            # look at the previous test case for the logics of translating each element of the vector to a bit string
            for i in range(len(space.nvec)):
                int_bin = bin(element[i])[2:]
                int_bin = ('0' * (len(bin(space.nvec[i] - 1)[2:]) - len(int_bin))) + int_bin
                for b in int_bin:
                    features.append(b == '1')
            return features
        elif isinstance(space, gym.spaces.multi_binary.MultiBinary):
            features = []
            for b in element:
                features.append(bool(b))
            return features
        elif isinstance(space, gym.spaces.tuple.Tuple):
            features = []
            for i in range(len(space.spaces)):
                l = self._binary_features(space.spaces[i], element[i])
                features += l
            return features
        elif isinstance(space, gym.spaces.dict.Dict):
            features = []
            for k, v in space.spaces:
                l = self._binary_features(v, element[k])
                features += l
            return features
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))
    
    class EllipticalMapping2D:

        def __init__(self, input, _x0, _xG, bands):
            self.input = input
            self.x0 = _x0
            self.xG = _xG
            self.projected_goal = np.array([self.xG[self.input[0]], self.xG[self.input[1]]])
            self.bands = bands
        
        def evaluate(self, state):
            projected_state = np.array([state[self.input[0]], state[self.input[1]]])
            c = np.linalg.norm(projected_state - self.projected_goal)
            for k, v in enumerate(self.bands):
                if c > v:
                    return len(self.bands) - k
            return 0
    
    def init_elliptical_features(self, x0: D.T_memory[D.T_state],
                                       xG: D.T_memory[D.T_state]):
        v0 = np.array(self._get_variables(self._gym_env.observation_space, x0))
        vG = np.array(self._get_variables(self._gym_env.observation_space, xG))
        d = xG.shape[0] # column vector
        self._elliptical_features = []
        for i in range(d):
            for j in range(i+1, d):
                input = (i, j)
                c = np.linalg.norm(np.array([v0[i], v0[j]]) - np.array([vG[i], vG[j]]))
                bands = []
                num_levels = 10.0 * max(np.log10(c), 1.0)
                delta_c = c / float(num_levels)
                c_k = c
                while c_k > delta_c:
                    bands += [c_k]
                    c_k -= delta_c
                if len(bands) == 0:
                    bands += [c_k]
                self._elliptical_features += [self.EllipticalMapping2D(input, v0, vG, bands)]
    
    def elliptical_features(self, state: D.T_memory[D.T_state]):
        state = state._state if isinstance(state, GymDomainStateProxy) else state
        vS = np.array(self._get_variables(self._gym_env.observation_space, state))
        return [f.evaluate(vS) for f in self._elliptical_features]
    
    def _get_variables(self, space: gym.spaces.Space, element: Any) -> List:
        if isinstance(space, gym.spaces.box.Box):
            var = []
            for cell in np.nditer(element):
                var.append(cell)
            return var
        elif isinstance(space, gym.spaces.discrete.Discrete):
            return [element]
        elif isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            return [e for e in element]
        elif isinstance(space, gym.spaces.multi_binary.MultiBinary):
            return [e for e in element]
        elif isinstance(space, gym.spaces.tuple.Tuple):
            var = []
            for i in range(len(space.spaces)):
                l = self._get_variables(space.spaces[i], element[i])
                var += l
            return var
        elif isinstance(space, gym.spaces.dict.Dict):
            var = []
            for k in space.spaces.keys():
                index, l = self._get_variables(space.spaces[k], element[k])
                var += l
            return var
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))
        

class GymDiscreteActionDomain(UnrestrictedActions):
    """This class wraps an OpenAI Gym environment as a domain
        usable by a solver that requires enumerable applicable action sets

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, discretization_factor: int = 10,
                       branching_factor: int = None) -> None:
        """Initialize GymDiscreteActionDomain.

        # Parameters
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        """
        self._discretization_factor = discretization_factor
        self._branching_factor = branching_factor
        self._applicable_actions = self._discretize_action_space(self.get_action_space()._gym_space)
        if self._branching_factor is not None and len(self._applicable_actions.get_elements()) > self._branching_factor:
            self._applicable_actions = ListSpace(random.sample(self._applicable_actions.get_elements(), self._branching_factor))

    def _get_applicable_actions_from(self, memory: D.T_memory[D.T_state]) -> D.T_agent[Space[D.T_event]]:
        return self._applicable_actions
    
    def _discretize_action_space(self, action_space: gym.spaces.Space) -> D.T_agent[Space[D.T_event]]:
        if isinstance(action_space, gym.spaces.box.Box):
            nb_elements = 1
            for dim in action_space.shape:
                nb_elements *= dim
            actions = []
            for l, h in np.nditer([action_space.low, action_space.high]):
                if l == -float('inf') or h == float('inf'):
                    actions.append([gym.spaces.box.Box(low=l, high=h).sample() for i in range(self._discretization_factor)])
                else:
                    actions.append([l + ((h-l)/(self._discretization_factor - 1))*i for i in range(self._discretization_factor)])
            alist = []
            self._generate_box_action_combinations(actions, action_space.shape, action_space.dtype, 0, [], alist)
            return ListSpace(alist)
        elif isinstance(action_space, gym.spaces.discrete.Discrete):
            return ListSpace([i for i in range(action_space.n)])
        elif isinstance(action_space, gym.spaces.multi_discrete.MultiDiscrete):
            generate = lambda d: ([[e] + g for e in range(action_space.nvec[d]) for g in generate(d+1)]
                                  if d < len(action_space.nvec)-1 else
                                  [[e] for e in range(action_space.nvec[d])])
            return ListSpace(generate(0))
        elif isinstance(action_space, gym.spaces.multi_binary.MultiBinary):
            generate = lambda d: ([[e] + g for e in [True, False] for g in generate(d+1)]
                                  if d < len(action_space.n)-1 else
                                  [[e] for e in [True, False]])
            return ListSpace(generate(0))
        elif isinstance(action_space, gym.spaces.tuple.Tuple):
            generate = lambda d: ([[e] + g for e in self._discretize_action_space(action_space.spaces[d]).get_elements() for g in generate(d+1)]
                                  if d < len(action_space.spaces)-1 else
                                  [[e] for e in self._discretize_action_space(action_space.spaces[d]).get_elements()])
            return ListSpace(generate(0))
        elif isinstance(action_space, gym.spaces.dict.Dict):
            dkeys = action_space.spaces.keys()
            generate = lambda d: ([[e] + g for e in self._discretize_action_space(action_space.spaces[dkeys[d]]).get_elements() for g in generate(d+1)]
                                  if d < len(dkeys)-1 else
                                  [[e] for e in self._discretize_action_space(action_space.spaces[dkeys[d]]).get_elements()])
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(action_space)))
    
    def _generate_box_action_combinations(self, actions, shape, dtype, index, alist, rlist):
        if index < len(actions):
            for a in actions[index]:
                clist = list(alist)
                clist.append(a)
                self._generate_box_action_combinations(actions, shape, dtype, index+1, clist, rlist)
        else:
            ar = np.ndarray(shape=shape, dtype=dtype)
            if len(shape) == 1:
                ar[0] = alist[0]
            else:
                k = 0
                for i, in np.nditer(ar, op_flags=['readwrite']):
                    print(str(i))
                    i = alist[k]
                    k += 1
            rlist += [ar]


class D(Domain, SingleAgent, Sequential, DeterministicTransitions, UnrestrictedActions, DeterministicInitialized,
        Markovian, FullyObservable, Renderable, Rewards):  # TODO: replace Rewards by PositiveCosts??
    pass


class DeterministicGymDomain(D):
    """This class wraps a deterministic OpenAI Gym environment (gym.env) as a scikit-decide domain.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None) -> None:
        """Initialize DeterministicGymDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        """
        self._gym_env = gym_env
        self._set_state = set_state
        self._get_state = get_state
        self._init_env = None

    def _get_initial_state_(self) -> D.T_state:
        initial_state = self._gym_env.reset()
        return GymDomainStateProxy(state=initial_state,
                                                context=[self._gym_env, None, None, None,
                                                         self._get_state(self._gym_env) if (self._get_state is not None and self._set_state is not None) else None])

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:
        env = memory._context[0]
        if self._set_state is None or self._get_state is None:
            env = deepcopy(env)
        elif memory._context[4] != self._get_state(env):
            self._set_state(env, memory._context[4])
        self._gym_env = env  # Just in case the simulation environment would be different from the planner's environment...
        obs, reward, done, info = env.step(action)
        outcome = TransitionOutcome(state=obs, value=TransitionValue(reward=reward), termination=done, info=info)
        # print('Transition:', str(memory._state), ' -> ', str(action), ' -> ', str(outcome.state))
        return GymDomainStateProxy(state=outcome.state,
                                                context=[env, memory._state, action, outcome,
                                                         self._get_state(env) if (self._get_state is not None and self._set_state is not None) else None])

    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                              next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:
        last_memory, last_action, outcome = next_state._context[1:4]
        # assert (self._are_same(self._gym_env.observation_space, memory._state, last_memory) and
        #         self._are_same(self._gym_env.action_space, action, last_action))
        return outcome.value

    def _is_terminal(self, state: D.T_state) -> bool:
        outcome = state._context[3]
        return outcome.termination if outcome is not None else False
    
    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return GymSpace(self._gym_env.action_space)

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return GymSpace(self._gym_env.observation_space)

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        if 'mode' in kwargs:
            render =  self._gym_env.render(mode=kwargs['mode'])
        else:
            render =  self._gym_env.render()
        if self._set_state is None or self._get_state is None:
            self._gym_env.close()  # avoid deepcopy errors
        return render
    
    def close(self):
        return self._gym_env.close()

    def unwrapped(self):
        """Unwrap the deterministic Gym environment (gym.env) and return it.

        # Returns
        The original Gym environment.
        """
        return self._gym_env

    def _are_same(self, space: gym.spaces.Space, e1: Any, e2: Any):
        assert e1.__class__ == e2.__class__
        if isinstance(space, gym.spaces.box.Box):
            return (e1 == e2).all()
        elif isinstance(space, gym.spaces.discrete.Discrete):
            return e1 == e2
        elif isinstance(space, gym.spaces.multi_binary.MultiBinary):
            return (e1 == e2).all()
        elif isinstance(space, gym.spaces.tuple.Tuple):
            assert len(e1) == len(e2) == len(space)
            for i in range(len(space)):
                if not self._are_same(space[i], e1[i], e2[i]):
                    return False
            return True
        elif isinstance(space, gym.spaces.dict.Dict):
            assert e1.keys() == e2.keys() == space.keys()
            for k in space.keys():
                if not self._are_same(space[k], e1[k], e2[k]):
                    return False
            return True
        else:
            raise RuntimeError('Unknown Gym space of type ' + str(type(space)))


class CostDeterministicGymDomain(DeterministicGymDomain, PositiveCosts):
    pass


class GymPlanningDomain(CostDeterministicGymDomain, Goals):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a classical planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       termination_is_goal: bool = False,
                       max_depth: int = 50) -> None:
        """Initialize GymPlanningDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        termination_is_goal: True if the termination condition is a goal (and not a dead-end)
        max_depth: maximum depth of states to explore from the initial state
        """
        super().__init__(gym_env, set_state, get_state)
        self._termination_is_goal = termination_is_goal
        self._max_depth = max_depth
        self._initial_state = None
        self._current_depth = 0
        self._restarting_from_initial_state = False

    def _get_initial_state_(self) -> D.T_state:
        initial_state = super()._get_initial_state_()
        initial_state._context.append(0)  # Depth
        initial_state._context.append(0)  # Accumulated reward
        self._initial_state = initial_state
        self._current_depth = 0
        return initial_state

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:
        if self._initial_state is None:  # the solver's domain does not use _get_initial_state_() but gets its initial state from the rollout env shipped with the state context
            self._initial_state = memory
        if self._are_same(self._gym_env.observation_space, memory._state, self._initial_state._state):
            self._restart_from_initial_state()
        else:
            self._restarting_from_initial_state = False
        next_state = super()._get_next_state(memory, action)
        next_state._context.append(memory._context[5] + 1)
        next_state._context.append(memory._context[6] + memory._context[3].value.reward if memory._context[3] is not None else memory._context[6])
        if (memory._context[5] + 1 > self._current_depth):
            self._current_depth = memory._context[5] + 1
            print('Current depth:', str(self._current_depth), '/', str(self._max_depth))
        return next_state
    
    def _restart_from_initial_state(self):
        # following test is mandatory since we restart from the initial state
        # when expanding each action of the initial state
        # and we want to do it only once
        if not self._restarting_from_initial_state:
            self._current_depth = 0
            if isinstance(self, GymWidthDomain):
                self._reset_features()
            self._restarting_from_initial_state = True

    def _get_goals_(self):
        return ImplicitSpace(lambda observation: ((observation._context[5] >= self._max_depth) or
                                                  (self._termination_is_goal and (observation._context[3].termination
                                                                                  if observation._context[3] is not None else False))))


class AsGymEnv(gym.Env):
    """This class wraps a scikit-decide domain as an OpenAI Gym environment.

    !!! warning
        The scikit-decide domain to wrap should inherit #UnrestrictedActionDomain since OpenAI Gym environments usually assume
        that all their actions are always applicable.

    An OpenAI Gym environment encapsulates an environment with arbitrary behind-the-scenes dynamics. An environment can
    be partially or fully observed.

    The main API methods that users of this class need to know are:

    - step
    - reset
    - render
    - close
    - seed

    And set the following attributes:

    - action_space: The Space object corresponding to valid actions.
    - observation_space: The Space object corresponding to valid observations.
    - reward_range: A tuple corresponding to the min and max possible rewards.

    !!! tip
        A default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range. The methods are
        accessed publicly as "step", "reset", etc.. The non-underscored versions are wrapper methods to which
        functionality may be added over time.
    """

    def __init__(self, domain: Domain, unwrap_spaces: bool = True) -> None:
        """Initialize AsGymEnv.

        # Parameters
        domain: The scikit-decide domain to wrap as an OpenAI Gym environment.
        unwrap_spaces: Boolean specifying whether the action & observation spaces should be unwrapped.
        """
        self._domain = domain
        self._unwrap_spaces = unwrap_spaces
        if unwrap_spaces:
            self.observation_space = domain.get_observation_space().unwrapped()
            self.action_space = domain.get_action_space().unwrapped()  # assumes all actions are always applicable
        else:
            self.observation_space = domain.get_observation_space()
            self.action_space = domain.get_action_space()  # assumes all actions are always applicable

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for
        calling `reset()` to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        # Parameters
        action (object): An action provided by the environment.

        # Returns
        A tuple with following elements:

        - observation (object): The agent's observation of the current environment.
        - reward (float) : The amount of reward returned after previous action.
        - done (boolean): Whether the episode ended, in which case further step() calls will return undefined results.
        - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        action = next(iter(self._domain.get_action_space().from_unwrapped([action])))
        outcome = self._domain.step(action)
        outcome_observation = next(iter(self._domain.get_observation_space().to_unwrapped([outcome.observation])))
        # Some solvers dealing with OpenAI Gym environments crash when info is None (e.g. baselines solver)
        outcome_info = outcome.info if outcome.info is not None else {}
        return outcome_observation, outcome.value.reward, outcome.termination, outcome_info

    def reset(self):
        """Reset the state of the environment and returns an initial observation.

        # Returns
        observation (object): The initial observation of the space.
        """
        return self._domain.reset()

    def render(self, mode='human'):
        """Render the environment.

        The set of supported modes varies per environment. (And some environments do not support rendering at all.) By
        convention, if mode is:

        - human: Render to the current display or terminal and return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
        suitable for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation. The text can
        include newlines and ANSI escape sequences (e.g. for colors).

        !!! tip
            Make sure that your class's metadata 'render.modes' key includes he list of supported modes. It's
            recommended to call super() in implementations to use the functionality of this method.

        # Parameters
        mode (str): The mode to render with.
        close (bool): Close all open renderings.

        # Example
        ```python
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        ```
        """
        return self._domain.render(mode=mode)

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when garbage collected or when the program exits.
        """
        return self._domain.close()

    def unwrapped(self):
        """Unwrap the scikit-decide domain and return it.

        # Returns
        The original scikit-decide domain.
        """
        return self._domain
