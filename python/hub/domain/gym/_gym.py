# TODO: support OpenAI GoalEnv
from __future__ import annotations

import struct
import random
from copy import deepcopy
from typing import Any, Optional
from collections import namedtuple

import gym
import numpy as np

from airlaps import hub, Domain, Space, TransitionValue, TransitionOutcome, ImplicitSpace
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, UnrestrictedActions, \
    Initializable, DeterministicInitialized, Markovian, Memoryless, FullyObservable, Renderable, Rewards, PositiveCosts, Goals

GymSpace = hub.load('GymSpace', folder='hub/space/gym')
ListSpace = hub.load('ListSpace', folder='hub/space/gym')


class D(Domain, SingleAgent, Sequential, UnrestrictedActions, Initializable, Memoryless, FullyObservable, Renderable,
        Rewards):
    pass


# TODO: update with latest Gym Env spec (with seed)
class GymDomain(D):
    """This class wraps an OpenAI Gym environment (gym.env) as an AIRLAPS domain.

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


class D(Domain, SingleAgent, Sequential, DeterministicTransitions, UnrestrictedActions, DeterministicInitialized,
        Markovian, FullyObservable, Renderable, Rewards):  # TODO: replace Rewards by PositiveCosts??
    pass


DeterministicGymDomainExtendedState = namedtuple('DeterministicGymDomainExtendedState', ['state', 'context'])

class DeterministicGymDomain(D):
    """This class wraps a deterministic OpenAI Gym environment (gym.env) as an AIRLAPS domain.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       change_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None) -> None:
        """Initialize DeterministicGymDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        change_state: Function to call to change the state of the gym environment.
                      If None, default behavior is to deepcopy the environment when changing state
        """
        self._gym_env = gym_env
        self._change_state = change_state

    def _get_initial_state_(self) -> D.T_state:
        initial_state = self._gym_env.reset()
        return DeterministicGymDomainExtendedState(state=initial_state, context=[self._gym_env, None, None, None])

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:
        if self._change_state is None:
            env = memory.context[0]
            env = deepcopy(env)
        else:
            env = self._gym_env
            self._change_state(env, memory)
        obs, reward, done, info = env.step(action)
        outcome = TransitionOutcome(state=obs, value=TransitionValue(reward=reward), termination=done, info=info)
        return DeterministicGymDomainExtendedState(state=outcome.state, context=[env, memory.state, action, outcome])

    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                              next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:
        last_memory, last_action, outcome = next_state.context[1:4]
        assert (self._are_same(self._gym_env.observation_space, memory.state, last_memory) and
                self._are_same(self._gym_env.action_space, action, last_action))
        return outcome.value

    def _is_terminal(self, state: D.T_state) -> bool:
        outcome = state.context[3]
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
        usable by a classical planner that requires enumerable applicable action sets

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       change_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = 50) -> None:
        """Initialize WidthPlanningGymDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        change_state: Function to call to change the state of the gym environment.
                      If None, default behavior is to deepcopy the environment when changing state
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        super().__init__(gym_env, change_state)
        self._discretization_factor = discretization_factor
        self._branching_factor = branching_factor
        self._max_depth = max_depth
    
    def _get_initial_state_(self) -> D.T_state:
        initial_state = super()._get_initial_state_()
        initial_state.context.append(0)
        return initial_state

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:
        next_state = super()._get_next_state(memory, action)
        next_state.context.append(memory.context[4] + 1)
        return next_state

    def _get_applicable_actions_from(self, memory: D.T_memory[D.T_state]) -> D.T_agent[Space[D.T_event]]:
        applicable_actions = self._discretize_action_space(self.get_action_space()._gym_space)
        if self._branching_factor is not None and len(applicable_actions.get_elements()) > self._branching_factor:
            return ListSpace(random.sample(applicable_actions.get_elements(), self._branching_factor))
        else:
            return applicable_actions
    
    def _discretize_action_space(self, action_space: gym.spaces.Space) -> D.T_agent[Space[D.T_event]]:
        if isinstance(action_space, gym.spaces.box.Box):
            nb_elements = 1
            for dim in action_space.shape:
                nb_elements *= dim
            return ListSpace([action_space.sample() for i in range(self._discretization_factor * nb_elements)])
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
            generate = lambda d: ([[e] + g for e in _discretize_action_space(action_space.spaces[d]).get_elements() for g in generate(d+1)]
                                  if d < len(action_space.spaces)-1 else
                                  [[e] for e in _discretize_action_space(action_space.spaces[d]).get_elements()])
            return ListSpace(generate(0))
        elif isinstance(action_space, gym.spaces.dict.Dict):
            dkeys = action_space.spaces.keys()
            generate = lambda d: ([[e] + g for e in _discretize_action_space(action_space.spaces[dkeys[d]]).get_elements() for g in generate(d+1)]
                                  if d < len(dkeys)-1 else
                                  [[e] for e in _discretize_action_space(action_space.spaces[dkeys[d]]).get_elements()])
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(action_space)))
    
    def _get_goals_(self):
        return ImplicitSpace(lambda observation: observation.context[4] > self._max_depth)


class GymWidthPlanningDomain(GymPlanningDomain):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by width-based planning algorithm (e.g. IW)

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       change_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       discretization_factor: int = 10,
                       branching_factor: int = None,
                       max_depth: int = 50) -> None:
        """Initialize WidthPlanningGymDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        change_state: Function to call to change the state of the gym environment.
                      If None, default behavior is to deepcopy the environment when changing state
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        super().__init__(gym_env, change_state, discretization_factor, branching_factor, max_depth)
    
    def nb_of_binary_features(self) -> int:
        """Return the size of the bit vector encoding an observation
        """
        return self._binarize_gym_space_element(self._gym_env.observation_space,
                                                self._gym_env.observation_space.sample(),
                                                0,
                                                lambda i: None)

    def binarize(self, memory: D.T_memory[D.T_state], func: Callable[[int], None]) -> None:
        """Transform state in a bit vector and call f on each true value of this vector

        # Parameters
        memory: The Gym state (in observation_space) to binarize
        func: The function called on each true bit of the binarized state
        """
        self._binarize_gym_space_element(self._gym_env.observation_space, memory.state, 0, func)
    
    def _binarize_gym_space_element(self, space: gym.spaces.Space,
                                          element: Any,
                                          start: int,
                                          func: Callable[[int], None]) -> int:
        current_index = start
        if isinstance(space, gym.spaces.box.Box):
            # compute the size of the bit vector encoding the largest float
            maxlen = len(bin(struct.unpack('!i', struct.pack('!f', float('inf')))[0])[2:])
            for cell in np.nditer(element):
                # convert float to string of 1s and 0s
                float_bin = bin(struct.unpack('!i', struct.pack('!f', cell))[0])
                # the sign of the float is encoded in the first bit in our translation
                if float_bin[0] == '-':
                    func(current_index)
                    float_bin = float_bin[3:]
                else:
                    float_bin = float_bin[2:]
                current_index += 1
                # add 0s at the beginning of the string so that all elements are encoded with the same number of bits
                # that depends on the size of the bit vector encoding the largest float
                float_bin = ('0' * (maxlen - len(float_bin))) + float_bin
                for b in float_bin:
                    if b == '1':
                        func(current_index)
                    current_index += 1
        elif isinstance(space, gym.spaces.discrete.Discrete):
            # convert int to string of 1s and 0s
            int_bin = bin(element)[2:]
            # add 0s at the beginning of the string so that all elements are encoded with the same number of bits
            # that depends on the discrete space's highest element which is space.n - 1
            int_bin = ('0' * (len(bin(space.n - 1)[2:]) - len(int_bin))) + int_bin
            for b in int_bin:
                if b == '1':
                    func(current_index)
                current_index += 1
        elif isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
            # look at the previous test case for the logics of translating each element of the vector to a bit string
            for i in range(len(space.nvec)):
                int_bin = bin(element[i])[2:]
                int_bin = ('0' * (len(bin(space.nvec[i] - 1)[2:]) - len(int_bin))) + int_bin
                for b in int_bin:
                    if b == '1':
                        func(current_index)
                    current_index += 1
        elif isinstance(space, gym.spaces.multi_binary.MultiBinary):
            for b in element:
                if b:
                    func(current_index)
                current_index += 1
        elif isinstance(space, gym.spaces.tuple.Tuple):
            for i in range(len(space.spaces)):
                current_index = _binarize_gym_space_element(space.spaces[i], element[i], current_index, func)
        elif isinstance(space, gym.spaces.dict.Dict):
            for k, v in space.spaces:
                current_index = _binarize_gym_space_element(v, element[k], current_index, func)
        else:
            raise RuntimeError('Unknown Gym space element of type ' + str(type(space)))
        return current_index


class AsGymEnv(gym.Env):
    """This class wraps an AIRLAPS domain as an OpenAI Gym environment.

    !!! warning
        The AIRLAPS domain to wrap should inherit #UnrestrictedActionDomain since OpenAI Gym environments usually assume
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
        domain: The AIRLAPS domain to wrap as an OpenAI Gym environment.
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

        # Returns:
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

        # returns
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
        """Unwrap the AIRLAPS domain and return it.

        # Returns
        The original AIRLAPS domain.
        """
        return self._domain
