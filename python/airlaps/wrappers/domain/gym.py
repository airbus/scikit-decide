# TODO: support OpenAI GoalEnv
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import gym

from airlaps import Domain, Space, TransitionValue, TransitionOutcome
from airlaps.builders.domain import SingleAgent, Sequential, DeterministicTransitions, UnrestrictedActions, \
    Initializable, DeterministicInitialized, Markovian, Memoryless, FullyObservable, Renderable, Rewards, PositiveCosts
from airlaps.wrappers.space.gym import GymSpace

__all__ = ['GymDomain', 'DeterministicGymDomain', 'CostDeterministicGymDomain', 'AsGymEnv']


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


class DeterministicGymDomain(D):
    """This class wraps a deterministic OpenAI Gym environment (gym.env) as an AIRLAPS domain.

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env) -> None:
        """Initialize DeterministicGymDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        """
        self._gym_env = gym_env
        self._env_dict = {}

    def _get_initial_state_(self) -> D.T_state:
        initial_state = self._gym_env.reset()
        self._env_dict[self._hash(initial_state)] = (self._gym_env, None, None, None)
        return initial_state

    def _get_next_state(self, memory: D.T_memory[D.T_state],
                        action: D.T_agent[D.T_concurrency[D.T_event]]) -> D.T_state:
        env, _, _, _ = self._env_dict[self._hash(memory)]
        env = deepcopy(env)
        obs, reward, done, info = env.step(action)
        outcome = TransitionOutcome(state=obs, value=TransitionValue(reward=reward), termination=done, info=info)
        self._env_dict[self._hash(outcome.state)] = (env, memory, action, outcome)
        return outcome.state

    def _get_transition_value(self, memory: D.T_memory[D.T_state], action: D.T_agent[D.T_concurrency[D.T_event]],
                              next_state: Optional[D.T_state] = None) -> D.T_agent[TransitionValue[D.T_value]]:
        _, last_memory, last_action, outcome = self._env_dict[self._hash(next_state)]
        assert self._hash(memory) == self._hash(last_memory) and self._hash(action) == self._hash(last_action)
        return outcome.value

    def _is_terminal(self, state: D.T_state) -> bool:
        _, _, _, outcome = self._env_dict[self._hash(state)]
        return outcome.termination

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

    def _hash(self, obj):
        hash_fn = getattr(obj, '__hash__', None)
        if hash_fn is None:
            custom_hash = hash(str(obj))
        else:
            custom_hash = hash_fn()
        return custom_hash

    def close(self):
        return self._gym_env.close()

    def unwrapped(self):
        """Unwrap the deterministic Gym environment (gym.env) and return it.

        # Returns
        The original Gym environment.
        """
        return self._gym_env


class CostDeterministicGymDomain(DeterministicGymDomain, PositiveCosts):
    pass


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

    !!! note
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

        !!! note
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
