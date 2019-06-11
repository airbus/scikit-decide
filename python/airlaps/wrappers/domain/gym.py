# TODO: support OpenAI GoalEnv

from typing import Any

import gym

from airlaps.builders.domain.dynamics import EnvironmentDomain
from airlaps.builders.domain.events import UnrestrictedActionDomain
from airlaps.builders.domain.initialization import InitializableDomain
from airlaps.builders.domain.memory import MemorylessDomain
from airlaps.builders.domain.observability import FullyObservableDomain
from airlaps.builders.domain.renderability import RenderableDomain
from airlaps.builders.domain.value import RewardDomain
from airlaps.core import T_state, T_observation, T_event, T_value, T_info, Space, TransitionValue, TransitionOutcome, \
    Memory
from airlaps.domains import Domain
from airlaps.wrappers.space.gym import GymSpace


class GymDomain(Domain, EnvironmentDomain, UnrestrictedActionDomain, InitializableDomain, MemorylessDomain,
                FullyObservableDomain, RenderableDomain, RewardDomain):
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

    def _reset(self) -> T_state:
        return self._gym_env.reset()

    def _step(self, event: T_event) -> TransitionOutcome[T_state, T_value, T_info]:
        obs, reward, done, info = self._gym_env.step(event)
        return TransitionOutcome(state=obs, value=TransitionValue(reward=reward), termination=done, info=info)

    def _get_action_space_(self) -> Space[T_event]:
        return GymSpace(self._gym_env.action_space)

    def _get_observation_space_(self) -> Space[T_observation]:
        return GymSpace(self._gym_env.observation_space)

    def _render(self, memory: Memory[T_state], **kwargs: Any) -> Any:
        if 'mode' in kwargs:
            return self._gym_env.render(mode=kwargs['mode'])
        else:
            return self._gym_env.render()

    def unwrapped(self):
        """Unwrap the Gym environment (gym.env) and return it.

        # Returns
        The original Gym environment.
        """
        return self._gym_env


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

    def unwrapped(self):
        """Unwrap the AIRLAPS domain and return it.

        # Returns
        The original AIRLAPS domain.
        """
        return self._domain
