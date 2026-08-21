"""A compatibility wrapper converting an old-style environment into a valid environment."""

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    SupportsFloat,
    Tuple,
    Union,
    runtime_checkable,
)

import gymnasium as gym
import numpy as np
from gymnasium import logger
from gymnasium.core import ObsType

DoneStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[SupportsFloat, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]

TerminatedTruncatedStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[SupportsFloat, np.ndarray],
    Union[bool, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]


def convert_to_terminated_truncated_step_api(
    step_returns: Union[DoneStepType, TerminatedTruncatedStepType], is_vector_env=False
) -> TerminatedTruncatedStepType:
    """Function to transform step returns to new step API irrespective of input API.

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 5:
        return step_returns
    else:
        assert len(step_returns) == 4
        observations, rewards, dones, infos = step_returns

        # Cases to handle - info single env /  info vector env (list) / info vector env (dict)
        if is_vector_env is False:
            truncated = infos.pop("TimeLimit.truncated", False)
            return (
                observations,
                rewards,
                dones and not truncated,
                dones and truncated,
                infos,
            )
        elif isinstance(infos, list):
            truncated = np.array(
                [info.pop("TimeLimit.truncated", False) for info in infos]
            )
            return (
                observations,
                rewards,
                np.logical_and(dones, np.logical_not(truncated)),
                np.logical_and(dones, truncated),
                infos,
            )
        elif isinstance(infos, dict):
            num_envs = len(dones)
            truncated = infos.pop("TimeLimit.truncated", np.zeros(num_envs, dtype=bool))
            return (
                observations,
                rewards,
                np.logical_and(dones, np.logical_not(truncated)),
                np.logical_and(dones, truncated),
                infos,
            )
        else:
            raise TypeError(
                f"Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {type(infos)}"
            )


@runtime_checkable
class LegacyEnv(Protocol):
    """A protocol for environments using the old step API."""

    observation_space: gym.Space
    action_space: gym.Space

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        ...

    def render(self, mode: Optional[str] = "human") -> Any:
        """Render the environment."""
        ...

    def close(self):
        """Close the environment."""
        ...

    def seed(self, seed: Optional[int] = None):
        """Set the seed for this env's random number generator(s)."""
        ...


class EnvCompatibility(gym.Env):
    r"""A wrapper which can transform an environment from the old API to the new API.

    Old step API refers to step() method returning (observation, reward, done, info), and reset() only retuning the observation.
    New step API refers to step() method returning (observation, reward, terminated, truncated, info) and reset() returning (observation, info).
    (Refer to docs for details on the API change)

    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(self, old_env: LegacyEnv, render_mode: Optional[str] = None):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            old_env (LegacyEnv): the env to wrap, implemented with the old API
            render_mode (str): the render mode to use when rendering the environment, passed automatically to env.render
        """
        logger.deprecation(
            "The `gymnasium.make(..., apply_api_compatibility=...)` parameter is deprecated and will be removed in v1.0. "
            "Instead use `gymnasium.make('GymV21Environment-v0', env_name=...)` or `from shimmy import GymV21CompatibilityV0`"
        )

        self.env = old_env
        self.metadata = getattr(old_env, "metadata", {"render_modes": []})
        self.render_mode = render_mode
        self.reward_range = getattr(old_env, "reward_range", None)
        self.spec = getattr(old_env, "spec", None)

        self.observation_space = old_env.observation_space
        self.action_space = old_env.action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        if seed is not None:
            self.env.seed(seed)
        # Options are ignored

        if self.render_mode == "human":
            self.render()

        return self.env.reset(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)

        if self.render_mode == "human":
            self.render()

        return convert_to_terminated_truncated_step_api((obs, reward, done, info))

    def render(self) -> Any:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render(mode=self.render_mode)

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)
