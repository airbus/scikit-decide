from typing import Union

import numpy as np
import torch as th
import torch_geometric as thg
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim as sb3_get_action_dim
from stable_baselines3.common.preprocessing import get_obs_shape as sb3_get_obs_shape
from stable_baselines3.common.preprocessing import preprocess_obs as sb3_preprocess_obs

from .utils import TorchObsType


def preprocess_obs(
    obs: TorchObsType,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> TorchObsType:
    """Preprocess observation to be fed to a neural network.

    Wraps original sb3 preprocess_obs to catch graph obs.

    """
    if isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(
                _obs, observation_space[key], normalize_images=normalize_images
            )
        return preprocessed_obs  # type: ignore[return-value]

    assert isinstance(
        obs, (th.Tensor, thg.data.Data)
    ), f"Expecting a torch Tensor or torch geometric Data, but got {type(obs)}"

    if isinstance(observation_space, spaces.Graph):
        return obs
    else:
        return sb3_preprocess_obs(
            obs, observation_space=observation_space, normalize_images=normalize_images
        )


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[tuple[int, ...], dict[str, tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Graph):
        # Will not be used
        return observation_space.node_space.shape + observation_space.edge_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    else:
        return sb3_get_obs_shape(observation_space=observation_space)


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Graph):
        return int(np.prod(action_space.node_space.shape))
    else:
        return sb3_get_action_dim(action_space=action_space)
