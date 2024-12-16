import torch as th
import torch_geometric as thg
from gymnasium import spaces
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
