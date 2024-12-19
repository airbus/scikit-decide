from typing import Union

import gymnasium as gym
import numpy as np
import stable_baselines3.common.utils
import torch as th
import torch_geometric as thg

SubObsType = Union[np.ndarray, gym.spaces.GraphInstance, list[gym.spaces.GraphInstance]]
ObsType = Union[SubObsType, dict[str, SubObsType]]
TorchSubObsType = Union[th.Tensor, thg.data.Data]
TorchObsType = Union[TorchSubObsType, dict[str, TorchSubObsType]]


def copy_graph_instance(g: gym.spaces.GraphInstance) -> gym.spaces.GraphInstance:
    return gym.spaces.GraphInstance(
        nodes=np.copy(g.nodes), edges=np.copy(g.edges), edge_links=np.copy(g.edge_links)
    )


def copy_np_array_or_list_of_graph_instances(
    obs: Union[np.ndarray, list[gym.spaces.GraphInstance]]
) -> Union[np.ndarray, list[gym.spaces.GraphInstance]]:
    if isinstance(obs[0], gym.spaces.GraphInstance):
        return [copy_graph_instance(g) for g in obs]
    else:
        return np.copy(obs)


def graph_obs_to_thg_data(
    obs: gym.spaces.GraphInstance, device: th.device
) -> thg.data.Data:
    # Node features
    flatten_node_features = obs.nodes.reshape((len(obs.nodes), -1))
    x = th.tensor(flatten_node_features).float()
    # Edge features
    if obs.edges is None:
        edge_attr = None
    else:
        flatten_edge_features = obs.edges.reshape((len(obs.edges), -1))
        edge_attr = th.tensor(flatten_edge_features).float()
    edge_index = th.tensor(obs.edge_links, dtype=th.long).t().contiguous().view(2, -1)
    return thg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


def obs_as_tensor(
    obs: ObsType,
    device: th.device,
) -> TorchObsType:
    """
    Moves the observation to the given device.

    Args:
        obs:
        device: PyTorch device

    Returns:
        PyTorch tensor of the observation on a desired device.

    """
    if isinstance(obs, gym.spaces.GraphInstance):
        return graph_obs_to_thg_data(obs, device=device)
    elif isinstance(obs, list) and isinstance(obs[0], gym.spaces.GraphInstance):
        if len(obs) > 1:
            raise NotImplementedError(
                "Not implemented for real vectorized environment "
                "(ie. with more than 1 wrapped environment)"
            )
        return graph_obs_to_thg_data(obs[0], device=device)
    elif isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: obs_as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def is_vectorized_observation(
    observation: SubObsType, observation_space: gym.spaces.Space
) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if isinstance(observation_space, gym.spaces.Graph):
        return isinstance(observation_space, list)
    else:
        return stable_baselines3.common.utils.is_vectorized_observation(
            observation=observation, observation_space=observation_space
        )