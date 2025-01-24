from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch_geometric as thg
from ray.rllib.utils.torch_utils import (
    convert_to_torch_tensor as convert_to_torch_tensor_original,
)
from ray.rllib.utils.typing import TensorStructType

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    extract_graph_dict_from_batched_graph_dict,
    is_graph_dict,
    is_graph_dict_multiinput,
    is_masked_obs,
)
from skdecide.hub.solver.utils.gnn.torch_utils import graph_instance_to_thg_data


def convert_to_torch_tensor(
    x: Union[
        TensorStructType,
        thg.data.Data,
        gym.spaces.GraphInstance,
        list[gym.spaces.GraphInstance],
    ],
    device: Optional[str] = None,
    pin_memory: bool = False,
) -> Union[TensorStructType, thg.data.Data]:
    """Converts any struct to torch.Tensors.

    Args:
        x: Any (possibly nested) struct, the values in which will be
            converted and returned as a new struct with all leaves converted
            to torch tensors.
        device: The device to create the tensor on.
        pin_memory: If True, will call the `pin_memory()` method on the created tensors.

    Returns:
        Any: A new struct with the same structure as `x`, but with all
        values converted to torch Tensor types. This does not convert possibly
        nested elements that are None because torch has no representation for that.
    """
    if isinstance(x, gym.spaces.GraphInstance):
        return graph_instance_to_thg_data(x, device=device, pin_memory=pin_memory)
    elif isinstance(x, list) and isinstance(x[0], gym.spaces.GraphInstance):
        return thg.data.Batch.from_data_list(
            [
                graph_instance_to_thg_data(graph, device=device, pin_memory=pin_memory)
                for graph in x
            ]
        )
    elif isinstance(x, thg.data.Data):
        return x
    elif is_masked_obs(x):
        return {
            k: convert_to_torch_tensor(v, device=device, pin_memory=pin_memory)
            for k, v in x.items()
        }
    elif is_graph_dict(x):
        return batched_graph_dict_to_thg_data(x, device=device, pin_memory=pin_memory)
    elif is_graph_dict_multiinput(x):
        return {
            k: convert_to_torch_tensor(v, device=device, pin_memory=pin_memory)
            for k, v in x.items()
        }
    else:
        return convert_to_torch_tensor_original(
            x=x, device=device, pin_memory=pin_memory
        )


def graph_dict_to_thg_data(
    graph_dict: dict[str, np.ndarray],
    device: Optional[str] = None,
    pin_memory: bool = False,
):
    # Node features
    flatten_node_features = graph_dict["nodes"].reshape((len(graph_dict["nodes"]), -1))
    x = th.tensor(flatten_node_features).float()
    # Edge features
    if graph_dict["edges"] is None:
        edge_attr = None
    else:
        flatten_edge_features = graph_dict["edges"].reshape(
            (len(graph_dict["edges"]), -1)
        )
        edge_attr = th.tensor(flatten_edge_features).float()
    edge_index = (
        th.tensor(graph_dict["edge_links"], dtype=th.long).t().contiguous().view(2, -1)
    )
    # thg.Data
    data = thg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Pin the tensor's memory (for faster transfer to GPU later).
    if pin_memory and th.cuda.is_available():
        data.pin_memory()

    return data if device is None else data.to(device)


def batched_graph_dict_to_thg_data(
    batched_graph_dict: dict[str, np.ndarray],
    device: Optional[str] = None,
    pin_memory: bool = False,
):
    batch_size = batched_graph_dict["nodes"].shape[0]
    return thg.data.Batch.from_data_list(
        [
            graph_dict_to_thg_data(
                graph_dict=extract_graph_dict_from_batched_graph_dict(
                    batched_graph_dict=batched_graph_dict, index=index
                ),
                device=device,
                pin_memory=pin_memory,
            )
            for index in range(batch_size)
        ]
    )
