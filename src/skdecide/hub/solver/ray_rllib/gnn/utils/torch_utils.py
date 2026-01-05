from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch_geometric as thg
from ray.rllib.utils.torch_utils import (
    convert_to_torch_tensor as convert_to_torch_tensor_original,
)
from ray.rllib.utils.typing import TensorStructType

from skdecide.hub.solver.ray_rllib.action_masking.utils.spaces.space_utils import (
    is_masked_obs,
)
from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    NODES,
    convert_dict_to_graph,
    extract_graph_dict_from_batched_graph_dict,
    is_graph_dict,
    is_graph_dict_multiinput,
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


def batched_graph_dict_to_thg_data(
    batched_graph_dict: dict[str, np.ndarray],
    device: Optional[str] = None,
    pin_memory: bool = False,
):
    batch_size = batched_graph_dict[NODES].shape[0]
    return thg.data.Batch.from_data_list(
        [
            graph_instance_to_thg_data(
                graph=convert_dict_to_graph(
                    extract_graph_dict_from_batched_graph_dict(
                        batched_graph_dict=batched_graph_dict, index=index
                    )
                ),
                device=device,
                pin_memory=pin_memory,
            )
            for index in range(batch_size)
        ]
    )
