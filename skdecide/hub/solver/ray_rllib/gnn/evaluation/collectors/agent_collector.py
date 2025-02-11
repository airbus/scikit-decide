from typing import Any

import numpy as np
import tree
from ray.rllib import SampleBatch
from ray.rllib.evaluation.collectors.agent_collector import (
    AgentCollector,
    _to_float_np_array,
)

from skdecide.hub.solver.ray_rllib.gnn.utils.spaces.space_utils import (
    EDGE_LINKS,
    EDGES,
    NODES,
    TRUE_OBS,
    is_graph_dict,
    is_graph_dict_multiinput,
    is_masked_obs,
    pad_axis,
    pad_graph,
)


def agent_collector_graph_cache_in_np(
    self: AgentCollector, cache_dict: dict[str, list[np.ndarray]], key: str
) -> None:
    """Update of AgentCollector._cache_in_np to handle buffers containing graphs of different shapes."""
    if key not in cache_dict:
        if key in (SampleBatch.OBS, SampleBatch.NEXT_OBS):
            buffers_entry = [sublist for sublist in self.buffers[key]]  # shallow copy
            buffers_entry_struct = self.buffer_structs[key]
            buffers_entry_indices_struct = tree.unflatten_as(
                buffers_entry_struct, range(len(buffers_entry))
            )
            pad_graph_buffers(buffers_entry, buffers_entry_indices_struct)
        else:
            buffers_entry = self.buffers[key]
        cache_dict[key] = [_to_float_np_array(d) for d in buffers_entry]


def agent_collector_graph2node_cache_in_np(
    self: AgentCollector, cache_dict: dict[str, list[np.ndarray]], key: str
) -> None:
    """Update of AgentCollector._cache_in_np to handle buffers containing graphs/actions logits of different shapes."""
    if key not in cache_dict:
        if key == SampleBatch.ACTION_DIST_INPUTS:
            assert len(self.buffers[key]) == 1  # singleton list
            max_nodes = max(len(logits) for logits in self.buffers[key][0])
            minus_infty_approx = np.finfo(self.buffers[key][0][0].dtype).min
            buffers_entry = [
                [
                    pad_axis(logits, max_dim=max_nodes, value=minus_infty_approx)
                    for logits in self.buffers[key][0]
                ]
            ]
            cache_dict[key] = [_to_float_np_array(d) for d in buffers_entry]
        else:
            agent_collector_graph_cache_in_np(self, cache_dict=cache_dict, key=key)


def pad_graph_buffers(
    buffers_entry: list[list[np.ndarray]], buffers_entry_indices_struct: Any
) -> None:
    if is_graph_dict(buffers_entry_indices_struct):
        if all(
            isinstance(a, np.ndarray)
            for idx in buffers_entry_indices_struct.values()
            for a in buffers_entry[idx]
        ) and any(
            len(set(a.shape for a in buffers_entry[idx])) > 1
            for idx in buffers_entry_indices_struct.values()
        ):
            list_edge_links = buffers_entry[buffers_entry_indices_struct[EDGE_LINKS]]
            list_edges = buffers_entry[buffers_entry_indices_struct[EDGES]]
            list_nodes = buffers_entry[buffers_entry_indices_struct[NODES]]

            new_list_nodes, new_list_edges, new_list_edge_links = [], [], []
            max_n_nodes = max(len(nodes) for nodes in list_nodes)
            max_n_edges = max(len(edges) for edges in list_edges)
            for nodes, edges, edge_links in zip(
                *(list_nodes, list_edges, list_edge_links)
            ):
                nodes, edges, edge_links = pad_graph(
                    nodes=nodes,
                    edges=edges,
                    edge_links=edge_links,
                    max_n_nodes=max_n_nodes,
                    max_n_edges=max_n_edges,
                )
                new_list_nodes.append(nodes)
                new_list_edges.append(edges)
                new_list_edge_links.append(edge_links)

            buffers_entry[
                buffers_entry_indices_struct[EDGE_LINKS]
            ] = new_list_edge_links
            buffers_entry[buffers_entry_indices_struct[EDGES]] = new_list_edges
            buffers_entry[buffers_entry_indices_struct[NODES]] = new_list_nodes

    elif is_graph_dict_multiinput(buffers_entry_indices_struct):
        for buffers_entry_indices_substruct in buffers_entry_indices_struct.values():
            pad_graph_buffers(
                buffers_entry=buffers_entry,
                buffers_entry_indices_struct=buffers_entry_indices_substruct,
            )

    elif is_masked_obs(buffers_entry_indices_struct):
        pad_graph_buffers(
            buffers_entry=buffers_entry,
            buffers_entry_indices_struct=buffers_entry_indices_struct[TRUE_OBS],
        )


def monkey_patch_agent_collector(graph2node: bool = False) -> None:
    """Monkey patch rllib so that buffers pad graph arrays if necessary."""
    if graph2node:
        AgentCollector._cache_in_np = agent_collector_graph2node_cache_in_np
    else:
        AgentCollector._cache_in_np = agent_collector_graph_cache_in_np
