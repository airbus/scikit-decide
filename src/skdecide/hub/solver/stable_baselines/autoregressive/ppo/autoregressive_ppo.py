import io
import pathlib
from typing import Any, ClassVar, Dict, Optional, Type, Union

import gymnasium as gym
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv

from skdecide.hub.solver.stable_baselines.autoregressive.common.on_policy_algorithm import (
    ApplicableActionsGraphOnPolicyAlgorithm,
    ApplicableActionsOnPolicyAlgorithm,
)
from skdecide.hub.solver.stable_baselines.autoregressive.common.policies import (
    AutoregressiveActorCriticPolicy,
    AutoregressiveGNNActorCriticPolicy,
    AutoregressiveGraph2NodeActorCriticPolicy,
    AutoregressiveHeteroGraph2NodeActorCriticPolicy,
)


class AutoregressivePPO(ApplicableActionsOnPolicyAlgorithm, MaskablePPO):
    """Proximal Policy Optimization algorithm (PPO) with autoregressive action prediction.

    The action is multidiscrete and each component is predicted by using the observation + the previous components
    (already predicted). A -1 in the components is valid and interpreted as "no component" and we assume that no further
    components have a meaning for this particular action (correspond to actions with a variable number of parameters).

    See the policy `AutoregressiveActorCriticPolicy` for more details.

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": AutoregressiveActorCriticPolicy,
    }


class AutoregressiveGraphPPO(
    ApplicableActionsGraphOnPolicyAlgorithm, AutoregressivePPO
):
    """Proximal Policy Optimization algorithm (PPO) with autoregressive action prediction and graph observation.

    As for AutoregressivePPO, components are predicted via neural networks applied to
    observations + previous components.

    Two default policies that derive from `AutoregressiveActorCriticPolicy`:
        - "GraphInputPolicy":  With no more hypotheses on actions structure, we simply use a GNN for feature extraction.
          See `skdecide.hub.solver.utils.gnn.torch_layers.GraphFeaturesExtractor` for more details.
        - "Graph2NodePolicy": We suppose that some action components are actually nodes of the observation graph.
            (given by a policy arg `graph2node_components` as alist of booleans). So we apply
            `skdecide.hub.solver.utils.gnn.torch_layers.Graph2NodeLayer` to predict these components.
            By default we assume that the first component is not a graph node but the others are.

    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphInputPolicy": AutoregressiveGNNActorCriticPolicy,
        "Graph2NodePolicy": AutoregressiveGraph2NodeActorCriticPolicy,
        "HeteroGraph2NodePolicy": AutoregressiveHeteroGraph2NodeActorCriticPolicy,
    }

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        try:
            return super().load(
                path=path,
                env=env,
                device=device,
                custom_objects=custom_objects,
                print_system_info=print_system_info,
                force_reset=force_reset,
                **kwargs,
            )
        except ValueError as e:
            # maybe error because different actions spaces (ex: apply on same pddl domain, different problem)
            # can be ok if same number of components and dim differ only for graph2node components,
            # but we need to update the action space
            data, params, pytorch_variables = load_from_zip_file(path)  # stored params
            if issubclass(
                data["policy_class"], AutoregressiveGraph2NodeActorCriticPolicy
            ):
                if (
                    "policy_kwargs" not in data
                    or "n_graph2node_components" not in data["policy_kwargs"]
                ):
                    default_n_graph2node_components_kwargs = dict(
                        action_space=data["action_space"]
                    )
                    if issubclass(
                        data["policy_class"],
                        AutoregressiveHeteroGraph2NodeActorCriticPolicy,
                    ):
                        default_n_graph2node_components_kwargs[
                            "action_components_node_flag_indices"
                        ] = data["policy_kwargs"]["action_components_node_flag_indices"]
                    n_graph2node_components = data[
                        "policy_class"
                    ].default_n_graph2node_components(
                        **default_n_graph2node_components_kwargs
                    )
                else:
                    n_graph2node_components = data["policy_kwargs"][
                        "n_graph2node_components"
                    ]
                # check action spaces have still same length
                assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
                assert len(env.action_space.nvec) == len(data["action_space"].nvec), (
                    f"Action spaces must have same length: len({env.action_space}.nvec) != len({data['action_space']}.nvec)"
                )
                # check independent from graph component have same dim
                for i_component in range(
                    len(env.action_space.nvec) - n_graph2node_components
                ):
                    assert (
                        env.action_space.nvec[i_component]
                        == data["action_space"].nvec[i_component]
                    ), (
                        f"Action spaces independent components must have same dim: comp #{i_component} of {env.action_space} and {data['action_space']}"
                    )

                # update action space
                data["action_space"] = env.action_space
                # create in-memory storage with updated data and retry loading from it
                with io.BytesIO() as file:
                    save_to_zip_file(
                        file,
                        data=data,
                        params=params,
                        pytorch_variables=pytorch_variables,
                    )
                    return super().load(
                        path=file,
                        env=env,
                        device=device,
                        custom_objects=custom_objects,
                        print_system_info=print_system_info,
                        force_reset=force_reset,
                        **kwargs,
                    )
            # else still error
            raise e
