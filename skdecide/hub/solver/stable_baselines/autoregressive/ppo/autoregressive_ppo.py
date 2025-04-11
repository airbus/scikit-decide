from typing import ClassVar

from sb3_contrib import MaskablePPO
from stable_baselines3.common.policies import BasePolicy

from skdecide.hub.solver.stable_baselines.autoregressive.common.on_policy_algorithm import (
    ApplicableActionsGraphOnPolicyAlgorithm,
    ApplicableActionsOnPolicyAlgorithm,
)
from skdecide.hub.solver.stable_baselines.autoregressive.common.policies import (
    AutoregressiveActorCriticPolicy,
    AutoregressiveGNNActorCriticPolicy,
    AutoregressiveGraph2NodeActorCriticPolicy,
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
    }
