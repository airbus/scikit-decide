from typing import Optional

from ray.rllib import Policy
from ray.rllib.algorithms import PPO, AlgorithmConfig

from skdecide.hub.solver.ray_rllib.gnn.algorithms.ppo.ppo_torch_policy import (
    PPOTorchGraphPolicy,
)


class GraphPPO(PPO):
    @classmethod
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[type[Policy]]:
        if config["framework"] == "torch":
            return PPOTorchGraphPolicy
        elif config["framework"] == "tf":
            raise NotImplementedError("GraphPPO implemented for torch context")
        else:
            raise NotImplementedError("GraphPPO implemented for torch context")
