from ray.rllib.algorithms.ppo import PPOTorchPolicy

from skdecide.hub.solver.ray_rllib.gnn.policy.torch_graph_policy import TorchGraphPolicy


class PPOTorchGraphPolicy(TorchGraphPolicy, PPOTorchPolicy):
    ...
