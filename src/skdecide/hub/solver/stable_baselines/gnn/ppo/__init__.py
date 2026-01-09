from ..common.policies import GNNActorCriticPolicy, MultiInputGNNActorCriticPolicy
from .ppo import GraphPPO as GraphPPO

GraphInputPolicy = GNNActorCriticPolicy
MultiInputPolicy = MultiInputGNNActorCriticPolicy
