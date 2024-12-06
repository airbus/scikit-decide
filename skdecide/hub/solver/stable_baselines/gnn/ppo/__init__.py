from ..common.policies import GNNActorCriticPolicy, MultiInputGNNActorCriticPolicy
from .ppo import GraphPPO

GraphInputPolicy = GNNActorCriticPolicy
MultiInputPolicy = MultiInputGNNActorCriticPolicy
