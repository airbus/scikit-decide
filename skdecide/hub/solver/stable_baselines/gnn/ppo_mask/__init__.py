from ..common.policies import (
    MaskableGNNActorCriticPolicy,
    MaskableMultiInputGNNActorCriticPolicy,
)
from .ppo_mask import MaskableGraphPPO

GraphInputPolicy = MaskableGNNActorCriticPolicy
MultiInputPolicy = MaskableMultiInputGNNActorCriticPolicy
