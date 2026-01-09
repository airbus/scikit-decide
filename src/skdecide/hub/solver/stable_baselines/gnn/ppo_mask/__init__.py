from ..common.policies import (
    MaskableGNNActorCriticPolicy,
    MaskableMultiInputGNNActorCriticPolicy,
)
from .ppo_mask import MaskableGraphPPO as MaskableGraphPPO

GraphInputPolicy = MaskableGNNActorCriticPolicy
MultiInputPolicy = MaskableMultiInputGNNActorCriticPolicy
