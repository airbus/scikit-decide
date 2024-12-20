import functools

from ray.rllib import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

from skdecide.hub.solver.ray_rllib.gnn.utils.torch_utils import convert_to_torch_tensor


class TorchGraphPolicy(TorchPolicyV2):
    def _lazy_tensor_dict(self, postprocessed_batch: SampleBatch, device=None):
        if not isinstance(postprocessed_batch, SampleBatch):
            postprocessed_batch = SampleBatch(postprocessed_batch)
        postprocessed_batch.set_get_interceptor(
            functools.partial(convert_to_torch_tensor, device=device or self.device)
        )
        return postprocessed_batch
