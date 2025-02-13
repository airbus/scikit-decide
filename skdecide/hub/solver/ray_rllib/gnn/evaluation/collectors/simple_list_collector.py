from ray.rllib import SampleBatch
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollector

from skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch import concat_samples


def policy_collector_build(self: _PolicyCollector) -> SampleBatch:
    # Create batch from our buffers.
    batch = concat_samples(self.batches)
    # Clear batches for future samples.
    self.batches = []
    # Reset agent steps to 0.
    self.agent_steps = 0
    # Add num_grad_updates counter to the policy's batch.
    batch.num_grad_updates = self.policy.num_grad_updates

    return batch


def monkey_patch_policy_collector() -> None:
    """Monkey patch rllib so that concat_samples pad graph arrays if necessary."""
    _PolicyCollector.build = policy_collector_build
