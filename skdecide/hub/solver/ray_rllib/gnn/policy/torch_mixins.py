from ray.rllib.policy.torch_mixins import ValueNetworkMixin

from skdecide.hub.solver.ray_rllib.gnn.policy.sample_batch import GraphSampleBatch


class ValueNetworkGraphMixin(ValueNetworkMixin):
    def __init__(self, config):
        if config.get("use_gae") or config.get("vtrace"):
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def value(**input_dict):
                input_dict = GraphSampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value
