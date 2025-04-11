(window.webpackJsonp=window.webpackJsonp||[]).push([[119],{631:function(t,e,a){"use strict";a.r(e);var r=a(38),s=Object(r.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"hub-solver-ray-rllib-action-masking-models-tf-parametric-actions"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hub-solver-ray-rllib-action-masking-models-tf-parametric-actions"}},[t._v("#")]),t._v(" hub.solver.ray_rllib.action_masking.models.tf.parametric_actions")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),a("skdecide-summary")],1),t._v(" "),a("h2",{attrs:{id:"tfparametricactionsmodel"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tfparametricactionsmodel"}},[t._v("#")]),t._v(" TFParametricActionsModel")]),t._v(" "),a("p",[t._v("Parametric action model that handles the dot product and masking and\nthat also learns action embeddings. TensorFlow version.")]),t._v(" "),a("p",[t._v("This assumes the outputs are logits for a single Categorical action dist.")]),t._v(" "),a("h3",{attrs:{id:"constructor"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[t._v("#")]),t._v(" Constructor "),a("Badge",{attrs:{text:"TFParametricActionsModel",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"TFParametricActionsModel",sig:{params:[{name:"obs_space"},{name:"action_space"},{name:"num_outputs"},{name:"model_config"},{name:"name"},{name:"**kw"}]}}}),t._v(" "),a("p",[t._v("Initializes a TFModelV2 instance.")]),t._v(" "),a("p",[t._v("Here is an example implementation for a subclass\n"),a("code",[t._v("MyModelClass(TFModelV2)")]),t._v("::")]),t._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[t._v("def __init__(self, *args, **kwargs):\n    super(MyModelClass, self).__init__(*args, **kwargs)\n    input_layer = tf.keras.layers.Input(...)\n    hidden_layer = tf.keras.layers.Dense(...)(input_layer)\n    output_layer = tf.keras.layers.Dense(...)(hidden_layer)\n    value_layer = tf.keras.layers.Dense(...)(hidden_layer)\n    self.base_model = tf.keras.Model(\n        input_layer, [output_layer, value_layer])\n")])])]),a("h3",{attrs:{id:"context"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#context"}},[t._v("#")]),t._v(" context "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"context",sig:{params:[{name:"self"}],return:"<class 'contextlib.AbstractContextManager'>"}}}),t._v(" "),a("p",[t._v("Returns a contextmanager for the current TF graph.")]),t._v(" "),a("h3",{attrs:{id:"custom-loss"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#custom-loss"}},[t._v("#")]),t._v(" custom_loss "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"custom_loss",sig:{params:[{name:"self"},{name:"policy_loss",annotation:"typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]"},{name:"loss_inputs",annotation:"typing.Dict[str, typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]"}],return:"typing.Union[typing.List[typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]], <built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]"}}}),t._v(" "),a("p",[t._v("Override to customize the loss function used to optimize this model.")]),t._v(" "),a("p",[t._v("This can be used to incorporate self-supervised losses (by defining\na loss over existing input and output tensors of this model), and\nsupervised losses (by defining losses over a variable-sharing copy of\nthis model's layers).")]),t._v(" "),a("p",[t._v("You can find an runnable example in examples/custom_loss.py.")]),t._v(" "),a("p",[t._v("Args:\npolicy_loss: List of or single policy loss(es) from the policy.\nloss_inputs: map of input placeholders for rollout data.")]),t._v(" "),a("p",[t._v("Returns:\nList of or scalar tensor for the customized loss(es) for this\nmodel.")]),t._v(" "),a("h3",{attrs:{id:"forward"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#forward"}},[t._v("#")]),t._v(" forward "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"forward",sig:{params:[{name:"self"},{name:"input_dict"},{name:"state"},{name:"seq_lens"}]}}}),t._v(" "),a("p",[t._v("Call the model with the given input tensors and state.")]),t._v(" "),a("p",[t._v("Any complex observations (dicts, tuples, etc.) will be unpacked by\n"),a("strong",[t._v("call")]),t._v(' before being passed to forward(). To access the flattened\nobservation tensor, refer to input_dict["obs_flat"].')]),t._v(" "),a("p",[t._v("This method can be called any number of times. In eager execution,\neach call to forward() will eagerly evaluate the model. In symbolic\nexecution, each call to forward creates a computation graph that\noperates over the variables of this model (i.e., shares weights).")]),t._v(" "),a("p",[t._v("Custom models should override this instead of "),a("strong",[t._v("call")]),t._v(".")]),t._v(" "),a("p",[t._v('Args:\ninput_dict: dictionary of input tensors, including "obs",\n"obs_flat", "prev_action", "prev_reward", "is_training",\n"eps_id", "agent_id", "infos", and "t".\nstate: list of state tensors with sizes matching those\nreturned by get_initial_state + the batch dimension\nseq_lens: 1d tensor holding input sequence lengths')]),t._v(" "),a("p",[t._v("Returns:\nA tuple consisting of the model output tensor of size\n[BATCH, num_outputs] and the list of new RNN state(s) if any.")]),t._v(" "),a("p",[t._v(".. testcode::\n:skipif: True")]),t._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[t._v('import numpy as np\nfrom ray.rllib.models.modelv2 import ModelV2\nclass MyModel(ModelV2):\n    # ...\n    def forward(self, input_dict, state, seq_lens):\n        model_out, self._value_out = self.base_model(\n            input_dict["obs"])\n        return model_out, state\n')])])]),a("h3",{attrs:{id:"get-initial-state"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-initial-state"}},[t._v("#")]),t._v(" get_initial_state "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_initial_state",sig:{params:[{name:"self"}],return:"typing.List[typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]"}}}),t._v(" "),a("p",[t._v("Get the initial recurrent state values for the model.")]),t._v(" "),a("p",[t._v("Returns:\nList of np.array (for tf) or Tensor (for torch) objects containing the\ninitial hidden state of an RNN, if applicable.")]),t._v(" "),a("p",[t._v(".. testcode::\n:skipif: True")]),t._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[t._v("import numpy as np\nfrom ray.rllib.models.modelv2 import ModelV2\nclass MyModel(ModelV2):\n    # ...\n    def get_initial_state(self):\n        return [\n            np.zeros(self.cell_size, np.float32),\n            np.zeros(self.cell_size, np.float32),\n        ]\n")])])]),a("h3",{attrs:{id:"is-time-major"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-time-major"}},[t._v("#")]),t._v(" is_time_major "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"is_time_major",sig:{params:[{name:"self"}],return:"<class 'bool'>"}}}),t._v(" "),a("p",[t._v("If True, data for calling this ModelV2 must be in time-major format.")]),t._v(" "),a("p",[t._v("Returns\nWhether this ModelV2 requires a time-major (TxBx...) data\nformat.")]),t._v(" "),a("h3",{attrs:{id:"last-output"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#last-output"}},[t._v("#")]),t._v(" last_output "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"last_output",sig:{params:[{name:"self"}],return:"typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]"}}}),t._v(" "),a("p",[t._v("Returns the last output returned from calling the model.")]),t._v(" "),a("h3",{attrs:{id:"metrics"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#metrics"}},[t._v("#")]),t._v(" metrics "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"metrics",sig:{params:[{name:"self"}],return:"typing.Dict[str, typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]"}}}),t._v(" "),a("p",[t._v("Override to return custom metrics from your model.")]),t._v(" "),a("p",[t._v('The stats will be reported as part of the learner stats, i.e.,\ninfo.learner.[policy_id, e.g. "default_policy"].model.key1=metric1')]),t._v(" "),a("p",[t._v("Returns:\nThe custom metrics for this model.")]),t._v(" "),a("h3",{attrs:{id:"register-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#register-variables"}},[t._v("#")]),t._v(" register_variables "),a("Badge",{attrs:{text:"TFModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"register_variables",sig:{params:[{name:"self"},{name:"variables",annotation:"typing.List[typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]"}],return:null}}}),t._v(" "),a("p",[t._v("Register the given list of variables with this model.")]),t._v(" "),a("h3",{attrs:{id:"trainable-variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#trainable-variables"}},[t._v("#")]),t._v(" trainable_variables "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"trainable_variables",sig:{params:[{name:"self"},{name:"as_dict",default:"False",annotation:"<class 'bool'>"}],return:"typing.Union[typing.List[typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]], typing.Dict[str, typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]]"}}}),t._v(" "),a("p",[t._v("Returns the list of trainable variables for this model.")]),t._v(" "),a("p",[t._v("Args:\nas_dict: Whether variables should be returned as dict-values\n(using descriptive keys).")]),t._v(" "),a("p",[t._v("Returns:\nThe list (or dict if "),a("code",[t._v("as_dict")]),t._v(" is True) of all trainable\n(tf)/requires_grad (torch) variables of this ModelV2.")]),t._v(" "),a("h3",{attrs:{id:"update-ops"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#update-ops"}},[t._v("#")]),t._v(" update_ops "),a("Badge",{attrs:{text:"TFModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"update_ops",sig:{params:[{name:"self"}],return:"typing.List[typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]"}}}),t._v(" "),a("p",[t._v("Return the list of update ops for this model.")]),t._v(" "),a("p",[t._v("For example, this should include any BatchNorm update ops.")]),t._v(" "),a("h3",{attrs:{id:"value-function"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#value-function"}},[t._v("#")]),t._v(" value_function "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"value_function",sig:{params:[{name:"self"}]}}}),t._v(" "),a("p",[t._v("Returns the value function output for the most recent forward pass.")]),t._v(" "),a("p",[t._v("Note that a "),a("code",[t._v("forward")]),t._v(" call has to be performed first, before this\nmethods can return anything and thus that calling this method does not\ncause an extra forward pass through the network.")]),t._v(" "),a("p",[t._v("Returns:\nValue estimate tensor of shape [BATCH].")]),t._v(" "),a("h3",{attrs:{id:"variables"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#variables"}},[t._v("#")]),t._v(" variables "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"variables",sig:{params:[{name:"self"},{name:"as_dict",default:"False",annotation:"<class 'bool'>"}],return:"typing.Union[typing.List[typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]], typing.Dict[str, typing.Union[<built-in function array>, ForwardRef('jnp.ndarray'), ForwardRef('tf.Tensor'), ForwardRef('torch.Tensor')]]]"}}}),t._v(" "),a("p",[t._v("Returns the list (or a dict) of variables for this model.")]),t._v(" "),a("p",[t._v("Args:\nas_dict: Whether variables should be returned as dict-values\n(using descriptive str keys).")]),t._v(" "),a("p",[t._v("Returns:\nThe list (or dict if "),a("code",[t._v("as_dict")]),t._v(" is True) of all variables of this\nModelV2.")]),t._v(" "),a("h3",{attrs:{id:"annotated-type"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#annotated-type"}},[t._v("#")]),t._v(" _annotated_type "),a("Badge",{attrs:{text:"ModelV2",type:"warn"}})],1)],1)}),[],!1,null,null,null);e.default=s.exports}}]);