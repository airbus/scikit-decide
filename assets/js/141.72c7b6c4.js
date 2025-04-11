(window.webpackJsonp=window.webpackJsonp||[]).push([[141],{654:function(t,a,e){"use strict";e.r(a);var s=e(38),i=Object(s.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"hub-solver-stable-baselines-autoregressive-common-distributions"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#hub-solver-stable-baselines-autoregressive-common-distributions"}},[t._v("#")]),t._v(" hub.solver.stable_baselines.autoregressive.common.distributions")]),t._v(" "),e("div",{staticClass:"custom-block tip"},[e("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),e("skdecide-summary")],1),t._v(" "),e("h2",{attrs:{id:"multimaskablecategoricaldistribution"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#multimaskablecategoricaldistribution"}},[t._v("#")]),t._v(" MultiMaskableCategoricalDistribution")]),t._v(" "),e("p",[t._v("Distribution for variable-length multidiscrete actions with partial masking on each component.")]),t._v(" "),e("p",[t._v("This is meant for autoregressive prediction.")]),t._v(" "),e("p",[t._v("The distribution is considered as the joint distribution of discrete distributions (MaskableCategoricalDistribution)\nwith the possibility to mask each marginal.\nThis distribution is meant to be used for autoregressive action:")]),t._v(" "),e("ul",[e("li",[t._v("Each component is sampled sequentially")]),t._v(" "),e("li",[t._v("The partial mask for the next component is conditioned by the previous components")]),t._v(" "),e("li",[t._v("It is possible to have missing components when this has no meaning for the action.\nthis corresponds in the simulation to\n"),e("ul",[e("li",[t._v("either not initialized marginal (if all samples discard the component)")]),t._v(" "),e("li",[t._v("0 masks for the given sample (the partial mask row corresponding to the sample has only 0's)")])])])]),t._v(" "),e("p",[t._v("When computing entropy of the distribution or log-probability of an action, we add only contribution\nof marginal distributions for which we have an actual component (dropping the one with a 0-mask).")]),t._v(" "),e("p",[t._v("As this distribution is used to sample component by component, the sample(), and mode() methods are left\nunimplemented.")]),t._v(" "),e("h3",{attrs:{id:"constructor"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[t._v("#")]),t._v(" Constructor "),e("Badge",{attrs:{text:"MultiMaskableCategoricalDistribution",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"MultiMaskableCategoricalDistribution",sig:{params:[{name:"distributions",annotation:"list[sb3_contrib.common.maskable.distributions.MaskableCategoricalDistribution]"}]}}}),t._v(" "),e("p",[t._v("Initialize self.  See help(type(self)) for accurate signature.")]),t._v(" "),e("h3",{attrs:{id:"actions-from-params"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#actions-from-params"}},[t._v("#")]),t._v(" actions_from_params "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"actions_from_params",sig:{params:[{name:"self"},{name:"*args"},{name:"**kwargs"}],return:"<class 'torch.Tensor'>"}}}),t._v(" "),e("p",[t._v("Returns samples from the probability distribution\ngiven its parameters.")]),t._v(" "),e("p",[t._v(":return: actions")]),t._v(" "),e("h3",{attrs:{id:"entropy"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#entropy"}},[t._v("#")]),t._v(" entropy "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"entropy",sig:{params:[{name:"self"}],return:"typing.Optional[torch.Tensor]"}}}),t._v(" "),e("p",[t._v("Returns Shannon's entropy of the probability")]),t._v(" "),e("p",[t._v(":return: the entropy, or None if no analytical form is known")]),t._v(" "),e("h3",{attrs:{id:"get-actions"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#get-actions"}},[t._v("#")]),t._v(" get_actions "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"get_actions",sig:{params:[{name:"self"},{name:"deterministic",default:"False",annotation:"<class 'bool'>"}],return:"<class 'torch.Tensor'>"}}}),t._v(" "),e("p",[t._v("Return actions according to the probability distribution.")]),t._v(" "),e("p",[t._v(":param deterministic:\n:return:")]),t._v(" "),e("h3",{attrs:{id:"log-prob"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#log-prob"}},[t._v("#")]),t._v(" log_prob "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"log_prob",sig:{params:[{name:"self"},{name:"x",annotation:"<class 'torch.Tensor'>"}],return:"<class 'torch.Tensor'>"}}}),t._v(" "),e("p",[t._v("Returns the log likelihood")]),t._v(" "),e("p",[t._v(":param x: the taken action\n:return: The log likelihood of the distribution")]),t._v(" "),e("h3",{attrs:{id:"log-prob-from-params"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#log-prob-from-params"}},[t._v("#")]),t._v(" log_prob_from_params "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"log_prob_from_params",sig:{params:[{name:"self"},{name:"*args"},{name:"**kwargs"}],return:"typing.Tuple[torch.Tensor, torch.Tensor]"}}}),t._v(" "),e("p",[t._v("Returns samples and the associated log probabilities\nfrom the probability distribution given its parameters.")]),t._v(" "),e("p",[t._v(":return: actions and log prob")]),t._v(" "),e("h3",{attrs:{id:"mode"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#mode"}},[t._v("#")]),t._v(" mode "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"mode",sig:{params:[{name:"self"}],return:"<class 'torch.Tensor'>"}}}),t._v(" "),e("p",[t._v("Returns the most likely action (deterministic output)\nfrom the probability distribution")]),t._v(" "),e("p",[t._v(":return: the stochastic action")]),t._v(" "),e("h3",{attrs:{id:"proba-distribution"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#proba-distribution"}},[t._v("#")]),t._v(" proba_distribution "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"proba_distribution",sig:{params:[{name:"self",annotation:"~SelfDistribution"},{name:"*args"},{name:"**kwargs"}],return:"~SelfDistribution"}}}),t._v(" "),e("p",[t._v("Set parameters of the distribution.")]),t._v(" "),e("p",[t._v(":return: self")]),t._v(" "),e("h3",{attrs:{id:"proba-distribution-net"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#proba-distribution-net"}},[t._v("#")]),t._v(" proba_distribution_net "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"proba_distribution_net",sig:{params:[{name:"self"},{name:"*args"},{name:"**kwargs"}],return:"typing.Union[torch.nn.modules.module.Module, typing.Tuple[torch.nn.modules.module.Module, torch.nn.parameter.Parameter]]"}}}),t._v(" "),e("p",[t._v("Create the layers and parameters that represent the distribution.")]),t._v(" "),e("p",[t._v("Subclasses must define this, but the arguments and return type vary between\nconcrete classes.")]),t._v(" "),e("h3",{attrs:{id:"sample"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#sample"}},[t._v("#")]),t._v(" sample "),e("Badge",{attrs:{text:"Distribution",type:"warn"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"sample",sig:{params:[{name:"self"}],return:"<class 'torch.Tensor'>"}}}),t._v(" "),e("p",[t._v("Returns a sample from the probability distribution")]),t._v(" "),e("p",[t._v(":return: the stochastic action")]),t._v(" "),e("h3",{attrs:{id:"set-proba-distribution-component"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#set-proba-distribution-component"}},[t._v("#")]),t._v(" set_proba_distribution_component "),e("Badge",{attrs:{text:"MultiMaskableCategoricalDistribution",type:"tip"}})],1),t._v(" "),e("skdecide-signature",{attrs:{name:"set_proba_distribution_component",sig:{params:[{name:"self"},{name:"i_component",annotation:"<class 'int'>"},{name:"action_component_logits",annotation:"<class 'torch.Tensor'>"}],return:null}}}),t._v(" "),e("p",[t._v("Fix parameters of the marginal distribution.")]),t._v(" "),e("p",[t._v("We allow to modify dynamically the marginal dimension by inferring it from\nlast dimension of "),e("code",[t._v("action_component_logits")]),t._v(".\nThis is useful when the dimension of the marginal can change during rollout\n(e.g. when this predict node id's of a graph whose structure vary)")])],1)}),[],!1,null,null,null);a.default=i.exports}}]);