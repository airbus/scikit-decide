(window.webpackJsonp=window.webpackJsonp||[]).push([[128],{639:function(e,t,a){"use strict";a.r(t);var s=a(38),r=Object(s.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"solvers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solvers"}},[e._v("#")]),e._v(" solvers")]),e._v(" "),a("p",[e._v("This module contains base classes for quickly building solvers.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),a("skdecide-summary")],1),e._v(" "),a("h2",{attrs:{id:"solver"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solver"}},[e._v("#")]),e._v(" Solver")]),e._v(" "),a("p",[e._v("This is the highest level solver class (inheriting top-level class for each mandatory solver characteristic).")]),e._v(" "),a("p",[e._v("This helper class can be used as the main base class for solvers.")]),e._v(" "),a("p",[e._v("Typical use:")]),e._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("class")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[e._v("MySolver")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),e._v("Solver"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\n")])])]),a("p",[e._v('with "..." replaced when needed by a number of classes from following domain characteristics (the ones in\nparentheses are optional):')]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("(assessability)")]),e._v(": Utilities -> QValues")]),e._v(" "),a("li",[a("strong",[e._v("(policy)")]),e._v(": Policies -> UncertainPolicies -> DeterministicPolicies")]),e._v(" "),a("li",[a("strong",[e._v("(restorability)")]),e._v(": Restorable")])]),e._v(" "),a("h3",{attrs:{id:"constructor"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[e._v("#")]),e._v(" Constructor "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"Solver",sig:{params:[{name:"domain_factory",annotation:"Callable[[], Domain]"}]}}}),e._v(" "),a("h4",{attrs:{id:"parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain_factory")]),e._v(": A callable with no argument returning the domain to solve (can be a mere domain class).\nThe resulting domain will be auto-cast to the level expected by the solver.")])]),e._v(" "),a("h3",{attrs:{id:"autocast"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#autocast"}},[e._v("#")]),e._v(" autocast "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"autocast",sig:{params:[{name:"self"},{name:"domain_cls",default:"None",annotation:"Optional[Type[Domain]]"}],return:"None"}}}),e._v(" "),a("p",[e._v("Autocast itself to the level corresponding to the given domain class.")]),e._v(" "),a("h4",{attrs:{id:"parameters-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain_cls")]),e._v(": the domain class to which level the solver needs to autocast itself.\nBy default, use the original domain factory passed to its constructor.")])]),e._v(" "),a("h3",{attrs:{id:"check-domain"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-domain"}},[e._v("#")]),e._v(" check_domain "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"check_domain",sig:{params:[{name:"domain",annotation:"Domain"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether a domain is compliant with this solver type.")]),e._v(" "),a("p",[e._v("By default, "),a("code",[e._v("Solver.check_domain()")]),e._v(" provides some boilerplate code and internally\ncalls "),a("code",[e._v("Solver._check_domain_additional()")]),e._v(' (which returns True by default but can be overridden  to define\nspecific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all\ndomain requirements are met.')]),e._v(" "),a("h4",{attrs:{id:"parameters-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-3"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": The domain to check.")])]),e._v(" "),a("h4",{attrs:{id:"returns"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the domain is compliant with the solver type (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"get-domain-requirements"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-domain-requirements"}},[e._v("#")]),e._v(" get_domain_requirements "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_domain_requirements",sig:{params:[],return:"List[type]"}}}),e._v(" "),a("p",[e._v("Get domain requirements for this solver class to be applicable.")]),e._v(" "),a("p",[e._v("Domain requirements are classes from the "),a("code",[e._v("skdecide.builders.domain")]),e._v(" package that the domain needs to inherit from.")]),e._v(" "),a("h4",{attrs:{id:"returns-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("A list of classes to inherit from.")]),e._v(" "),a("h3",{attrs:{id:"reset"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#reset"}},[e._v("#")]),e._v(" reset "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"reset",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Reset whatever is needed on this solver before running a new episode.")]),e._v(" "),a("p",[e._v("This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM\npolicy network, which carries information about past observations seen in the previous episode).")]),e._v(" "),a("h3",{attrs:{id:"solve"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solve"}},[e._v("#")]),e._v(" solve "),a("Badge",{attrs:{text:"FromInitialState",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"solve",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Run the solving process.")]),e._v(" "),a("p",[e._v("After solving by calling self._solve(), autocast itself so that rollout methods apply\nto the domain original characteristics.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("The nature of the solutions produced here depends on other solver's characteristics like\n"),a("code",[e._v("policy")]),e._v(" and "),a("code",[e._v("assessibility")]),e._v(".")])]),e._v(" "),a("h3",{attrs:{id:"check-domain-additional"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-domain-additional"}},[e._v("#")]),e._v(" _check_domain_additional "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_check_domain_additional",sig:{params:[{name:"domain",annotation:"Domain"}],return:"bool"}}}),e._v(" "),a("p",[e._v('Check whether the given domain is compliant with the specific requirements of this solver type (i.e. the\nones in addition to "domain requirements").')]),e._v(" "),a("p",[e._v("This is a helper function called by default from "),a("code",[e._v("Solver.check_domain()")]),e._v(". It focuses on specific checks, as\nopposed to taking also into account the domain requirements for the latter.")]),e._v(" "),a("h4",{attrs:{id:"parameters-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-4"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": The domain to check.")])]),e._v(" "),a("h4",{attrs:{id:"returns-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the domain is compliant with the specific requirements of this solver type (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"cleanup"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cleanup"}},[e._v("#")]),e._v(" _cleanup "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_cleanup",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("Runs cleanup code here, or code to be executed at the exit of a\n'with' context statement.")]),e._v(" "),a("h3",{attrs:{id:"initialize"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#initialize"}},[e._v("#")]),e._v(" _initialize "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_initialize",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("Runs long-lasting initialization code here, or code to be executed at the\nentering of a 'with' context statement.")]),e._v(" "),a("h3",{attrs:{id:"reset-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#reset-2"}},[e._v("#")]),e._v(" _reset "),a("Badge",{attrs:{text:"Solver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_reset",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Reset whatever is needed on this solver before running a new episode.")]),e._v(" "),a("p",[e._v("This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM\npolicy network, which carries information about past observations seen in the previous episode).")]),e._v(" "),a("h3",{attrs:{id:"solve-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solve-2"}},[e._v("#")]),e._v(" _solve "),a("Badge",{attrs:{text:"FromInitialState",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_solve",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Run the solving process.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("The nature of the solutions produced here depends on other solver's characteristics like\n"),a("code",[e._v("policy")]),e._v(" and "),a("code",[e._v("assessibility")]),e._v(".")])]),e._v(" "),a("h2",{attrs:{id:"deterministicpolicysolver"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#deterministicpolicysolver"}},[e._v("#")]),e._v(" DeterministicPolicySolver")]),e._v(" "),a("p",[e._v("This is a typical deterministic policy solver class.")]),e._v(" "),a("p",[e._v("This helper class can be used as an alternate base class for domains, inheriting the following:")]),e._v(" "),a("ul",[a("li",[e._v("Solver")]),e._v(" "),a("li",[e._v("DeterministicPolicies")])]),e._v(" "),a("p",[e._v("Typical use:")]),e._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("class")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[e._v("MySolver")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),e._v("DeterministicPolicySolver"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\n")])])]),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("It is also possible to refine any alternate base class, like for instance:")]),e._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("class")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[e._v("MySolver")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("(")]),e._v("DeterministicPolicySolver"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v(" QValues"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(")")]),e._v("\n")])])])]),e._v(" "),a("h3",{attrs:{id:"constructor-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor-2"}},[e._v("#")]),e._v(" Constructor "),a("Badge",{attrs:{text:"DeterministicPolicySolver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"DeterministicPolicySolver",sig:{params:[{name:"domain_factory",annotation:"Callable[[], Domain]"}]}}}),e._v(" "),a("h4",{attrs:{id:"parameters-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-5"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain_factory")]),e._v(": A callable with no argument returning the domain to solve (can be a mere domain class).\nThe resulting domain will be auto-cast to the level expected by the solver.")])]),e._v(" "),a("h3",{attrs:{id:"autocast-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#autocast-2"}},[e._v("#")]),e._v(" autocast "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"autocast",sig:{params:[{name:"self"},{name:"domain_cls",default:"None",annotation:"Optional[Type[Domain]]"}],return:"None"}}}),e._v(" "),a("p",[e._v("Autocast itself to the level corresponding to the given domain class.")]),e._v(" "),a("h4",{attrs:{id:"parameters-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-6"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain_cls")]),e._v(": the domain class to which level the solver needs to autocast itself.\nBy default, use the original domain factory passed to its constructor.")])]),e._v(" "),a("h3",{attrs:{id:"check-domain-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-domain-2"}},[e._v("#")]),e._v(" check_domain "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"check_domain",sig:{params:[{name:"domain",annotation:"Domain"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether a domain is compliant with this solver type.")]),e._v(" "),a("p",[e._v("By default, "),a("code",[e._v("Solver.check_domain()")]),e._v(" provides some boilerplate code and internally\ncalls "),a("code",[e._v("Solver._check_domain_additional()")]),e._v(' (which returns True by default but can be overridden  to define\nspecific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all\ndomain requirements are met.')]),e._v(" "),a("h4",{attrs:{id:"parameters-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-7"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": The domain to check.")])]),e._v(" "),a("h4",{attrs:{id:"returns-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the domain is compliant with the solver type (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"get-domain-requirements-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-domain-requirements-2"}},[e._v("#")]),e._v(" get_domain_requirements "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_domain_requirements",sig:{params:[],return:"List[type]"}}}),e._v(" "),a("p",[e._v("Get domain requirements for this solver class to be applicable.")]),e._v(" "),a("p",[e._v("Domain requirements are classes from the "),a("code",[e._v("skdecide.builders.domain")]),e._v(" package that the domain needs to inherit from.")]),e._v(" "),a("h4",{attrs:{id:"returns-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("A list of classes to inherit from.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action"}},[e._v("#")]),e._v(" get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Get the next deterministic action (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-8"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which next action is requested.")])]),e._v(" "),a("h4",{attrs:{id:"returns-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-6"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The next deterministic action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-distribution"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution"}},[e._v("#")]),e._v(" get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),e._v(" "),a("p",[e._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-9"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-9"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-7"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The probabilistic distribution of next action.")]),e._v(" "),a("h3",{attrs:{id:"is-policy-defined-for"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for"}},[e._v("#")]),e._v(" is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether the solver's current policy is defined for the given observation.")]),e._v(" "),a("h4",{attrs:{id:"parameters-10"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-10"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-8"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the policy is defined for the given observation memory (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"reset-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#reset-3"}},[e._v("#")]),e._v(" reset "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"reset",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Reset whatever is needed on this solver before running a new episode.")]),e._v(" "),a("p",[e._v("This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM\npolicy network, which carries information about past observations seen in the previous episode).")]),e._v(" "),a("h3",{attrs:{id:"sample-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action"}},[e._v("#")]),e._v(" sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Sample an action for the given observation (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-11"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-11"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which an action must be sampled.")])]),e._v(" "),a("h4",{attrs:{id:"returns-9"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-9"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The sampled action.")]),e._v(" "),a("h3",{attrs:{id:"solve-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solve-3"}},[e._v("#")]),e._v(" solve "),a("Badge",{attrs:{text:"FromInitialState",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"solve",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Run the solving process.")]),e._v(" "),a("p",[e._v("After solving by calling self._solve(), autocast itself so that rollout methods apply\nto the domain original characteristics.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("The nature of the solutions produced here depends on other solver's characteristics like\n"),a("code",[e._v("policy")]),e._v(" and "),a("code",[e._v("assessibility")]),e._v(".")])]),e._v(" "),a("h3",{attrs:{id:"check-domain-additional-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-domain-additional-2"}},[e._v("#")]),e._v(" _check_domain_additional "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_check_domain_additional",sig:{params:[{name:"domain",annotation:"Domain"}],return:"bool"}}}),e._v(" "),a("p",[e._v('Check whether the given domain is compliant with the specific requirements of this solver type (i.e. the\nones in addition to "domain requirements").')]),e._v(" "),a("p",[e._v("This is a helper function called by default from "),a("code",[e._v("Solver.check_domain()")]),e._v(". It focuses on specific checks, as\nopposed to taking also into account the domain requirements for the latter.")]),e._v(" "),a("h4",{attrs:{id:"parameters-12"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-12"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": The domain to check.")])]),e._v(" "),a("h4",{attrs:{id:"returns-10"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-10"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the domain is compliant with the specific requirements of this solver type (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"cleanup-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cleanup-2"}},[e._v("#")]),e._v(" _cleanup "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_cleanup",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("Runs cleanup code here, or code to be executed at the exit of a\n'with' context statement.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-2"}},[e._v("#")]),e._v(" _get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Get the next deterministic action (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-13"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-13"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which next action is requested.")])]),e._v(" "),a("h4",{attrs:{id:"returns-11"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-11"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The next deterministic action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-distribution-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution-2"}},[e._v("#")]),e._v(" _get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),e._v(" "),a("p",[e._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-14"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-14"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-12"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-12"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The probabilistic distribution of next action.")]),e._v(" "),a("h3",{attrs:{id:"initialize-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#initialize-2"}},[e._v("#")]),e._v(" _initialize "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_initialize",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("Runs long-lasting initialization code here, or code to be executed at the\nentering of a 'with' context statement.")]),e._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-2"}},[e._v("#")]),e._v(" _is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether the solver's current policy is defined for the given observation.")]),e._v(" "),a("h4",{attrs:{id:"parameters-15"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-15"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-13"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-13"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the policy is defined for the given observation memory (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"reset-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#reset-4"}},[e._v("#")]),e._v(" _reset "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_reset",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Reset whatever is needed on this solver before running a new episode.")]),e._v(" "),a("p",[e._v("This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM\npolicy network, which carries information about past observations seen in the previous episode).")]),e._v(" "),a("h3",{attrs:{id:"sample-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-2"}},[e._v("#")]),e._v(" _sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Sample an action for the given observation (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-16"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-16"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which an action must be sampled.")])]),e._v(" "),a("h4",{attrs:{id:"returns-14"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-14"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The sampled action.")]),e._v(" "),a("h3",{attrs:{id:"solve-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solve-4"}},[e._v("#")]),e._v(" _solve "),a("Badge",{attrs:{text:"FromInitialState",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_solve",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Run the solving process.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("The nature of the solutions produced here depends on other solver's characteristics like\n"),a("code",[e._v("policy")]),e._v(" and "),a("code",[e._v("assessibility")]),e._v(".")])])],1)}),[],!1,null,null,null);t.default=r.exports}}]);