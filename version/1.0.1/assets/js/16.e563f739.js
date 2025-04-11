(window.webpackJsonp=window.webpackJsonp||[]).push([[16],{530:function(t,s,e){"use strict";e.r(s);var n=e(38),i=Object(n.a)({},(function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("skdecide-spec",{attrs:{isSolver:""},scopedSlots:t._u([{key:"Solver",fn:function(){return[e("p",[t._v("This is the highest level solver class (inheriting top-level class for each mandatory solver characteristic).")]),t._v(" "),e("p",[t._v("This helper class can be used as the main base class for solvers.")]),t._v(" "),e("p",[t._v("Typical use:")]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("MySolver")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("Solver"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("p",[t._v('with "..." replaced when needed by a number of classes from following domain characteristics (the ones in\nparentheses are optional):')]),t._v(" "),e("ul",[e("li",[e("strong",[t._v("(assessability)")]),t._v(": Utilities -> QValues")]),t._v(" "),e("li",[e("strong",[t._v("(policy)")]),t._v(": Policies -> UncertainPolicies -> DeterministicPolicies")]),t._v(" "),e("li",[e("strong",[t._v("(restorability)")]),t._v(": Restorable")])])]},proxy:!0},{key:"DeterministicPolicySolver",fn:function(){return[e("p",[t._v("This is a typical deterministic policy solver class.")]),t._v(" "),e("p",[t._v("This helper class can be used as an alternate base class for domains, inheriting the following:")]),t._v(" "),e("ul",[e("li",[t._v("Solver")]),t._v(" "),e("li",[t._v("DeterministicPolicies")])]),t._v(" "),e("p",[t._v("Typical use:")]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("MySolver")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("DeterministicPolicySolver"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])]),e("div",{staticClass:"custom-block tip"},[e("p",{staticClass:"custom-block-title"},[t._v("TIP")]),t._v(" "),e("p",[t._v("It is also possible to refine any alternate base class, like for instance:")]),t._v(" "),e("div",{staticClass:"language-python extra-class"},[e("pre",{pre:!0,attrs:{class:"language-python"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("MySolver")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("DeterministicPolicySolver"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" QValues"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])])])]},proxy:!0},{key:"Utilities",fn:function(){return[e("p",[t._v("A solver must inherit this class if it can provide the utility function (i.e. value function).")])]},proxy:!0},{key:"QValues",fn:function(){return[e("p",[t._v("A solver must inherit this class if it can provide the Q function (i.e. action-value function).")])]},proxy:!0},{key:"FromInitialState",fn:function(){return[e("p",[t._v('"A solver must inherit this class if it can solve only from the initial state')])]},proxy:!0},{key:"FromAnyState",fn:function(){return[e("p",[t._v("A solver must inherit this class if it can solve from any given state.")])]},proxy:!0},{key:"ParallelSolver",fn:function(){return[e("p",[t._v("A solver must inherit this class if it wants to call several cloned parallel domains in separate concurrent processes.\nThe solver is meant to be called either within a 'with' context statement, or to be cleaned up using the close() method.")])]},proxy:!0},{key:"Policies",fn:function(){return[e("p",[t._v("A solver must inherit this class if it computes a stochastic policy as part of the solving process.")])]},proxy:!0},{key:"UncertainPolicies",fn:function(){return[e("p",[t._v("A solver must inherit this class if it computes a stochastic policy (providing next action distribution\nexplicitly) as part of the solving process.")])]},proxy:!0},{key:"DeterministicPolicies",fn:function(){return[e("p",[t._v("A solver must inherit this class if it computes a deterministic policy as part of the solving process.")])]},proxy:!0},{key:"Restorable",fn:function(){return[e("p",[t._v("A solver must inherit this class if its state can be saved and reloaded (to continue computation later on or\nreuse its solution).")])]},proxy:!0}])})],1)}),[],!1,null,null,null);s.default=i.exports}}]);