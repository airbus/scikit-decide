(window.webpackJsonp=window.webpackJsonp||[]).push([[29],{543:function(t,e,a){"use strict";a.r(e);var s=a(38),o=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"builders-domain-goals"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-goals"}},[t._v("#")]),t._v(" builders.domain.goals")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),a("skdecide-summary")],1),t._v(" "),a("h2",{attrs:{id:"goals"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#goals"}},[t._v("#")]),t._v(" Goals")]),t._v(" "),a("p",[t._v("A domain must inherit this class if it has formalized goals.")]),t._v(" "),a("h3",{attrs:{id:"get-goals"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-goals"}},[t._v("#")]),t._v(" get_goals "),a("Badge",{attrs:{text:"Goals",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_goals",sig:{params:[{name:"self"}],return:"D.T_agent[Space[D.T_observation]]"}}}),t._v(" "),a("p",[t._v("Get the (cached) domain goals space (finite or infinite set).")]),t._v(" "),a("p",[t._v("By default, "),a("code",[t._v("Goals.get_goals()")]),t._v(" internally calls "),a("code",[t._v("Goals._get_goals_()")]),t._v(" the first time and automatically caches its\nvalue to make future calls more efficient (since the goals space is assumed to be constant).")]),t._v(" "),a("div",{staticClass:"custom-block warning"},[a("p",{staticClass:"custom-block-title"},[t._v("WARNING")]),t._v(" "),a("p",[t._v("Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty\nabout whether the goal has been reached or not. This assumption guarantees that any policy that does not\nreach the goal with certainty incurs in infinite expected cost. - "),a("em",[t._v("Geffner, 2013: A Concise Introduction to\nModels and Methods for Automated Planning")])])]),t._v(" "),a("h4",{attrs:{id:"returns"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The goals space.")]),t._v(" "),a("h3",{attrs:{id:"is-goal"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-goal"}},[t._v("#")]),t._v(" is_goal "),a("Badge",{attrs:{text:"Goals",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"is_goal",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_predicate]"}}}),t._v(" "),a("p",[t._v("Indicate whether an observation belongs to the goals.")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("TIP")]),t._v(" "),a("p",[t._v("By default, this function is implemented using the "),a("code",[t._v("skdecide.core.Space.contains()")]),t._v(" function on the domain\ngoals space provided by "),a("code",[t._v("Goals.get_goals()")]),t._v(", but it can be overridden for faster implementations.")])]),t._v(" "),a("h4",{attrs:{id:"parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the observation is a goal (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"get-goals-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-goals-2"}},[t._v("#")]),t._v(" _get_goals "),a("Badge",{attrs:{text:"Goals",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_goals",sig:{params:[{name:"self"}],return:"D.T_agent[Space[D.T_observation]]"}}}),t._v(" "),a("p",[t._v("Get the (cached) domain goals space (finite or infinite set).")]),t._v(" "),a("p",[t._v("By default, "),a("code",[t._v("Goals._get_goals()")]),t._v(" internally calls "),a("code",[t._v("Goals._get_goals_()")]),t._v(" the first time and automatically caches\nits value to make future calls more efficient (since the goals space is assumed to be constant).")]),t._v(" "),a("div",{staticClass:"custom-block warning"},[a("p",{staticClass:"custom-block-title"},[t._v("WARNING")]),t._v(" "),a("p",[t._v("Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty\nabout whether the goal has been reached or not. This assumption guarantees that any policy that does not\nreach the goal with certainty incurs in infinite expected cost. - "),a("em",[t._v("Geffner, 2013: A Concise Introduction to\nModels and Methods for Automated Planning")])])]),t._v(" "),a("h4",{attrs:{id:"returns-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The goals space.")]),t._v(" "),a("h3",{attrs:{id:"get-goals-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-goals-3"}},[t._v("#")]),t._v(" _get_goals_ "),a("Badge",{attrs:{text:"Goals",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_goals_",sig:{params:[{name:"self"}],return:"D.T_agent[Space[D.T_observation]]"}}}),t._v(" "),a("p",[t._v("Get the domain goals space (finite or infinite set).")]),t._v(" "),a("p",[t._v("This is a helper function called by default from "),a("code",[t._v("Goals._get_goals()")]),t._v(", the difference being that the result is\nnot cached here.")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("TIP")]),t._v(" "),a("p",[t._v("The underscore at the end of this function's name is a convention to remind that its result should be\nconstant.")])]),t._v(" "),a("h4",{attrs:{id:"returns-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The goals space.")]),t._v(" "),a("h3",{attrs:{id:"is-goal-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-goal-2"}},[t._v("#")]),t._v(" _is_goal "),a("Badge",{attrs:{text:"Goals",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_is_goal",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_predicate]"}}}),t._v(" "),a("p",[t._v("Indicate whether an observation belongs to the goals.")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("TIP")]),t._v(" "),a("p",[t._v("By default, this function is implemented using the "),a("code",[t._v("skdecide.core.Space.contains()")]),t._v(" function on the domain\ngoals space provided by "),a("code",[t._v("Goals._get_goals()")]),t._v(", but it can be overridden for faster implementations.")])]),t._v(" "),a("h4",{attrs:{id:"parameters-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the observation is a goal (False otherwise).")])],1)}),[],!1,null,null,null);e.default=o.exports}}]);