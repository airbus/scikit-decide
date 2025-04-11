(window.webpackJsonp=window.webpackJsonp||[]).push([[106],{620:function(t,o,i){"use strict";i.r(o);var a=i(38),s=Object(a.a)({},(function(){var t=this,o=t.$createElement,i=t._self._c||o;return i("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[i("h1",{attrs:{id:"hub-solver-do-solver-sk-to-do-binding"}},[i("a",{staticClass:"header-anchor",attrs:{href:"#hub-solver-do-solver-sk-to-do-binding"}},[t._v("#")]),t._v(" hub.solver.do_solver.sk_to_do_binding")]),t._v(" "),i("div",{staticClass:"custom-block tip"},[i("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),i("skdecide-summary")],1),t._v(" "),i("h2",{attrs:{id:"from-last-state-to-solution"}},[i("a",{staticClass:"header-anchor",attrs:{href:"#from-last-state-to-solution"}},[t._v("#")]),t._v(" from_last_state_to_solution")]),t._v(" "),i("skdecide-signature",{attrs:{name:"from_last_state_to_solution",sig:{params:[{name:"state",annotation:"State"},{name:"domain",annotation:"SchedulingDomain"}],return:"RcpspSolution"}}}),t._v(" "),i("p",[t._v("Transform a scheduling state into a RcpspSolution\nThis function reads the schedule from the state object and transform it back to a discrete-optimization solution\nobject.")]),t._v(" "),i("h2",{attrs:{id:"build-do-domain"}},[i("a",{staticClass:"header-anchor",attrs:{href:"#build-do-domain"}},[t._v("#")]),t._v(" build_do_domain")]),t._v(" "),i("skdecide-signature",{attrs:{name:"build_do_domain",sig:{params:[{name:"scheduling_domain",annotation:"Union[SingleModeRCPSP, SingleModeRCPSPCalendar, MultiModeRCPSP, MultiModeRCPSPWithCost, MultiModeRCPSPCalendar, MultiModeMultiSkillRCPSP, MultiModeMultiSkillRCPSPCalendar, SingleModeRCPSP_Stochastic_Durations]"}],return:"Union[RcpspProblem, MultiskillRcpspProblem]"}}}),t._v(" "),i("p",[t._v("Transform the scheduling domain (from scikit-decide) into a discrete-optimization problem.")]),t._v(" "),i("p",[t._v("This only works for scheduling template given in the type docstring.")]),t._v(" "),i("h2",{attrs:{id:"build-sk-domain"}},[i("a",{staticClass:"header-anchor",attrs:{href:"#build-sk-domain"}},[t._v("#")]),t._v(" build_sk_domain")]),t._v(" "),i("skdecide-signature",{attrs:{name:"build_sk_domain",sig:{params:[{name:"rcpsp_do_domain",annotation:"Union[MultiskillRcpspProblem, RcpspProblem]"},{name:"varying_ressource",default:"False",annotation:"bool"}],return:"Union[RCPSP, MSRCPSP, MRCPSP, MSRCPSPCalendar]"}}}),t._v(" "),i("p",[t._v("Build a scheduling domain (scikit-decide) from a discrete-optimization problem")])],1)}),[],!1,null,null,null);o.default=s.exports}}]);