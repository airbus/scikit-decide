(window.webpackJsonp=window.webpackJsonp||[]).push([[102],{616:function(e,t,a){"use strict";a.r(t);var r=a(38),s=Object(r.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"hub-solver-do-solver-do-solver-scheduling"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hub-solver-do-solver-do-solver-scheduling"}},[e._v("#")]),e._v(" hub.solver.do_solver.do_solver_scheduling")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),a("skdecide-summary")],1),e._v(" "),a("h2",{attrs:{id:"solvingmethod"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solvingmethod"}},[e._v("#")]),e._v(" SolvingMethod")]),e._v(" "),a("p",[e._v("Type of discrete-optimization algorithm to use")]),e._v(" "),a("h3",{attrs:{id:"cp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cp"}},[e._v("#")]),e._v(" CP "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with constraint programming solver")]),e._v(" "),a("h3",{attrs:{id:"ga"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ga"}},[e._v("#")]),e._v(" GA "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with genetic algorithm")]),e._v(" "),a("h3",{attrs:{id:"lns-cp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#lns-cp"}},[e._v("#")]),e._v(" LNS_CP "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with large neighborhood search + CP solver")]),e._v(" "),a("h3",{attrs:{id:"lns-lp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#lns-lp"}},[e._v("#")]),e._v(" LNS_LP "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with large neighborhood search + LP solver")]),e._v(" "),a("h3",{attrs:{id:"lp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#lp"}},[e._v("#")]),e._v(" LP "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with linear programming solver")]),e._v(" "),a("h3",{attrs:{id:"ls"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ls"}},[e._v("#")]),e._v(" LS "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with local search algorithm (hill climber or simulated annealing)")]),e._v(" "),a("h3",{attrs:{id:"pile"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pile"}},[e._v("#")]),e._v(" PILE "),a("Badge",{attrs:{text:"SolvingMethod",type:"tip"}})],1),e._v(" "),a("p",[e._v("solve scheduling problem with greedy queue method")]),e._v(" "),a("h2",{attrs:{id:"build-solver"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#build-solver"}},[e._v("#")]),e._v(" build_solver")]),e._v(" "),a("skdecide-signature",{attrs:{name:"build_solver",sig:{params:[{name:"solving_method",annotation:"Optional[SolvingMethod]"},{name:"solver_type",annotation:"Optional[type[SolverDO]]"},{name:"do_domain",annotation:"Problem"}],return:"tuple[type[SolverDO], dict[str, Any]]"}}}),e._v(" "),a("p",[e._v("Build the discrete-optimization solver for a given solving method")]),e._v(" "),a("h4",{attrs:{id:"parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("solving_method")]),e._v(": method of the solver (enum)")]),e._v(" "),a("li",[a("strong",[e._v("solver_type")]),e._v(": potentially a solver class already specified by the do_solver")]),e._v(" "),a("li",[a("strong",[e._v("do_domain")]),e._v(": discrete-opt problem to solve.")])]),e._v(" "),a("h4",{attrs:{id:"returns"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("A class of do-solver, associated with some default parameters to be passed to its constructor and solve function\n(and potentially init_model function)")]),e._v(" "),a("h2",{attrs:{id:"from-solution-to-policy"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#from-solution-to-policy"}},[e._v("#")]),e._v(" from_solution_to_policy")]),e._v(" "),a("skdecide-signature",{attrs:{name:"from_solution_to_policy",sig:{params:[{name:"solution",annotation:"Union[RcpspSolution, MultiskillRcpspSolution, VariantMultiskillRcpspSolution]"},{name:"domain",annotation:"SchedulingDomain"},{name:"policy_method_params",annotation:"PolicyMethodParams"}],return:"PolicyRCPSP"}}}),e._v(" "),a("p",[e._v("Create a PolicyRCPSP object (a skdecide policy) from a scheduling solution\nfrom the discrete-optimization library.")]),e._v(" "),a("h2",{attrs:{id:"dosolver"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dosolver"}},[e._v("#")]),e._v(" DOSolver")]),e._v(" "),a("p",[e._v("Wrapper of discrete-optimization solvers for scheduling problems")]),e._v(" "),a("h4",{attrs:{id:"attributes"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#attributes"}},[e._v("#")]),e._v(" Attributes")]),e._v(" "),a("ul",[a("li",[e._v("policy_method_params:  params for the returned policy.")]),e._v(" "),a("li",[e._v("method: method of the discrete-optim solver used")]),e._v(" "),a("li",[e._v("solver_type: direct method class of do solver (will be used instead of method if solver_type is not None)")]),e._v(" "),a("li",[e._v("dict_params: specific params passed to the do-solver")]),e._v(" "),a("li",[e._v("callback: scikit-decide callback to be called inside do-solver when relevant.")])]),e._v(" "),a("h3",{attrs:{id:"constructor"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[e._v("#")]),e._v(" Constructor "),a("Badge",{attrs:{text:"DOSolver",type:"tip"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"DOSolver",sig:{params:[{name:"domain_factory",annotation:"Callable[[], Domain]"},{name:"policy_method_params",default:"None",annotation:"Optional[PolicyMethodParams]"},{name:"method",default:"None",annotation:"Optional[SolvingMethod]"},{name:"do_solver_type",default:"None",annotation:"Optional[type[SolverDO]]"},{name:"dict_params",default:"None",annotation:"Optional[dict[Any, Any]]"},{name:"callback",default:"<lambda function>",annotation:"Callable[[DOSolver], bool]"},{name:"policy_method_params_kwargs",default:"None",annotation:"Optional[dict[str, Any]]"}]}}}),e._v(" "),a("h4",{attrs:{id:"parameters-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain_factory")]),e._v(": A callable with no argument returning the domain to solve (can be a mere domain class).\nThe resulting domain will be auto-cast to the level expected by the solver.")])]),e._v(" "),a("h3",{attrs:{id:"autocast"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#autocast"}},[e._v("#")]),e._v(" autocast "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"autocast",sig:{params:[{name:"self"},{name:"domain_cls",default:"None",annotation:"Optional[type[Domain]]"}],return:"None"}}}),e._v(" "),a("p",[e._v("Autocast itself to the level corresponding to the given domain class.")]),e._v(" "),a("h4",{attrs:{id:"parameters-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-3"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain_cls")]),e._v(": the domain class to which level the solver needs to autocast itself.\nBy default, use the original domain factory passed to its constructor.")])]),e._v(" "),a("h3",{attrs:{id:"check-domain"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-domain"}},[e._v("#")]),e._v(" check_domain "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"check_domain",sig:{params:[{name:"domain",annotation:"Domain"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether a domain is compliant with this solver type.")]),e._v(" "),a("p",[e._v("By default, "),a("code",[e._v("Solver.check_domain()")]),e._v(" provides some boilerplate code and internally\ncalls "),a("code",[e._v("Solver._check_domain_additional()")]),e._v(' (which returns True by default but can be overridden  to define\nspecific checks in addition to the "domain requirements"). The boilerplate code automatically checks whether all\ndomain requirements are met.')]),e._v(" "),a("h4",{attrs:{id:"parameters-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-4"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": The domain to check.")])]),e._v(" "),a("h4",{attrs:{id:"returns-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the domain is compliant with the solver type (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"complete-with-default-hyperparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#complete-with-default-hyperparameters"}},[e._v("#")]),e._v(" complete_with_default_hyperparameters "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"complete_with_default_hyperparameters",sig:{params:[{name:"kwargs",annotation:"dict[str, Any]"},{name:"names",default:"None",annotation:"Optional[list[str]]"}]}}}),e._v(" "),a("p",[e._v("Add missing hyperparameters to kwargs by using default values")]),e._v(" "),a("p",[e._v("Args:\nkwargs: keyword arguments to complete (e.g. for "),a("code",[e._v("__init__")]),e._v(", "),a("code",[e._v("init_model")]),e._v(", or "),a("code",[e._v("solve")]),e._v(")\nnames: names of the hyperparameters to add if missing.\nBy default, all available hyperparameters.")]),e._v(" "),a("p",[e._v("Returns:\na new dictionary, completion of kwargs")]),e._v(" "),a("h3",{attrs:{id:"copy-and-update-hyperparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#copy-and-update-hyperparameters"}},[e._v("#")]),e._v(" copy_and_update_hyperparameters "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"copy_and_update_hyperparameters",sig:{params:[{name:"names",default:"None",annotation:"Optional[list[str]]"},{name:"**kwargs_by_name",annotation:"dict[str, Any]"}],return:"list[Hyperparameter]"}}}),e._v(" "),a("p",[e._v("Copy hyperparameters definition of this class and update them with specified kwargs.")]),e._v(" "),a("p",[e._v("This is useful to define hyperparameters for a child class\nfor which only choices of the hyperparameter change for instance.")]),e._v(" "),a("p",[e._v("Args:\nnames: names of hyperparameters to copy. Default to all.\n**kwargs_by_name: for each hyperparameter specified by its name,\nthe attributes to update. If a given hyperparameter name is not specified,\nthe hyperparameter is copied without further update.")]),e._v(" "),a("p",[e._v("Returns:")]),e._v(" "),a("h3",{attrs:{id:"get-default-hyperparameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-default-hyperparameters"}},[e._v("#")]),e._v(" get_default_hyperparameters "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_default_hyperparameters",sig:{params:[{name:"names",default:"None",annotation:"Optional[list[str]]"}],return:"dict[str, Any]"}}}),e._v(" "),a("p",[e._v("Get hyperparameters default values.")]),e._v(" "),a("p",[e._v("Args:\nnames: names of the hyperparameters to choose.\nBy default, all available hyperparameters will be suggested.")]),e._v(" "),a("p",[e._v("Returns:\na mapping between hyperparameter's name_in_kwargs and its default value (None if not specified)")]),e._v(" "),a("h3",{attrs:{id:"get-domain-requirements"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-domain-requirements"}},[e._v("#")]),e._v(" get_domain_requirements "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_domain_requirements",sig:{params:[],return:"list[type]"}}}),e._v(" "),a("p",[e._v("Get domain requirements for this solver class to be applicable.")]),e._v(" "),a("p",[e._v("Domain requirements are classes from the "),a("code",[e._v("skdecide.builders.domain")]),e._v(" package that the domain needs to inherit from.")]),e._v(" "),a("h4",{attrs:{id:"returns-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("A list of classes to inherit from.")]),e._v(" "),a("h3",{attrs:{id:"get-hyperparameter"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-hyperparameter"}},[e._v("#")]),e._v(" get_hyperparameter "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_hyperparameter",sig:{params:[{name:"name",annotation:"str"}],return:"Hyperparameter"}}}),e._v(" "),a("p",[e._v("Get hyperparameter from given name.")]),e._v(" "),a("h3",{attrs:{id:"get-hyperparameters-by-name"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-hyperparameters-by-name"}},[e._v("#")]),e._v(" get_hyperparameters_by_name "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_hyperparameters_by_name",sig:{params:[],return:"dict[str, Hyperparameter]"}}}),e._v(" "),a("p",[e._v("Mapping from name to corresponding hyperparameter.")]),e._v(" "),a("h3",{attrs:{id:"get-hyperparameters-names"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-hyperparameters-names"}},[e._v("#")]),e._v(" get_hyperparameters_names "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_hyperparameters_names",sig:{params:[],return:"list[str]"}}}),e._v(" "),a("p",[e._v("List of hyperparameters names.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action"}},[e._v("#")]),e._v(" get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Get the next deterministic action (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-5"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which next action is requested.")])]),e._v(" "),a("h4",{attrs:{id:"returns-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The next deterministic action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-distribution"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution"}},[e._v("#")]),e._v(" get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),e._v(" "),a("p",[e._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-6"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The probabilistic distribution of next action.")]),e._v(" "),a("h3",{attrs:{id:"is-policy-defined-for"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for"}},[e._v("#")]),e._v(" is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether the solver's current policy is defined for the given observation.")]),e._v(" "),a("h4",{attrs:{id:"parameters-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-7"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-6"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the policy is defined for the given observation memory (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"reset"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#reset"}},[e._v("#")]),e._v(" reset "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"reset",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Reset whatever is needed on this solver before running a new episode.")]),e._v(" "),a("p",[e._v("This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM\npolicy network, which carries information about past observations seen in the previous episode).")]),e._v(" "),a("h3",{attrs:{id:"sample-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action"}},[e._v("#")]),e._v(" sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Sample an action for the given observation (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-8"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which an action must be sampled.")])]),e._v(" "),a("h4",{attrs:{id:"returns-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-7"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The sampled action.")]),e._v(" "),a("h3",{attrs:{id:"solve"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solve"}},[e._v("#")]),e._v(" solve "),a("Badge",{attrs:{text:"FromInitialState",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"solve",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Run the solving process.")]),e._v(" "),a("p",[e._v("After solving by calling self._solve(), autocast itself so that rollout methods apply\nto the domain original characteristics.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("The nature of the solutions produced here depends on other solver's characteristics like\n"),a("code",[e._v("policy")]),e._v(" and "),a("code",[e._v("assessibility")]),e._v(".")])]),e._v(" "),a("h3",{attrs:{id:"suggest-hyperparameter-with-optuna"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#suggest-hyperparameter-with-optuna"}},[e._v("#")]),e._v(" suggest_hyperparameter_with_optuna "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"suggest_hyperparameter_with_optuna",sig:{params:[{name:"trial",annotation:"optuna.trial.Trial"},{name:"name",annotation:"str"},{name:"prefix",default:"",annotation:"str"},{name:"**kwargs"}],return:"Any"}}}),e._v(" "),a("p",[e._v("Suggest hyperparameter value during an Optuna trial.")]),e._v(" "),a("p",[e._v("This can be used during Optuna hyperparameters tuning.")]),e._v(" "),a("p",[e._v("Args:\ntrial: optuna trial during hyperparameters tuning\nname: name of the hyperparameter to choose\nprefix: prefix to add to optuna corresponding parameter name\n(useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)\n**kwargs: options for optuna hyperparameter suggestions")]),e._v(" "),a("p",[e._v("Returns:")]),e._v(" "),a("p",[e._v("kwargs can be used to pass relevant arguments to")]),e._v(" "),a("ul",[a("li",[e._v("trial.suggest_float()")]),e._v(" "),a("li",[e._v("trial.suggest_int()")]),e._v(" "),a("li",[e._v("trial.suggest_categorical()")])]),e._v(" "),a("p",[e._v("For instance it can")]),e._v(" "),a("ul",[a("li",[e._v("add a low/high value if not existing for the hyperparameter\nor override it to narrow the search. (for float or int hyperparameters)")]),e._v(" "),a("li",[e._v("add a step or log argument (for float or int hyperparameters,\nsee optuna.trial.Trial.suggest_float())")]),e._v(" "),a("li",[e._v("override choices for categorical or enum parameters to narrow the search")])]),e._v(" "),a("h3",{attrs:{id:"suggest-hyperparameters-with-optuna"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#suggest-hyperparameters-with-optuna"}},[e._v("#")]),e._v(" suggest_hyperparameters_with_optuna "),a("Badge",{attrs:{text:"Hyperparametrizable",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"suggest_hyperparameters_with_optuna",sig:{params:[{name:"trial",annotation:"optuna.trial.Trial"},{name:"names",default:"None",annotation:"Optional[list[str]]"},{name:"kwargs_by_name",default:"None",annotation:"Optional[dict[str, dict[str, Any]]]"},{name:"fixed_hyperparameters",default:"None",annotation:"Optional[dict[str, Any]]"},{name:"prefix",default:"",annotation:"str"}],return:"dict[str, Any]"}}}),e._v(" "),a("p",[e._v("Suggest hyperparameters values during an Optuna trial.")]),e._v(" "),a("p",[e._v("Args:\ntrial: optuna trial during hyperparameters tuning\nnames: names of the hyperparameters to choose.\nBy default, all available hyperparameters will be suggested.\nIf "),a("code",[e._v("fixed_hyperparameters")]),e._v(" is provided, the corresponding names are removed from "),a("code",[e._v("names")]),e._v(".\nkwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name\nfixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subbrick hyperparameters,\nif the subbrick class is not suggested by this method, but already fixed.\nWill be added to the suggested hyperparameters.\nprefix: prefix to add to optuna corresponding parameters\n(useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)")]),e._v(" "),a("p",[e._v("Returns:\nmapping between the hyperparameter name and its suggested value.\nIf the hyperparameter has an attribute "),a("code",[e._v("name_in_kwargs")]),e._v(", this is used as the key in the mapping\ninstead of the actual hyperparameter name.\nthe mapping is updated with "),a("code",[e._v("fixed_hyperparameters")]),e._v(".")]),e._v(" "),a("p",[e._v("kwargs_by_name[some_name] will be passed as **kwargs to suggest_hyperparameter_with_optuna(name=some_name)")]),e._v(" "),a("h3",{attrs:{id:"check-domain-additional"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#check-domain-additional"}},[e._v("#")]),e._v(" _check_domain_additional "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_check_domain_additional",sig:{params:[{name:"domain",annotation:"Domain"}],return:"bool"}}}),e._v(" "),a("p",[e._v('Check whether the given domain is compliant with the specific requirements of this solver type (i.e. the\nones in addition to "domain requirements").')]),e._v(" "),a("p",[e._v("This is a helper function called by default from "),a("code",[e._v("Solver.check_domain()")]),e._v(". It focuses on specific checks, as\nopposed to taking also into account the domain requirements for the latter.")]),e._v(" "),a("h4",{attrs:{id:"parameters-9"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-9"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("domain")]),e._v(": The domain to check.")])]),e._v(" "),a("h4",{attrs:{id:"returns-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-8"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the domain is compliant with the specific requirements of this solver type (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"cleanup"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cleanup"}},[e._v("#")]),e._v(" _cleanup "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_cleanup",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("Runs cleanup code here, or code to be executed at the exit of a\n'with' context statement.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-2"}},[e._v("#")]),e._v(" _get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Get the next deterministic action (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-10"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-10"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which next action is requested.")])]),e._v(" "),a("h4",{attrs:{id:"returns-9"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-9"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The next deterministic action.")]),e._v(" "),a("h3",{attrs:{id:"get-next-action-distribution-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution-2"}},[e._v("#")]),e._v(" _get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),e._v(" "),a("p",[e._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-11"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-11"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-10"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-10"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The probabilistic distribution of next action.")]),e._v(" "),a("h3",{attrs:{id:"initialize"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#initialize"}},[e._v("#")]),e._v(" _initialize "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_initialize",sig:{params:[{name:"self"}]}}}),e._v(" "),a("p",[e._v("Runs long-lasting initialization code here.")]),e._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-2"}},[e._v("#")]),e._v(" _is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),e._v(" "),a("p",[e._v("Check whether the solver's current policy is defined for the given observation.")]),e._v(" "),a("h4",{attrs:{id:"parameters-12"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-12"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation to consider.")])]),e._v(" "),a("h4",{attrs:{id:"returns-11"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-11"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("True if the policy is defined for the given observation memory (False otherwise).")]),e._v(" "),a("h3",{attrs:{id:"reset-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#reset-2"}},[e._v("#")]),e._v(" _reset "),a("Badge",{attrs:{text:"Solver",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_reset",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Reset whatever is needed on this solver before running a new episode.")]),e._v(" "),a("p",[e._v("This function does nothing by default but can be overridden if needed (e.g. to reset the hidden state of a LSTM\npolicy network, which carries information about past observations seen in the previous episode).")]),e._v(" "),a("h3",{attrs:{id:"sample-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-2"}},[e._v("#")]),e._v(" _sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),e._v(" "),a("p",[e._v("Sample an action for the given observation (from the solver's current policy).")]),e._v(" "),a("h4",{attrs:{id:"parameters-13"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-13"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),a("ul",[a("li",[a("strong",[e._v("observation")]),e._v(": The observation for which an action must be sampled.")])]),e._v(" "),a("h4",{attrs:{id:"returns-12"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-12"}},[e._v("#")]),e._v(" Returns")]),e._v(" "),a("p",[e._v("The sampled action.")]),e._v(" "),a("h3",{attrs:{id:"solve-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#solve-2"}},[e._v("#")]),e._v(" _solve "),a("Badge",{attrs:{text:"FromInitialState",type:"warn"}})],1),e._v(" "),a("skdecide-signature",{attrs:{name:"_solve",sig:{params:[{name:"self"}],return:"None"}}}),e._v(" "),a("p",[e._v("Run the solving process.")]),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[e._v("TIP")]),e._v(" "),a("p",[e._v("The nature of the solutions produced here depends on other solver's characteristics like\n"),a("code",[e._v("policy")]),e._v(" and "),a("code",[e._v("assessibility")]),e._v(".")])])],1)}),[],!1,null,null,null);t.default=s.exports}}]);