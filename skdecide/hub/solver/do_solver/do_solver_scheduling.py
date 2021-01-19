from typing import Union, Callable, Dict, Any

from skdecide.builders.discrete_optimization.generic_tools.ea.ga import Ga

from skdecide.builders.discrete_optimization.rcpsp.solver import PileSolverRCPSP, LP_RCPSP, CP_RCPSP_MZN

from skdecide.builders.scheduling.scheduling_domains import SchedulingDomain, \
    SingleModeRCPSP, SingleModeRCPSPCalendar, MultiModeRCPSP, MultiModeRCPSPCalendar,\
    MultiModeMultiSkillRCPSPCalendar, MultiModeMultiSkillRCPSP, MultiModeRCPSPWithCost
from skdecide.builders.scheduling.scheduling_domains_modelling import State
from skdecide.hub.solver.sgs_policies.sgs_policies import PolicyRCPSP, PolicyMethodParams, BasePolicyMethod
from skdecide.solvers import Solver, DeterministicPolicies
from skdecide.hub.solver.do_solver.sk_to_do_binding import build_do_domain
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, SingleModeRCPSPModel, \
    MultiModeRCPSPModel, RCPSPModelCalendar, RCPSPSolution
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant,\
    MS_RCPSPSolution_Variant, MS_RCPSPSolution


from enum import Enum


class D(SchedulingDomain):
    pass


class SolvingMethod(Enum):
    PILE = 0
    GA = 1
    LS = 2
    LP = 3
    CP = 4
    LNS_LP = 5
    LNS_CP = 6
    LNS_CP_CALENDAR = 7
    # New algorithm, similar to lns, adding iterativelyu constraint to fulfill calendar constraints..


def build_solver(solving_method: SolvingMethod, do_domain):
    if isinstance(do_domain, (RCPSPModelCalendar, RCPSPModel, MultiModeRCPSPModel)):
        from skdecide.builders.discrete_optimization.rcpsp.rcpsp_solvers import look_for_solver, solvers_map
        available = look_for_solver(do_domain)
        solving_method_to_str = {SolvingMethod.PILE: "greedy",
                                 SolvingMethod.GA: "ga",
                                 SolvingMethod.LS: "ls",
                                 SolvingMethod.LP: "lp",
                                 SolvingMethod.CP: "cp",
                                 SolvingMethod.LNS_LP: "lns-lp",
                                 SolvingMethod.LNS_CP: "lns-cp",
                                 SolvingMethod.LNS_CP_CALENDAR: "lns-cp-calendar"
                                 }
        smap = [(av, solvers_map[av]) for av in available
                if solvers_map[av][0] == solving_method_to_str[solving_method]]
        if len(smap) > 0:
            return smap[0]
    if isinstance(do_domain, (MS_RCPSPModel, MS_RCPSPModel, MultiModeRCPSPModel)):
        from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_solvers import look_for_solver, solvers_map
        available = look_for_solver(do_domain)
        solving_method_to_str = {SolvingMethod.PILE: "greedy",
                                 SolvingMethod.GA: "ga",
                                 SolvingMethod.LS: "ls",
                                 SolvingMethod.LP: "lp",
                                 SolvingMethod.CP: "cp",
                                 SolvingMethod.LNS_LP: "lns-lp",
                                 SolvingMethod.LNS_CP: "lns-cp",
                                 SolvingMethod.LNS_CP_CALENDAR: "lns-cp-calendar"}
        print([(av, solvers_map[av]) for av in available])
        smap = [(av, solvers_map[av]) for av in available
                if solvers_map[av][0] == solving_method_to_str[solving_method]]
        if len(smap) > 0:
            return smap[0]

    return None


def from_solution_to_policy(solution: Union[RCPSPSolution,
                                            MS_RCPSPSolution,
                                            MS_RCPSPSolution_Variant],
                            domain,
                            policy_method_params: PolicyMethodParams):
    permutation_task = None
    modes_dictionnary = None
    schedule = None
    resource_allocation = None
    resource_allocation_priority = None
    if isinstance(solution, RCPSPSolution):
        permutation_task = sorted(solution.rcpsp_schedule,
                                  key=lambda x: (solution.rcpsp_schedule[x]["start_time"], x))
        schedule = solution.rcpsp_schedule
        modes_dictionnary = {}
        modes_dictionnary[1] = 1
        modes_dictionnary[solution.problem.n_jobs + 2] = 1
        for i in range(len(solution.rcpsp_modes)):
            modes_dictionnary[i + 2] = solution.rcpsp_modes[i]
    if isinstance(solution, MS_RCPSPSolution):
        permutation_task = sorted(solution.schedule,
                                  key=lambda x: (solution.schedule[x]["start_time"], x))
        schedule = solution.schedule
        employees = sorted(domain.get_resource_units_names())
        resource_allocation = {task: [employees[i] for i in solution.employee_usage[task].keys()] # warning here...
                               for task in solution.employee_usage}
        modes_dictionnary = solution.modes
    if isinstance(solution, MS_RCPSPSolution_Variant):
        resource_allocation_priority = solution.priority_worker_per_task
        permutation_task = sorted(solution.schedule,
                                  key=lambda x: (solution.schedule[x]["start_time"], x))
        modes_dictionnary = {}
        modes_dictionnary[1] = 1
        modes_dictionnary[solution.problem.n_jobs_non_dummy + 2] = 1
        for i in range(len(solution.modes_vector)):
            modes_dictionnary[i + 2] = solution.modes_vector[i]
        employees = sorted(domain.get_resource_units_names())
        resource_allocation = {task: [employees[i] for i in solution.employee_usage[task].keys()]  # warning here...
                               for task in solution.employee_usage}
    return PolicyRCPSP(domain=domain,
                       policy_method_params=policy_method_params,
                       permutation_task=permutation_task,
                       modes_dictionnary=modes_dictionnary,
                       schedule=schedule,
                       resource_allocation=resource_allocation,
                       resource_allocation_priority=resource_allocation_priority)


class DOSolver(Solver, DeterministicPolicies):
    T_domain = D

    def __init__(self,
                 policy_method_params: PolicyMethodParams,
                 method: SolvingMethod=SolvingMethod.PILE,
                 dict_params: Dict[Any, Any]=None):
        self.method = method
        self.policy_method_params = policy_method_params
        self.dict_params = dict_params
        if self.dict_params is None:
            self.dict_params = {}

    def get_available_methods(self, domain: SchedulingDomain):
        do_domain = build_do_domain(domain)
        if isinstance(do_domain, (MS_RCPSPModel)):
            from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_solvers import look_for_solver, solvers_map
            available = look_for_solver(do_domain)
        elif isinstance(do_domain, (SingleModeRCPSPModel, RCPSPModel, MultiModeRCPSPModel)):
            from skdecide.builders.discrete_optimization.rcpsp.rcpsp_solvers import look_for_solver, solvers_map
            available = look_for_solver(do_domain)
        smap = [(av, solvers_map[av]) for av in available]
        print("available solvers :", smap)
        return smap

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self.domain = domain_factory()
        self.do_domain = build_do_domain(self.domain)
        solvers = build_solver(solving_method=self.method,
                               do_domain=self.do_domain)
        solver_class = solvers[0]
        key, params = solvers[1]
        for k in params:
            if k not in self.dict_params:
                self.dict_params[k] = params[k]
        self.solver = solver_class(self.do_domain, **self.dict_params)
        try:
            self.solver.init_model(**self.dict_params)
        except:
            pass
        result_storage = self.solver.solve(**self.dict_params)
        best_solution: RCPSPSolution = result_storage.get_best_solution()
        fits = self.do_domain.evaluate(best_solution)
        print("Best solution fitness found : ", fits)
        self.best_solution = best_solution
        print("Satisfiable ", self.do_domain.satisfy(self.best_solution))
        self.policy_object = from_solution_to_policy(solution=best_solution,
                                                     domain=self.domain,
                                                     policy_method_params=self.policy_method_params)

    def get_external_policy(self)->PolicyRCPSP:
        return self.policy_object

    def compute_external_policy(self, policy_method_params: PolicyMethodParams):
        return from_solution_to_policy(solution=self.best_solution,
                                       domain=self.domain,
                                       policy_method_params=policy_method_params)

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self.policy_object.get_next_action(observation=observation)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return self.policy_object.is_policy_defined_for(observation=observation)


