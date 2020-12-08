from typing import Iterable, Any, Union

from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialSolutionRCPSP, InitialMethodRCPSP

from skdecide.builders.discrete_optimization.generic_tools.do_problem import get_default_objective_setup

from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP

from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.solver.cp_solvers import CP_RCPSP_MZN, CP_MRCPSP_MZN
from skdecide.builders.discrete_optimization.generic_tools.lns_cp import ConstraintHandler, SolverDO, LNS_CP
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
import random


class ConstraintHandlerStartTimeInterval_CP(ConstraintHandler):
    def __init__(self, problem: RCPSPModel,
                 fraction_to_fix: float=0.9,  # TODO not really fix
                 minus_delta: int=2,
                 plus_delta: int=2):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta

    def adding_constraint_from_results_store(self, cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN],
                                             child_instance,
                                             result_storage: ResultStorage) -> Iterable[Any]:
        constraints_dict = {}
        current_solution, fit = result_storage.get_best_solution_fit()
        max_time = max([current_solution.rcpsp_schedule[x]["end_time"]
                        for x in current_solution.rcpsp_schedule])
        last_jobs = [x for x in current_solution.rcpsp_schedule
                     if current_solution.rcpsp_schedule[x]["end_time"] >= max_time-5]
        nb_jobs = self.problem.n_jobs + 2
        jobs_to_fix = set(random.sample(current_solution.rcpsp_schedule.keys(),
                                        int(self.fraction_to_fix * nb_jobs)))
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        list_strings = []
        for job in jobs_to_fix:
            start_time_j = current_solution.rcpsp_schedule[job]["start_time"]
            min_st = max(start_time_j-self.minus_delta, 0)
            max_st = min(start_time_j+self.plus_delta, max_time)
            if isinstance(cp_solver, CP_RCPSP_MZN):
                string1 = "constraint s[" + str(job) + "] <= " + str(max_st) + ";\n"
                string2 = "constraint s[" + str(job) + "] >= " + str(min_st) + ";\n"
            elif isinstance(cp_solver, CP_MRCPSP_MZN):
                string1 = "constraint start[" + str(job) + "] <= " + str(max_st) + ";\n"
                string2 = "constraint start[" + str(job) + "] >= " + str(min_st) + ";\n"
            list_strings += [string1]
            list_strings += [string2]
            child_instance.add_string(string1)
            child_instance.add_string(string2)
        return list_strings

    def remove_constraints_from_previous_iteration(self,
                                                   cp_solver: CP_RCPSP_MZN,
                                                   child_instance,
                                                   previous_constraints: Iterable[Any]):
        pass


class LNS_CP_RCPSP_SOLVER(SolverDO):
    def __init__(self, rcpsp_model: RCPSPModel, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.solver = CP_MRCPSP_MZN(rcpsp_model=self.rcpsp_model,
                               cp_solver_name=CPSolverName.CHUFFED)
        self.solver.init_model()
        self.parameters_cp = ParametersCP.default()
        params_objective_function = get_default_objective_setup(problem=self.rcpsp_model)
        # constraint_handler = ConstraintHandlerFixStartTime(problem=rcpsp_problem,
        #                                                    fraction_fix_start_time=0.5)
        self.constraint_handler = ConstraintHandlerStartTimeInterval_CP(problem=self.rcpsp_model,
                                                                        fraction_to_fix=0.6,
                                                                        minus_delta=5,
                                                                        plus_delta=5)
        self.initial_solution_provider = InitialSolutionRCPSP(problem=self.rcpsp_model,
                                                              initial_method=InitialMethodRCPSP.LS,
                                                              params_objective_function=params_objective_function)
        self.lns_solver = LNS_CP(problem=self.rcpsp_model,
                                 cp_solver=self.solver,
                                 initial_solution_provider=self.initial_solution_provider,
                                 constraint_handler=self.constraint_handler,
                                 params_objective_function=params_objective_function)

    def solve(self, **kwargs) -> ResultStorage:
        return self.lns_solver.solve_lns(parameters_cp=self.parameters_cp,
                                         nb_iteration_lns=kwargs.get("nb_iteration_lns", 100))