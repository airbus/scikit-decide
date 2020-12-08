from typing import Iterable, Any, Tuple, Union
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction, \
    get_default_objective_setup, build_evaluate_function_aggregated
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialMethodRCPSP, RCPSPSolution
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.lp_model import LP_Solver_MRSCPSP
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolverName
from skdecide.builders.discrete_optimization.generic_tools.lns_mip import ConstraintHandler, InitialSolution
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPSolution, \
    MS_RCPSPModel, MS_RCPSPSolution_Variant
import random


# TODO, better call the rcpsp_based_solvers directly.
class InitialSolutionMS_RCPSP(InitialSolution):
    def __init__(self, problem: MS_RCPSPModel,
                 params_objective_function: ParamsObjectiveFunction=None,
                 initial_method: InitialMethodRCPSP=InitialMethodRCPSP.PILE):
        self.problem = problem
        self.params_objective_function = params_objective_function
        if self.params_objective_function is None:
            self.params_objective_function = get_default_objective_setup(problem=self.problem)
        self.aggreg, _ = build_evaluate_function_aggregated(problem=self.problem,
                                                            params_objective_function=self.params_objective_function)
        self.initial_method = initial_method

    def get_starting_solution(self) -> ResultStorage:
        multi_skill_rcpsp = self.problem.build_multimode_rcpsp_calendar_representative()
        from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialSolutionRCPSP
        init_solution = InitialSolutionRCPSP(problem=multi_skill_rcpsp,
                                             params_objective_function=self.params_objective_function,
                                             initial_method=self.initial_method)
        s = init_solution.get_starting_solution()
        list_solution_fits = []
        for s, fit in s.list_solution_fits:
            sol: RCPSPSolution = s
            mode = sol.rcpsp_modes
            modes = {i+2: mode[i] for i in range(len(mode))}
            modes[self.problem.source_task] = 1
            modes[self.problem.sink_task] = 1
            # ms_rcpsp_solution = MS_RCPSPSolution(problem=self.problem,
            #                                      modes=modes,
            #                                      schedule=sol.rcpsp_schedule,
            #                                      employee_usage=None)
            ms_rcpsp_solution = MS_RCPSPSolution_Variant(problem=self.problem,
                                                         priority_list_task=sol.rcpsp_permutation,
                                                         modes_vector=sol.rcpsp_modes,
                                                         priority_worker_per_task=[[w for w in self.problem.employees]
                                                                                   for i
                                                                                   in
                                                                                   range(self.problem.n_jobs_non_dummy)])
            list_solution_fits += [(ms_rcpsp_solution, self.aggreg(ms_rcpsp_solution))]
        return ResultStorage(list_solution_fits=list_solution_fits,
                             mode_optim=self.params_objective_function.sense_function)


class ConstraintHandlerFixStartTime(ConstraintHandler):
    def __init__(self, problem: MS_RCPSPModel,
                 fraction_fix_start_time: float=0.9):
        self.problem = problem
        self.fraction_fix_start_time = fraction_fix_start_time

    def adding_constraint_from_results_store(self, milp_solver: LP_Solver_MRSCPSP,
                                             result_storage: ResultStorage) -> Iterable[Any]:

        nb_jobs = self.problem.nb_tasks
        constraints_dict = {}
        current_solution, fit = result_storage.get_best_solution_fit()

        start = []
        for j in current_solution.schedule:
            start_time_j = current_solution.schedule[j]["start_time"]
            mode = current_solution.modes[j]
            start += [(milp_solver.start_times_task[j], start_time_j)]
            start += [(milp_solver.modes[j][mode], 1)]
            for m in milp_solver.modes[j]:
                start += [(milp_solver.modes[j][m], 1 if mode == m else 0)]
        milp_solver.model.start = start
        # Fix start time for a subset of task.
        jobs_to_fix = set(random.sample(current_solution.rcpsp_schedule.keys(),
                                        int(self.fraction_fix_start_time * nb_jobs)))
        constraints_dict["fix_start_time"] = []
        for job_to_fix in jobs_to_fix:
            constraints_dict["fix_start_time"].append(milp_solver.model.add_constr(
                milp_solver.start_times_task[job_to_fix]-current_solution.schedule[job_to_fix]["start_time"] == 0))
        if milp_solver.lp_solver == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        return constraints_dict

    def remove_constraints_from_previous_iteration(self,
                                                   milp_solver: LP_Solver_MRSCPSP,
                                                   previous_constraints: Iterable[Any]):
        milp_solver.model.remove(previous_constraints["fix_start_time"])
        if milp_solver.lp_solver == MilpSolverName.GRB:
            milp_solver.model.solver.update()


class ConstraintHandlerStartTimeIntervalMRCPSP(ConstraintHandler):
    def __init__(self, problem: MS_RCPSPModel,
                 fraction_to_fix: float=0.9,
                 minus_delta: int=2,
                 plus_delta: int=2):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta

    def adding_constraint_from_results_store(self, milp_solver: LP_Solver_MRSCPSP, result_storage: ResultStorage) -> Iterable[
        Any]:
        current_solution: MS_RCPSPSolution = result_storage.get_best_solution()
        # st = milp_solver.start_solution
        # if self.problem.evaluate(st)["makespan"] < self.problem.evaluate(current_solution)["makespan"]:
        #    current_solution = st
        start = []
        for j in current_solution.schedule:
            start_time_j = current_solution.schedule[j]["start_time"]
            mode = current_solution.modes[j]
            start += [(milp_solver.start_times_task[j], start_time_j)]
            start += [(milp_solver.modes[j][mode], 1)]
            for m in milp_solver.modes[j]:
                start += [(milp_solver.modes[j][m], 1 if mode == m else 0)]
        milp_solver.model.start = start
        constraints_dict = {}
        constraints_dict["range_start_time"] = []
        max_time = max([current_solution.schedule[x]["end_time"]
                        for x in current_solution.schedule])
        last_jobs = [x for x in current_solution.schedule
                     if current_solution.schedule[x]["end_time"] >= max_time - 5]
        nb_jobs = self.problem.nb_tasks
        jobs_to_fix = set(random.sample(current_solution.schedule.keys(),
                                        int(self.fraction_to_fix * nb_jobs)))
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        for job in jobs_to_fix:
            start_time_j = current_solution.schedule[job]["start_time"]
            min_st = max(start_time_j - self.minus_delta, 0)
            max_st = min(start_time_j + self.plus_delta, max_time)
            constraints_dict["range_start_time"].append(milp_solver.model.add_constr(milp_solver.start_times_task[job]
                                                                                     <= max_st))
            constraints_dict["range_start_time"].append(milp_solver.model.add_constr(milp_solver.start_times_task[job]
                                                                                     >= min_st))
        if milp_solver.lp_solver == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        return constraints_dict

    def remove_constraints_from_previous_iteration(self, milp_solver: LP_Solver_MRSCPSP, previous_constraints: Iterable[Any]):
        milp_solver.model.remove(previous_constraints["range_start_time"])
        if milp_solver.lp_solver == MilpSolverName.GRB:
            milp_solver.model.solver.update()
