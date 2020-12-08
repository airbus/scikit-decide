from typing import Union
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_solvers import solvers, solve
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPSolution, MS_RCPSPModel, \
    MS_RCPSPModel_Variant, MS_RCPSPSolution_Variant


class Solver_RCPSP_Based(SolverDO):
    def __init__(self, model: Union[MS_RCPSPModel, MS_RCPSPModel_Variant],
                 method,
                 params_objective_function: ParamsObjectiveFunction=None,
                 **args):
        self.model = model
        self.model_rcpsp = model.build_multimode_rcpsp_calendar_representative()
        self.method = method
        self.args_solve = args
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.model,
                                                       params_objective_function=
                                                       params_objective_function)
        self.args_solve["params_objective_function"] = self.params_objective_function

    def solve(self, **kwargs):
        res_storage = solve(method=self.method,
                            rcpsp_model=self.model_rcpsp,
                            **self.args_solve)
        list_solution_fits = []
        for s, fit in res_storage.list_solution_fits:
            sol: RCPSPSolution = s
            mode = sol.rcpsp_modes
            modes = {i+2: mode[i] for i in range(len(mode))}
            modes[self.model.source_task] = 1
            modes[self.model.sink_task] = 1
            # print(fit, " found by ", self.method.__name__)
            ms_rcpsp_solution = MS_RCPSPSolution_Variant(problem=self.model,
                                                         priority_list_task=sol.rcpsp_permutation,
                                                         modes_vector=sol.rcpsp_modes,
                                                         priority_worker_per_task=[[w for w in self.model.employees]
                                                                                   for i
                                                                                   in
                                                                                   range(self.model.n_jobs_non_dummy)])
            list_solution_fits += [(ms_rcpsp_solution, self.aggreg_from_sol(ms_rcpsp_solution))]
        return ResultStorage(list_solution_fits=list_solution_fits,
                             mode_optim=self.params_objective_function.sense_function)



