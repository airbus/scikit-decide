from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.vrp.vrp_model import VrpProblem, trivial_solution


class GreedyVRPSolver(SolverDO):
    def __init__(self, vrp_model: VrpProblem, params_objective_function: ParamsObjectiveFunction=None):
        self.vrp_model = vrp_model
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.vrp_model,
                                                       params_objective_function=params_objective_function)

    def solve(self, **kwargs):
        sol, fit = trivial_solution(self.vrp_model)
        fit = self.aggreg_sol(sol)
        return ResultStorage(list_solution_fits=[(sol, fit)],
                             mode_optim=self.params_objective_function.sense_function)
