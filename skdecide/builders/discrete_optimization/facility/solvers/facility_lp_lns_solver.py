from typing import Iterable, Any

from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction
from skdecide.builders.discrete_optimization.generic_tools.lns_mip import ConstraintHandler, InitialSolution
from skdecide.builders.discrete_optimization.facility.facility_model import FacilityProblem
from skdecide.builders.discrete_optimization.facility.solvers.greedy_solvers import GreedySolverFacility, ResultStorage
from skdecide.builders.discrete_optimization.facility.solvers.facility_lp_solver import LP_Facility_Solver_PyMip, MilpSolverName
import random
from enum import Enum
import mip


class InitialFacilityMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialFacilitySolution(InitialSolution):
    def __init__(self, problem: FacilityProblem,
                 initial_method: InitialFacilityMethod,
                 params_objective_function: ParamsObjectiveFunction):
        self.problem = problem
        self.initial_method = initial_method
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=params_objective_function)

    def get_starting_solution(self):
        if self.initial_method == InitialFacilityMethod.GREEDY:
            greedy_solver = GreedySolverFacility(self.problem,
                                                 params_objective_function=self.params_objective_function)
            return greedy_solver.solve()
        else:
            solution = self.problem.get_dummy_solution()
            fit = self.aggreg_sol(solution)
            return ResultStorage(list_solution_fits=[(solution, fit)],
                                 best_solution=solution,
                                 mode_optim=self.params_objective_function.sense_function)


class ConstraintHandlerFacility(ConstraintHandler):
    def __init__(self, problem: FacilityProblem, fraction_to_fix: float=0.9, skip_first_iter=True):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0
        self.skip_first_iter = skip_first_iter

    def adding_constraint_from_results_store(self, milp_solver: LP_Facility_Solver_PyMip, result_storage: ResultStorage) \
            -> Iterable[Any]:
        if self.iter == 0 and self.skip_first_iter:
            print("Dummy : ")
            print(self.problem.evaluate(result_storage.get_best_solution_fit()[0]))
            self.iter += 1
            return {}
        subpart_customer = set(random.sample(range(self.problem.customer_count),
                                             int(self.fraction_to_fix * self.problem.customer_count)))
        current_solution = result_storage.get_best_solution_fit()[0]
        dict_f_fixed = {}
        dict_f_start = {}
        start = []
        for c in range(self.problem.customer_count):
            dict_f_start[c] = current_solution.facility_for_customers[c]
            if c in subpart_customer:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = milp_solver.variable_decision["x"]
        lns_constraint = {}
        for key in x_var:
            f, c = key
            if f == dict_f_start[c]:
                if isinstance(x_var[f, c], mip.Var):
                    start += [(x_var[f, c], 1)]
            else:
                if isinstance(x_var[f, c], mip.Var):
                    start += [(x_var[f, c], 0)]
            if c in dict_f_fixed:
                if f == dict_f_fixed[c]:
                    if isinstance(x_var[f, c], mip.Var):
                        lns_constraint[(f, c)] = milp_solver.model.add_constr(x_var[key] == 1, name=str((f, c)))
                else:
                    if isinstance(x_var[f, c], mip.Var):
                        lns_constraint[(f, c)] = milp_solver.model.add_constr(x_var[key] == 0, name=str((f, c)))
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        print(len(lns_constraint), " constraints")
        milp_solver.model.start = start
        return lns_constraint

    def remove_constraints_from_previous_iteration(self, milp_solver: LP_Facility_Solver_PyMip, previous_constraints: Iterable[Any]):
        milp_solver.model.remove([previous_constraints[k] for k in previous_constraints])
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
