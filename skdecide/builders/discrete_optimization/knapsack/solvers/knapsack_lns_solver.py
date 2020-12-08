from typing import Iterable, Any
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction
from skdecide.builders.discrete_optimization.generic_tools.lns_mip import ConstraintHandler, InitialSolution
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolverName
from skdecide.builders.discrete_optimization.knapsack.knapsack_model import KnapsackModel, KnapsackSolution
from skdecide.builders.discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest, ResultStorage
from skdecide.builders.discrete_optimization.knapsack.solvers.lp_solvers import LPKnapsack
import random
from enum import Enum
import mip


class InitialKnapsackMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialKnapsackSolution(InitialSolution):
    def __init__(self, problem: KnapsackModel,
                 initial_method: InitialKnapsackMethod,
                 params_objective_function: ParamsObjectiveFunction):
        self.problem = problem
        self.initial_method = initial_method
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=params_objective_function)

    def get_starting_solution(self):
        if self.initial_method == InitialKnapsackMethod.GREEDY:
            greedy_solver = GreedyBest(self.problem,
                                       params_objective_function=self.params_objective_function)
            return greedy_solver.solve()
        else:
            solution = self.problem.get_dummy_solution()
            fit = self.aggreg_sol(solution)
            return ResultStorage(list_solution_fits=[(solution, fit)],
                                 best_solution=solution,
                                 mode_optim=self.params_objective_function.sense_function)


class ConstraintHandlerKnapsack(ConstraintHandler):
    def __init__(self, problem: KnapsackModel, fraction_to_fix: float=0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(self, milp_solver: LPKnapsack, result_storage: ResultStorage) \
            -> Iterable[Any]:
        subpart_item = set(random.sample(range(self.problem.nb_items),
                                         int(self.fraction_to_fix * self.problem.nb_items)))
        current_solution: KnapsackSolution = result_storage.get_best_solution_fit()[0]
        dict_f_fixed = {}
        dict_f_start = {}
        start = []
        for c in range(self.problem.nb_items):
            dict_f_start[c] = current_solution.list_taken[c]
            if c in subpart_item:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = milp_solver.variable_decision["x"]
        lns_constraint = {}
        for key in x_var:
            start += [(x_var[key], dict_f_start[key])]
            if key in subpart_item:
                lns_constraint[key] = milp_solver.model.add_constr(x_var[key] == dict_f_start[key],
                                                                   name=str(key))
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        print(len(lns_constraint), " constraints")
        milp_solver.model.start = start
        return lns_constraint

    def remove_constraints_from_previous_iteration(self, milp_solver: LPKnapsack,
                                                   previous_constraints: Iterable[Any]):
        milp_solver.model.remove([previous_constraints[k] for k in previous_constraints])
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
