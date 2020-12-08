from typing import Iterable, Any, Union
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction
from skdecide.builders.discrete_optimization.generic_tools.lns_cp import ConstraintHandler, InitialSolution
from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolverName, CPSolver
from skdecide.builders.discrete_optimization.knapsack.knapsack_model import KnapsackModel, KnapsackSolution
from skdecide.builders.discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest, ResultStorage
from skdecide.builders.discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN, CPKnapsackMZN2
import random
from enum import Enum
import mip
from skdecide.builders.discrete_optimization.knapsack.solvers.knapsack_lns_solver import InitialKnapsackMethod, InitialKnapsackSolution


class ConstraintHandlerKnapsack(ConstraintHandler):
    def __init__(self, problem: KnapsackModel, fraction_to_fix: float=0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(self,
                                             cp_solver: CPKnapsackMZN2,
                                             child_instance,
                                             result_storage: ResultStorage) -> Iterable[Any]:
        subpart_item = set(random.sample(range(self.problem.nb_items),
                                         int(self.fraction_to_fix * self.problem.nb_items)))
        current_solution: KnapsackSolution = result_storage.get_best_solution_fit()[0]
        list_strings = []
        for item in subpart_item:
            list_strings += ["constraint taken[" + str(item + 1) + "] == "
                             + str(current_solution.list_taken[item]) + ";\n"]
            child_instance.add_string(list_strings[-1])
        return list_strings

    def remove_constraints_from_previous_iteration(self,
                                                   cp_solver: CPKnapsackMZN2,
                                                   child_instance,
                                                   previous_constraints: Iterable[Any]):
        pass
