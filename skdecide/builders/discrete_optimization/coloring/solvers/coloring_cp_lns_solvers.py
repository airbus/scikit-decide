from typing import Iterable, Any

from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolver
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.lns_cp import ConstraintHandler, InitialSolution, PostProcessSolution
from skdecide.builders.discrete_optimization.coloring.coloring_model import ColoringProblem, ColoringSolution
from enum import Enum
import random
from skdecide.builders.discrete_optimization.coloring.solvers.greedy_coloring import GreedyColoring


class InitialColoringMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialColoring(InitialSolution):
    def __init__(self, problem: ColoringProblem,
                 initial_method: InitialColoringMethod,
                 params_objective_function: ParamsObjectiveFunction):
        self.problem = problem
        self.initial_method = initial_method
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function =\
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=params_objective_function)

    def get_starting_solution(self)->ResultStorage:
        if self.initial_method == InitialColoringMethod.DUMMY:
            sol = self.problem.get_dummy_solution()
            fit = self.aggreg_sol(sol)
            return ResultStorage(list_solution_fits=[(sol, fit)], best_solution=sol,
                                 mode_optim=self.params_objective_function.sense_function)
        else:
            solver = GreedyColoring(color_problem=self.problem,
                                    params_objective_function=self.params_objective_function)
            return solver.solve()


class ConstraintHandlerFixColorsCP(ConstraintHandler):
    def __init__(self, problem: ColoringProblem,
                 fraction_to_fix: float=0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def remove_constraints_from_previous_iteration(self, cp_solver: CPSolver, child_instance,
                                                   previous_constraints: Iterable[Any]):
        pass

    def adding_constraint_from_results_store(self,
                                             cp_solver: CPSolver,
                                             child_instance,
                                             result_storage: ResultStorage) -> Iterable[Any]:
        range_node = range(1, self.problem.number_of_nodes + 1)
        current_solution = result_storage.get_best_solution()
        subpart_color = set(random.sample(range_node, int(self.fraction_to_fix * self.problem.number_of_nodes)))
        dict_color = {i + 1: current_solution.colors[i] + 1 for i in range(self.problem.number_of_nodes)}
        current_nb_color = max(dict_color.values())
        for i in range_node:
            if i in subpart_color and dict_color[i] < current_nb_color:
                child_instance.add_string("constraint color_graph[" + str(i) + "] == " + str(dict_color[i]) + ";\n")
            child_instance.add_string("constraint color_graph[" + str(i) + "] <= " + str(current_nb_color) + ";\n")


class PostProcessSolutionColoring(PostProcessSolution):
    def __init__(self, problem: ColoringProblem,
                 params_objective_function: ParamsObjectiveFunction):
        self.problem = problem
        self.params_objective_function = params_objective_function
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=
                                                       self.params_objective_function)

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        bs = result_storage.get_best_solution()
        colors = bs.colors
        set_colors = sorted(set(colors))
        nb_colors = len(set(set_colors))
        new_color_dict = {set_colors[i]: i for i in range(nb_colors)}
        new_solution = ColoringSolution(problem=self.problem,
                                        colors=[new_color_dict[colors[i]] for i in range(len(colors))])
        fit = self.aggreg_from_sol(new_solution)
        result_storage.add_solution(new_solution, fit)
        return result_storage

