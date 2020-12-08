import networkx as nx
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from skdecide.builders.discrete_optimization.coloring.coloring_model import ColoringSolution, ColoringProblem
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from enum import Enum

strategies = ["largest_first",
              "random_sequential",
              "smallest_last",
              "independent_set",
              "connected_sequential_dfs",
              "connected_sequential_bfs",
              "connected_sequential",
              "saturation_largest_first",
              "DSATUR"]


class NXGreedyColoringMethod(Enum):
    largest_first = "largest_first"
    random_sequential = "random_sequential"
    smallest_last = "smallest_last"
    independent_set = "independent_set"
    connected_sequential_dfs = "connected_sequential_dfs"
    connected_sequential_bfs = "connected_sequential_bfs"
    connected_sequential = "connected_sequential"
    saturation_largest_first = "saturation_largest_first"
    dsatur = "DSATUR"
    best = "best"


class GreedyColoring(SolverDO):
    def __init__(self, color_problem: ColoringProblem,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.color_problem = color_problem
        self.nx_graph = self.color_problem.graph.to_networkx()
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.color_problem,
                                                       params_objective_function=params_objective_function)

    def solve(self, **kwargs):
        strategy: NXGreedyColoringMethod = kwargs.get("strategy", NXGreedyColoringMethod.best)
        print(strategy)
        verbose: bool = kwargs.get("verbose", False)
        strategy_name = strategy.name
        if strategy_name == "best":
            strategies_to_test = strategies
        else:
            strategies_to_test = [strategy_name]
        best_solution = None
        best_nb_color = float('inf')
        for strategy in strategies_to_test:
            try:
                colors = nx.algorithms.coloring.greedy_color(self.nx_graph,
                                                             strategy=strategy,
                                                             interchange=False)
                sorted_nodes = sorted(list(colors.keys()))
                number_colors = len(set(list(colors.values())))
                solution = [colors[i] for i in sorted_nodes]
                if verbose:
                    print(strategy, " : number colors : ", number_colors)
                if number_colors < best_nb_color:
                    best_solution = solution
                    best_nb_color = number_colors
            except Exception as e:
                print("Failed strategy : ", strategy, e)
                pass
        if verbose:
            print("best : ", best_nb_color)
        solution = ColoringSolution(self.color_problem,
                                    colors=best_solution,
                                    nb_color=None)
        solution = solution.to_reformated_solution() # TODO : make this OPTIONAL
        fit = self.aggreg_sol(solution)
        if verbose:
            print("Solution found : ", solution)
        return ResultStorage(list_solution_fits=[(solution, fit)],
                             best_solution=solution,
                             mode_optim=self.params_objective_function.sense_function)
