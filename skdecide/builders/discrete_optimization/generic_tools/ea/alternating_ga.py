from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, EncodingRegister, TypeAttribute, ObjectiveHandling
from deap import creator, base, tools, algorithms
import random
from typing import Union, Optional, Any, Dict, List
import numpy as np
from enum import Enum
from skdecide.builders.discrete_optimization.generic_tools.ea.deap_wrappers import generic_mutate_wrapper
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ParamsObjectiveFunction, \
    ModeOptim, get_default_objective_setup, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.ea.ga import DeapMutation, DeapSelection, DeapCrossover, Ga


class AlternatingGa():
    """Multi-encoding single objective GA

        Args:
            problem:
                the problem to solve
            encoding:
                name (str) of an encoding registered in the register solution of Problem
                or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
                 to the dimension of the problem in this encoding (e.g. length of the vector)
                by default, the first encoding in the problem register_solution will be used.

    """
    def __init__(self, problem: Problem,
                 encodings: Union[List[str], List[Dict[str, Any]]] = None,
                 mutations:  Optional[Union[List[Mutation], List[DeapMutation]]] = None,
                 crossovers: Optional[List[DeapCrossover]] = None,
                 selections: Optional[List[DeapSelection]] = None,
                 objective_handling: Optional[ObjectiveHandling] = None,
                 objectives: Optional[Union[str,List[str]]] = None,
                 objective_weights: Optional[List[float]] = None,
                 pop_size: int = None,
                 max_evals: int = None,
                 sub_evals: List[int] = None,
                 mut_rate: float = None,
                 crossover_rate: float = None,
                 tournament_size: float = None,
                 deap_verbose: bool = None
                 ):
        self.problem = problem
        self.encodings = encodings
        self.mutations = mutations
        self.crossovers = crossovers
        self.selections = selections
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.sub_evals = sub_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose

        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                   params_objective_function=
                                                   None)
    def solve(self, **kwargs):
        # Initialise the population (here at random)

        count_evals = 0
        current_encoding_index = 0

        for i in range(len(self.encodings)):
            self.problem.set_fixed_attributes(self.encodings[i], self.problem.get_dummy_solution())

        while count_evals < self.max_evals:

            ga_solver = Ga(problem=self.problem,
                           encoding=self.encodings[current_encoding_index],
                           objective_handling=self.objective_handling,
                           objectives=self.objectives,
                           objective_weights=self.objective_weights,
                           mutation=self.mutations[current_encoding_index],
                           max_evals=self.sub_evals[current_encoding_index])
            tmp_sol = ga_solver.solve().get_best_solution()
            count_evals += self.sub_evals[current_encoding_index]

            # TODO: implement function below (1 in rcpsp domains, 1 in rcpsp solutions)
            self.problem.set_fixed_attributes(self.encodings[current_encoding_index], tmp_sol)

        problem_sol = tmp_sol

        result_storage = ResultStorage(list_solution_fits=[(problem_sol,
                                                            self.aggreg_from_sol(problem_sol))],
                                       best_solution=problem_sol,
                                       mode_optim=self.params_objective_function.sense_function)
        return result_storage
