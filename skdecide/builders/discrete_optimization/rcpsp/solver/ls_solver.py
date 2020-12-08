from typing import Union

from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialMethodRCPSP

from skdecide.builders.discrete_optimization.generic_tools.ls.hill_climber import HillClimber

from skdecide.builders.discrete_optimization.generic_tools.ls.simulated_annealing import TemperatureSchedulingFactor, SimulatedAnnealing

from skdecide.builders.discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit, ModeMutation

from skdecide.builders.discrete_optimization.generic_tools.mutations.mixed_mutation import BasicPortfolioMutation

from skdecide.builders.discrete_optimization.rcpsp.mutations.mutation_rcpsp import PermutationMutationRCPSP

from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.mutations.mutation_catalog import get_available_mutations

from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
import numpy as np
from enum import Enum

from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel
from skdecide.builders.discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import InitialSolutionMS_RCPSP


class LS_SOLVER(Enum):
    SA = 0
    HC = 1


class LS_RCPSP_Solver(SolverDO):
    def __init__(self, model: Union[RCPSPModel, MS_RCPSPModel],
                 params_objective_function: ParamsObjectiveFunction=None,
                 ls_solver: LS_SOLVER=LS_SOLVER.SA, **args):
        self.model = model
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.model,
                                                       params_objective_function=
                                                       params_objective_function)
        self.ls_solver = ls_solver

    def solve(self, **kwargs):
        model = self.model
        dummy = model.get_dummy_solution()
        if isinstance(model, MS_RCPSPModel):
            init = InitialSolutionMS_RCPSP(problem=self.model,
                                           initial_method=InitialMethodRCPSP.PILE_CALENDAR,
                                           params_objective_function=self.params_objective_function)
            sol = init.get_starting_solution()
            dummy = sol.get_best_solution()
        _, mutations = get_available_mutations(model, dummy)
        print(mutations)
        list_mutation = [mutate[0].build(model,
                                         dummy,
                                         **mutate[1]) for mutate in mutations
                         if mutate[0] == PermutationMutationRCPSP]
        #  and mutate[1]["other_mutation"] == TwoOptMutation]
        mixed_mutation = BasicPortfolioMutation(list_mutation,
                                                np.ones((len(list_mutation))))
        res = RestartHandlerLimit(200,
                                  cur_solution=dummy,
                                  cur_objective=model.evaluate(dummy))
        if self.ls_solver == LS_SOLVER.SA:
            ls = SimulatedAnnealing(evaluator=model,
                                    mutator=mixed_mutation,
                                    restart_handler=res,
                                    temperature_handler=TemperatureSchedulingFactor(temperature=2,
                                                                                    restart_handler=res,
                                                                                    coefficient=0.9999),
                                    mode_mutation=ModeMutation.MUTATE,
                                    params_objective_function=self.params_objective_function,
                                    store_solution=True,
                                    nb_solutions=10000)
        elif self.ls_solver == LS_SOLVER.HC:
            ls = HillClimber(evaluator=model,
                             mutator=mixed_mutation,
                             restart_handler=res,
                             mode_mutation=ModeMutation.MUTATE,
                             params_objective_function=self.params_objective_function,
                             store_solution=True,
                             nb_solutions=10000)

        result_sa = ls.solve(dummy,
                             nb_iteration_max=kwargs.get("nb_iteration_max", 2000),
                             pickle_result=False)
        return result_sa
