from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ObjectiveHandling, \
    ParamsObjectiveFunction, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution,\
    SingleModeRCPSPModel, MultiModeRCPSPModel, \
    RCPSPModelCalendar, PartialSolution
from typing import Dict
from skdecide.builders.discrete_optimization.generic_tools.ea.ga_tools import ParametersGa, ParametersAltGa
from skdecide.builders.discrete_optimization.generic_tools.ea.ga import Ga
from skdecide.builders.discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa


class GA_MSRCPSP_Solver(SolverDO):
    def __init__(self, rcpsp_model: MultiModeRCPSPModel,
                 params_objective_function: ParamsObjectiveFunction = None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)

    def solve(self, parameters_ga: ParametersAltGa = ParametersAltGa.default_msrcpsp(), **args):
        ga_solver = AlternatingGa(self.rcpsp_model,
                                  encodings=parameters_ga.encodings,
                                  objective_handling=parameters_ga.objective_handling,
                                  objectives=parameters_ga.objectives,
                                  objective_weights=parameters_ga.objective_weights,
                                  mutations=parameters_ga.mutations,
                                  crossovers=parameters_ga.crossovers,
                                  max_evals=parameters_ga.max_evals,
                                  sub_evals=parameters_ga.sub_evals)
        return ga_solver.solve()
