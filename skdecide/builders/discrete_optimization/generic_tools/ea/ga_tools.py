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
from skdecide.builders.discrete_optimization.generic_tools.ea.ga import Ga, DeapCrossover, DeapMutation, DeapSelection, ObjectiveHandling


class ParametersGa:
    mutation: Union[Mutation, DeapMutation] = None
    crossover: DeapCrossover = None
    selection: DeapSelection = None
    encoding: str = None
    objective_handling: ObjectiveHandling = None
    objectives: Union[str, List[str]] = None
    objective_weights: List[float] = None
    pop_size: int = None
    max_evals: int = None
    mut_rate: float = None
    crossover_rate: float = None
    tournament_size: float = None
    deap_verbose: bool = False

    def __init__(self,
                 mutation,
                 crossover,
                 selection,
                 encoding,
                 objective_handling,
                 objectives,
                 objective_weights,
                 pop_size,
                 max_evals,
                 mut_rate,
                 crossover_rate,
                 tournament_size,
                 deap_verbose
                 ):
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.encoding = encoding
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose

    @staticmethod
    def default_rcpsp():
        return ParametersGa(mutation=DeapMutation.MUT_SHUFFLE_INDEXES,
                 crossover=DeapCrossover.CX_PARTIALY_MATCHED,
                 selection=DeapSelection.SEL_TOURNAMENT,
                 encoding='rcpsp_permutation',
                 objective_handling=ObjectiveHandling.AGGREGATE,
                 objectives=['makespan'],
                 objective_weights=[-1],
                 pop_size=100,
                 max_evals=10000,
                 mut_rate=0.1,
                 crossover_rate=0.9,
                 tournament_size=5,
                 deap_verbose=False)

class ParametersAltGa:
    mutations: List[Union[Mutation, DeapMutation]] = None
    crossovers: List[DeapCrossover] = None
    selection: DeapSelection = None
    encodings: List[str] = None
    objective_handling: ObjectiveHandling = None
    objectives: Union[str, List[str]] = None
    objective_weights: List[float] = None
    pop_size: int = None
    max_evals: int = None
    mut_rate: float = None
    crossover_rate: float = None
    tournament_size: float = None
    deap_verbose: bool = False
    sub_evals: List[int] = None

    def __init__(self,
                 mutations,
                 crossovers,
                 selection,
                 encodings,
                 objective_handling,
                 objectives,
                 objective_weights,
                 pop_size,
                 max_evals,
                 mut_rate,
                 crossover_rate,
                 tournament_size,
                 deap_verbose,
                 sub_evals
                 ):
        self.mutations = mutations
        self.crossovers = crossovers
        self.selection = selection
        self.encodings = encodings
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose
        self.sub_evals = sub_evals

    @staticmethod
    def default_mrcpsp():
        return ParametersAltGa(mutations=[DeapMutation.MUT_UNIFORM_INT, DeapMutation.MUT_SHUFFLE_INDEXES],
                 crossovers=[DeapCrossover.CX_ONE_POINT, DeapCrossover.CX_PARTIALY_MATCHED],
                 selection=DeapSelection.SEL_TOURNAMENT,
                 encodings=['rcpsp_modes_arrity_fix', 'rcpsp_permutation'],
                 objective_handling=ObjectiveHandling.AGGREGATE,
                 objectives=['makespan'],
                 objective_weights=[-1],
                 pop_size=100,
                 max_evals=10000,
                 mut_rate=0.1,
                 crossover_rate=0.9,
                 tournament_size=5,
                 deap_verbose=False,
                 sub_evals=[1000, 1000])

    @staticmethod
    def default_msrcpsp():
        return ParametersAltGa(mutations=[DeapMutation.MUT_UNIFORM_INT, DeapMutation.MUT_SHUFFLE_INDEXES, DeapMutation.MUT_SHUFFLE_INDEXES],
                               crossovers=[DeapCrossover.CX_ONE_POINT, DeapCrossover.CX_PARTIALY_MATCHED, DeapCrossover.CX_PARTIALY_MATCHED],
                               selection=DeapSelection.SEL_TOURNAMENT,
                               # encodings=['modes_arrity_fix', 'priority_list_task', 'priority_worker_per_task_perm'],
                               encodings=['modes_arrity_fix_from_0', 'priority_list_task', 'priority_worker_per_task_perm'],
                               objective_handling=ObjectiveHandling.AGGREGATE,
                               objectives=['makespan'],
                               objective_weights=[-1],
                               pop_size=100,
                               max_evals=10000,
                               mut_rate=0.1,
                               crossover_rate=0.9,
                               tournament_size=5,
                               deap_verbose=False,
                               sub_evals=[500,500,500])
