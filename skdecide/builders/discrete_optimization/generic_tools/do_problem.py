from enum import Enum
from abc import abstractmethod
from typing import Dict, Any, List, Tuple
from skdecide.builders.discrete_optimization.generic_tools.result_storage.multiobj_utils import TupleFitness
import numpy as np


class TypeAttribute(Enum):
    LIST_INTEGER = 0
    LIST_BOOLEAN = 1
    PERMUTATION = 2
    PERMUTATION_TSP = 3
    PERMUTATION_RCPSP = 4
    SET_INTEGER = 5
    LIST_BOOLEAN_KNAP = 6
    LIST_INTEGER_SPECIFIC_ARRITY = 7
    SET_TUPLE_INTEGER = 8


class ModeOptim(Enum):
    MAXIMIZATION = 0
    MINIMIZATION = 1


class EncodingRegister:
    dict_attribute_to_type: Dict[str, Any]

    def __init__(self, dict_attribute_to_type: Dict[str, Any]):
        self.dict_attribute_to_type = dict_attribute_to_type
    
    def get_types(self):
        return [t for k in self.dict_attribute_to_type
                for t in self.dict_attribute_to_type[k]["type"]]

    def __str__(self):
        return "Encoding : "+str(self.dict_attribute_to_type)


class ObjectiveHandling(Enum):
    SINGLE = 0
    AGGREGATE = 1
    MULTI_OBJ = 2


class TypeObjective(Enum):
    OBJECTIVE = 0
    PENALTY = 1


class ObjectiveRegister:
    objective_sense: ModeOptim
    objective_handling: ObjectiveHandling
    dict_objective_to_doc: Dict[str, Any]

    def __init__(self,
                 objective_sense: ModeOptim,
                 objective_handling: ObjectiveHandling,
                 dict_objective_to_doc: Dict[str, Any]):
        self.objective_sense = objective_sense
        self.objective_handling = objective_handling
        self.dict_objective_to_doc = dict_objective_to_doc

    def get_list_objective_and_default_weight(self):
        d = [(k, self.dict_objective_to_doc[k]["default_weight"]) for k in self.dict_objective_to_doc]
        return [s[0] for s in d], [s[1] for s in d]

    def __str__(self):
        s = "Objective Register :\n"
        s += "Obj sense : "+str(self.objective_sense)+"\n"
        s += "Obj handling : "+str(self.objective_handling)+"\n"
        s += "detail : "+str(self.dict_objective_to_doc)
        return s


class Solution:
    @abstractmethod
    def copy(self):
        ...
    
    def lazy_copy(self):
        return self.copy()

    def get_attribute_register(self, problem)->EncodingRegister:
        return problem.get_attribute_register()

    @abstractmethod
    def change_problem(self, new_problem):
        ...


# TODO see if a repair function can be added to repair a solution, as a new class like mutation
class Problem:
    @abstractmethod
    def evaluate(self, variable: Solution)->Dict[str, float]:
        ...

    def evaluate_mobj(self, variable: Solution)->TupleFitness:
        # Default implementation of multiobjective.
        # you should probably custom this for your specific domain !
        obj_register = self.get_objective_register()
        keys = sorted(obj_register.dict_objective_to_doc.keys())
        dict_values = self.evaluate(variable)
        return TupleFitness(np.array([dict_values[k] for k in keys]), len(keys))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]):
        # output of evaluate(solution) typically
        keys = sorted(self.get_objective_register().dict_objective_to_doc.keys())
        return TupleFitness(np.array([dict_values[k] for k in keys]), len(keys))

    @abstractmethod
    def satisfy(self, variable: Solution)->bool:
        ...

    @abstractmethod
    def get_attribute_register(self)->EncodingRegister:
        ...

    @abstractmethod
    def get_solution_type(self):
        ...

    @abstractmethod
    def get_objective_register(self)->ObjectiveRegister:
        ...


class BaseMethodAggregating(Enum):
    MEAN = 0
    MEDIAN = 1
    PERCENTILE = 2
    PONDERATION = 3
    MIN = 4
    MAX = 5


class MethodAggregating:
    def __init__(self,
                 base_method_aggregating: BaseMethodAggregating,
                 percentile: float=90.,
                 ponderation: np.array=None):
        self.base_method_aggregating = base_method_aggregating
        self.percentile = percentile
        self.ponderation = ponderation


class RobustProblem(Problem):
    def __init__(self,
                 list_problem: List[Problem],
                 method_aggregating: MethodAggregating):
        self.list_problem = list_problem
        self.method_aggregating = method_aggregating
        self.nb_problem = len(self.list_problem)
        self.agg_vec = self.aggregate_vector()

    def aggregate_vector(self):
        if self.method_aggregating.base_method_aggregating == BaseMethodAggregating.MEAN:
            func = np.mean
        if self.method_aggregating.base_method_aggregating == BaseMethodAggregating.MEDIAN:
            func = np.median
        if self.method_aggregating.base_method_aggregating == BaseMethodAggregating.PERCENTILE:
            def func(x):
                return np.percentile(x, q=[self.method_aggregating.percentile])[0]
        if self.method_aggregating.base_method_aggregating == BaseMethodAggregating.PONDERATION:
            def func(x):
                return np.dot(x, self.method_aggregating.ponderation)
        if self.method_aggregating.base_method_aggregating == BaseMethodAggregating.MIN:
            func = np.min
        if self.method_aggregating.base_method_aggregating == BaseMethodAggregating.MAX:
            func = np.max
        return func

    def evaluate(self, variable: Solution) -> Dict[str, float]:
        fits = [self.list_problem[i].evaluate(variable) for i in range(self.nb_problem)]
        keys = fits[0].keys()
        aggreg = {}
        for k in keys:
            vals = np.array([fit[k] for fit in fits])
            aggreg[k] = self.agg_vec(vals)
        return aggreg

    def satisfy(self, variable: Solution) -> bool:
        return self.list_problem[0].satisfy(variable)

    def get_attribute_register(self) -> EncodingRegister:
        return self.list_problem[0].get_attribute_register()

    def get_solution_type(self):
        return self.list_problem[0].get_solution_type()

    def get_objective_register(self) -> ObjectiveRegister:
        return self.list_problem[0].get_objective_register()


class ParamsObjectiveFunction:
    objective_handling: ObjectiveHandling
    objectives: List[str]
    weights: List[float]
    sense_function: ModeOptim

    def __init__(self,
                 objective_handling: ObjectiveHandling,
                 objectives: List[str],
                 weights: List[float],
                 sense_function: ModeOptim):
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.weights = weights
        self.sense_function = sense_function

    def __str__(self):
        s = "Params objective function :  \n"
        s += "Sense : "+str(self.sense_function)+"\n"
        s += "Objective handling "+str(self.objective_handling)+"\n"
        s += "Objectives "+str(self.objectives)+'\n'
        s += "weights : "+str(self.weights)
        return s


def get_default_objective_setup(problem: Problem)->ParamsObjectiveFunction:
    register_objective = problem.get_objective_register()
    objs, weights = register_objective.get_list_objective_and_default_weight()
    sense = register_objective.objective_sense
    print(sense, register_objective.objective_handling, objs, weights)
    return ParamsObjectiveFunction(objective_handling=register_objective.objective_handling,
                                   objectives=objs,
                                   weights=weights,
                                   sense_function=sense)


def build_aggreg_function_and_params_objective(problem: Problem,
                                               params_objective_function: ParamsObjectiveFunction=None):
    if params_objective_function is None:
        params_objective_function = get_default_objective_setup(problem)
    eval_sol, eval_dict = build_evaluate_function_aggregated(problem=problem,
                                                             params_objective_function=params_objective_function)
    return eval_sol, eval_dict, params_objective_function


def build_evaluate_function_aggregated(problem: Problem,
                                       params_objective_function: ParamsObjectiveFunction=None):
    if params_objective_function is None:
        params_objective_function = get_default_objective_setup(problem)
    sense_problem = problem.get_objective_register().objective_sense
    sign = 1  # if sense_problem == params_objective_function.sense_function else -1
    objectives = params_objective_function.objectives
    weights = params_objective_function.weights
    objective_handling = params_objective_function.objective_handling
    eval = None
    eval_from_dict_values = None
    if objective_handling == ObjectiveHandling.AGGREGATE:
        length = len(objectives)

        def eval(solution: Solution):
            dict_values = problem.evaluate(solution)
            val = sum([dict_values[objectives[i]] * weights[i]
                       for i in range(length)])
            return sign*val

        def eval_from_dict_values(dict_values):
            val = sum([dict_values[objectives[i]] * weights[i]
                      for i in range(length)])
            return sign*val
    if objective_handling == ObjectiveHandling.SINGLE:
        length = len(objectives)

        def eval(solution: Solution):
            dict_values = problem.evaluate(solution)
            return sign*dict_values[objectives[0]]*weights[0]

        def eval_from_dict_values(dict_values):
            return sign*dict_values[objectives[0]] * weights[0]
    if objective_handling == ObjectiveHandling.MULTI_OBJ:
        length = len(objectives)

        def eval(solution: Solution):
            d = problem.evaluate(solution)
            return TupleFitness(np.array([weights[i]*d[objectives[i]] for i in range(length)]), length)*sign

        def eval_from_dict_values(dict_values):
            return TupleFitness(np.array([weights[i]*dict_values[objectives[i]] for i in range(length)]), length)*sign
    return eval, eval_from_dict_values


