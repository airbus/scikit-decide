import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, EncodingRegister, TypeAttribute, \
    ObjectiveRegister, TypeObjective, ObjectiveHandling, ModeOptim
from typing import List, Union, NamedTuple, Tuple, Dict
import numpy as np
from numba import njit
from functools import partial
import math
from abc import abstractmethod
from copy import deepcopy


class SolutionTSP(Solution):
    permutation_from0: Union[List[int], np.array]
    start_index: int
    end_index: int
    permutation: Union[List[int], np.array]
    lengths: List[float] # to store the details of length of the tsp if you want.
    length: float # to store the length of the tsp, in case your mutation computes it :)

    def __init__(self,
                 problem=None,
                 start_index=None,
                 end_index=None,
                 permutation=None,
                 lengths=None,
                 length=None,
                 permutation_from0=None):
        assert(permutation is not None or permutation_from0 is not None)
        # if permutation is not None and permutation_from0 is None:
        #     assert(start_index is not None and end_index is not None and lengths is not None and length is not None)
        self.start_index = start_index
        self.end_index = end_index
        self.permutation = permutation
        self.lengths = lengths
        self.length = length
        self.permutation_from0 = permutation_from0
        self.problem = problem
        if self.start_index is None:
            self.start_index = problem.start_index
        if self.end_index is None:
            self.end_index = problem.end_index
        # convert perm
        if self.permutation is None:
            self.permutation = self.problem.convert_perm_from0_to_original_perm(self.permutation_from0)
        if self.permutation_from0 is None:
            self.permutation_from0 = self.problem.convert_original_perm_to_perm_from0(self.permutation)
        # print('problem__:', problem)
        # print('permutation_from0__:', self.permutation_from0)
        # TODO: Think about moving this into another function (to prevent unecessary calls to evaluate()
        if self.length is None:
            self.problem.evaluate(self)

    def copy(self):
        return SolutionTSP(problem=self.problem,
                           start_index=self.start_index,
                           end_index=self.end_index,
                           permutation=list(self.permutation),
                           lengths=list(self.lengths),
                           length=self.length,
                           permutation_from0=deepcopy(self.permutation_from0))

    def lazy_copy(self):
        return SolutionTSP(problem=self.problem,
                           start_index=self.start_index,
                           end_index=self.end_index,
                           permutation=self.permutation,
                           lengths=self.lengths,
                           length=self.length,
                           permutation_from0=self.permutation_from0)

    def __str__(self):
        return "perm :"+str(self.permutation)+"\nobj="+str(self.length)

    def change_problem(self, new_problem):
        self.__init__(problem=new_problem,
                      start_index=self.start_index,
                      end_index=self.end_index,
                      permutation=list(self.permutation),
                      lengths=list(self.lengths),
                      length=self.length,
                      permutation_from0=deepcopy(self.permutation_from0))

class Point:
    ...


class TSPModel(Problem):
    list_points: List[Point]
    np_points: np.array
    node_count: int

    def __init__(self,
                 list_points: List[Point], 
                 node_count: int, 
                 start_index: int=0, 
                 end_index: int=0):
        self.list_points = list_points
        self.node_count = node_count
        self.start_index = start_index
        self.end_index = end_index
        if self.start_index is None:
            self.start_index = 0
        if self.end_index is None:
            self.end_index = 0
        self.ind_in_permutation = [i for i in range(self.node_count) if i != self.start_index and i != self.end_index]
        self.length_permutation = len(self.ind_in_permutation)
        # print('start_index: ', start_index)
        # print('end_index: ', end_index)
        self.original_indices_to_permutation_indices = [i for i in range(self.node_count)
                                                        if i != self.start_index and i != self.end_index]
        self.original_indices_to_permutation_indices_dict = {}
        counter = 0
        for i in range(self.node_count):
            if i != self.start_index and i != self.end_index:
                self.original_indices_to_permutation_indices_dict[i] = counter
                counter += 1

        # print('original_indices_to_permutation_indices: ', self.original_indices_to_permutation_indices)

    # for a given tsp kind of problem, you should provide a custom evaluate function, for now still abstract.
    @abstractmethod
    def evaluate_function(self, var_tsp: SolutionTSP):
        ...

    @abstractmethod
    def evaluate_function_indexes(self, index_1, index_2):
        ...

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'permutation_from0':
            tsp_sol = SolutionTSP(problem=self,
                                  start_index=self.start_index,
                                  end_index=self.end_index,
                                  permutation=self.convert_perm_from0_to_original_perm(int_vector))
        elif encoding_name == "permutation":
            tsp_sol = SolutionTSP(problem=self,
                                  start_index=self.start_index,
                                  end_index=self.end_index,
                                  permutation=int_vector)
        elif encoding_name == 'custom':
            kwargs = {encoding_name: int_vector, 'problem': self, 'start_index': self.start_index, 'end_index':self.end_index}
            tsp_sol = SolutionTSP(**kwargs)
        objectives = self.evaluate(tsp_sol)
        return objectives

    def evaluate(self, var_tsp: SolutionTSP)-> Dict[str, float]:
        lengths, obj = self.evaluate_function(var_tsp)
        var_tsp.length = obj
        var_tsp.lengths = lengths
        return {'length': obj}
        # return obj
    
    def satisfy(self, var_tsp: SolutionTSP)->bool:
        b = var_tsp.permutation[0] == self.start_index and var_tsp.permutation[-1] == self.end_index
        if not b:
            return False
    
    def get_dummy_solution(self):
        var = SolutionTSP(problem=self, start_index=self.start_index,
                          end_index=self.end_index,
                          permutation=list(self.ind_in_permutation),
                          permutation_from0=None,
                          lengths=None, length=None)
        self.evaluate(var)
        return var

    def __str__(self):
        return "TSP problem with number of nodes :  : "+str(self.node_count)

    def convert_perm_from0_to_original_perm(self, perm_from0):
        perm = [self.original_indices_to_permutation_indices[x] for x in perm_from0]
        return perm

    def convert_original_perm_to_perm_from0(self, perm):
        #print('mapping: ', self.original_indices_to_permutation_indices_dict)
        #print('original: ', perm)
        perm_from0 = [self.original_indices_to_permutation_indices_dict[i] for i in perm]
        return perm_from0

    def get_solution_type(self):
        return SolutionTSP

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {}
        dict_register["permutation_from0"] = {"name": "permutation_from0",
                                              "type": [TypeAttribute.PERMUTATION],
                                              "range": range(len(self.original_indices_to_permutation_indices)),
                                              "n": len(self.original_indices_to_permutation_indices)}
        dict_register["permutation"] = {"name": "permutation",
                                        "type": [TypeAttribute.PERMUTATION, TypeAttribute.PERMUTATION_TSP],
                                        "range": self.ind_in_permutation,
                                        "n": self.length_permutation}
        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"length": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}}
        return ObjectiveRegister(objective_sense=ModeOptim.MINIMIZATION,
                                 objective_handling=ObjectiveHandling.SINGLE,
                                 dict_objective_to_doc=dict_objective)


# One 
class Point2D(Point, NamedTuple):
    x: float
    y: float


class TSPModel2D(TSPModel):
    def __init__(self, list_points: List[Point2D], 
                 node_count: int, 
                 start_index: int=0, 
                 end_index: int=0,
                 use_numba=True):
        TSPModel.__init__(self, list_points, node_count,
                          start_index=start_index, 
                          end_index=end_index)
        self.np_points = np.zeros((node_count, 2))
        for i in range(self.node_count): 
            self.np_points[i, 0] = self.list_points[i].x
            self.np_points[i, 1] = self.list_points[i].y
        if use_numba:
            self.evaluate_function_2d = build_evaluate_function_np(self)
        else:
            self.evaluate_function_2d = build_evaluate_function(self)

    def evaluate_function(self, var_tsp: SolutionTSP):
        return self.evaluate_function_2d(solution=var_tsp.permutation)
    
    def evaluate_function_indexes(self, index_1, index_2)->float:
        return length(self.list_points[index_1], self.list_points[index_2])



def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def compute_length(start_index, 
                   end_index,
                   solution: List[int], 
                   list_points: List[Point2D], 
                   node_count: int,
                   length_permutation: int):
    obj = length(list_points[start_index], list_points[solution[0]])
    lengths = [obj]
    for index in range(0, length_permutation-1):
        ll = length(list_points[solution[index]], list_points[solution[index+1]])
        obj += ll
        lengths += [ll]
    lengths += [length(list_points[end_index], list_points[solution[-1]])]
    obj += lengths[-1]
    return lengths, obj

# More efficient implementation
@njit
def compute_length_np(start_index, 
                      end_index,
                      solution: Union[List[int], np.array], 
                      np_points,
                      node_count, 
                      length_permutation)->Tuple[Union[List[float], np.array], float]:
    obj = np.sqrt((np_points[start_index, 0]-np_points[solution[0], 0])**2+\
                  (np_points[start_index, 1]-np_points[solution[0], 1])**2)
    lengths = np.zeros((node_count))
    lengths[0] = obj
    pp = obj
    for index in range(0, length_permutation-1):
        ll = math.sqrt((np_points[solution[index], 0]-np_points[solution[index+1], 0])**2+\
                       (np_points[solution[index], 1]-np_points[solution[index+1], 1])**2)
        obj += ll
        lengths[index] = ll
    lengths[node_count-1] = np.sqrt((np_points[end_index, 0]-np_points[solution[-1], 0])**2+\
                                    (np_points[end_index, 1]-np_points[solution[-1], 1])**2)
    obj += lengths[node_count-1]
    return lengths, obj

def build_evaluate_function(tsp_model: TSPModel):
    return partial(compute_length,
                   start_index=tsp_model.start_index,
                   end_index=tsp_model.end_index, 
                   length_permutation=tsp_model.length_permutation,
                   list_points=tsp_model.list_points,
                   node_count=tsp_model.node_count)

def build_evaluate_function_np(tsp_model: TSPModel):
    return partial(compute_length_np,
                   start_index=tsp_model.start_index,
                   end_index=tsp_model.end_index, 
                   length_permutation=tsp_model.length_permutation,
                   np_points=tsp_model.np_points,
                   node_count=tsp_model.node_count)

