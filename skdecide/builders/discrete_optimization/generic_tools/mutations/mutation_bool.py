import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../"))
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, EncodingRegister, TypeAttribute
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation, LocalMove
from skdecide.builders.discrete_optimization.generic_tools.mutations.mutation_util import get_attribute_for_type
from typing import Tuple, Union, List
import numpy as np
from copy import deepcopy
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import LocalMoveDefault
import random

class BitFlipMove(LocalMove):
    def __init__(self, attribute, list_index_flip: List[int]):
        self.attribute = attribute
        self.list_index_flip = list_index_flip
    
    def apply_local_move(self, solution: Solution) -> Solution:
        l = getattr(solution, self.attribute)
        for index in self.list_index_flip:
            l[index] = (1-l[index])
        return solution
    
    def backtrack_local_move(self, solution: Solution) -> Solution:
        return self.apply_local_move(solution)


class MutationBitFlip(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return MutationBitFlip(problem, **kwargs)

    def __init__(self, problem: Problem, attribute: str=None, probability_flip: float=0.1):
        self.problem = problem
        self.attribute = attribute
        self.probability_flip = probability_flip
        if attribute is None:
            attributes = get_attribute_for_type(self.problem, TypeAttribute.LIST_BOOLEAN)
            if len(attributes)>0:
                self.attribute = attributes[0]
        self.length = problem.get_attribute_register().dict_attribute_to_type[self.attribute]["n"]
    
    def mutate(self, solution: Solution):
        indexes = np.where(np.random.random(self.length)<=self.probability_flip)
        move = BitFlipMove(self.attribute, indexes[0])
        return move.apply_local_move(solution), move

    def mutate_and_compute_obj(self, solution: Solution):
        s, move = self.mutate(solution)
        f = self.problem.evaluate(s)
        return s, move, f