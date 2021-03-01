# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from skdecide.discrete_optimization.generic_tools.do_problem import Solution, Problem, \
    EncodingRegister, TypeAttribute
from skdecide.discrete_optimization.generic_tools.do_mutation import Mutation, LocalMove
from typing import Tuple, Union, List, Dict
import numpy as np
from copy import deepcopy
import random


class ShuffleMove(LocalMove):
    def __init__(self, attribute, 
                 new_permutation: Union[List[int], np.array], 
                 prev_permutation: Union[List[int], np.array]):
        self.attribute = attribute
        self.permutation = new_permutation
        self.prev_permutation = prev_permutation

    def apply_local_move(self, solution: Solution) -> Solution:
        setattr(solution, self.attribute, self.permutation)
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        setattr(solution, self.attribute, self.prev_permutation)
        return solution


class PermutationShuffleMutation(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return PermutationShuffleMutation(problem, solution)

    def __init__(self, 
                 problem: Problem, 
                 solution: Solution, attribute: str=None):
        self.problem = problem
        self.register: EncodingRegister = solution.get_attribute_register(problem)
        self.attribute = attribute
        if self.attribute is None:
            attributes = [k
                          for k in self.register.dict_attribute_to_type
                          for t in self.register.dict_attribute_to_type[k]["type"]
                          if t == TypeAttribute.PERMUTATION]
            if len(attributes) > 0:
                self.attribute = attributes[0]
        self.range_shuffle = self.register.dict_attribute_to_type[self.attribute]["range"]
        self.range_int = list(range(len(self.range_shuffle)))

    def mutate(self, solution: Solution)->Tuple[Solution, LocalMove]:
        previous = list(getattr(solution, self.attribute))
        random.shuffle(self.range_int)
        new = [previous[i] for i in self.range_int]
        sol = solution.lazy_copy()
        setattr(sol, self.attribute, new)
        return sol, ShuffleMove(self.attribute, new_permutation=new, prev_permutation=previous)

    def mutate_and_compute_obj(self, solution: Solution)->Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj


class PermutationPartialShuffleMutation(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return PermutationPartialShuffleMutation(problem, solution, attribute=kwargs.get("attribute", None),
                                                 proportion=kwargs.get("proportion", 0.3))

    def __init__(self, 
                 problem: Problem, 
                 solution: Solution,
                 attribute: str=None,
                 proportion: float=0.3):
        self.problem = problem
        self.register: EncodingRegister = solution.get_attribute_register(problem)
        self.attribute = attribute
        if attribute is None:
            print("none")
            attributes = [k
                          for k in self.register.dict_attribute_to_type
                          for t in self.register.dict_attribute_to_type[k]["type"]
                          if t == TypeAttribute.PERMUTATION]
            print(attributes)
            if len(attributes) > 0:
                self.attribute = attributes[0]
        self.range_shuffle = self.register.dict_attribute_to_type[self.attribute]["range"]
        self.n_to_move = int(proportion*len(self.range_shuffle))
        self.range_int = list(range(self.n_to_move))
        self.range_int_total = list(range(len(self.range_shuffle)))

    def mutate(self, solution: Solution)->Tuple[Solution, LocalMove]:
        previous = deepcopy(getattr(solution, self.attribute))
        random.shuffle(self.range_int_total)
        int_to_move = self.range_int_total[:self.n_to_move]
        random.shuffle(self.range_int)
        new = getattr(solution, self.attribute)
        for k in range(self.n_to_move):
            # prevs += [new[int_to_move[k]]]
            new[int_to_move[k]] = previous[int_to_move[self.range_int[k]]]
            # news += [new[int_to_move[k]]]
            # inds += [int_to_move[k]]
        sol = solution.lazy_copy()
        setattr(sol, self.attribute, new)
        return sol, ShuffleMove(self.attribute, new, previous)
        
    def mutate_and_compute_obj(self, solution: Solution)->Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj


class SwapsLocalMove(LocalMove):
    def __init__(self, attribute, list_index_swap: List[Tuple[int, int]]):
        self.attribute = attribute
        self.list_index_swap = list_index_swap

    def apply_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i1, i2 in self.list_index_swap:
            v1, v2 = current[i1], current[i2]
            current[i1], current[i2] = v2, v1
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i1, i2 in self.list_index_swap:
            v1, v2 = current[i1], current[i2]
            current[i1], current[i2] = v2, v1
        return solution


class PermutationSwap(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return PermutationSwap(problem, solution,
                               attribute=kwargs.get("attribute", None),
                               nb_swap=kwargs.get("nb_swap", 1))

    def __init__(self, 
                 problem: Problem, 
                 solution: Solution,
                 attribute: str=None,
                 nb_swap: int=1):
        self.problem = problem
        self.register: EncodingRegister = solution.get_attribute_register(problem)
        self.nb_swap = nb_swap
        self.attribute = attribute
        if self.attribute is None:
            attributes = [k
                          for k in self.register.dict_attribute_to_type
                          for t in self.register.dict_attribute_to_type[k]["type"]
                          if t == TypeAttribute.PERMUTATION]
            if len(attributes) > 0:
                self.attribute = attributes[0]
        self.length = len(self.register.dict_attribute_to_type[self.attribute]["range"])

    def mutate(self, solution: Solution)->Tuple[Solution, LocalMove]:
        swaps = np.random.randint(low=0, high=self.length-1, size=(self.nb_swap, 2))
        move = SwapsLocalMove(self.attribute, [(swaps[i, 0], swaps[i, 1]) for i in range(self.nb_swap)])
        next_sol = move.apply_local_move(solution)
        return next_sol, move

    def mutate_and_compute_obj(self, solution: Solution)->Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj


class TwoOptMove(LocalMove):
    def __init__(self, attribute, index_2opt: List[Tuple[int, int]]):
        self.attribute = attribute
        self.index_2opt = index_2opt

    def apply_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i, j in self.index_2opt:
            current = current[:i]+current[i:j+1][::-1]+current[j+1:]
        setattr(solution, self.attribute, current)
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i, j in self.index_2opt[::-1]:
            current = current[:i]+current[i:j+1][::-1]+current[j+1:]
        setattr(solution, self.attribute, current)
        return solution


class TwoOptMutation(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, 
              **kwargs):
        return TwoOptMutation(problem, solution, attribute=kwargs.get('attribute', None))

    def __init__(self, 
                 problem: Problem, 
                 solution: Solution, 
                 attribute: str=None):
        self.problem = problem
        self.register: EncodingRegister = solution.get_attribute_register(problem)
        self.attribute = attribute
        if self.attribute is None:
            attributes = [k 
                          for k in self.register.dict_attribute_to_type 
                          for t in self.register.dict_attribute_to_type[k]["type"]
                          if t == TypeAttribute.PERMUTATION]
            if len(attributes) > 0:
                self.attribute = attributes[0]
        # self.range_shuffle = self.register.dict_attribute_to_type[self.attribute]["range"]
        self.length = len(self.register.dict_attribute_to_type[self.attribute]["range"])

    def mutate(self, solution: Solution)->Tuple[Solution, LocalMove]:
        i = random.randint(0, self.length-2)
        j = random.randint(i+1, self.length-1)
        two_opt_move = TwoOptMove(self.attribute, [(i, j)])
        new_sol = two_opt_move.apply_local_move(solution)
        return new_sol, two_opt_move
        
    def mutate_and_compute_obj(self, solution: Solution)->Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj
