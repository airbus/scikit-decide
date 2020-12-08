import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation
from typing import Union, List
import numpy as np


class BasicPortfolioMutation(Mutation):
    def __init__(self, list_mutation: List[Mutation], weight_mutation: Union[List[float], np.array]):
        self.list_mutation = list_mutation
        self.weight_mutation = weight_mutation
        if isinstance(self.weight_mutation, list):
            self.weight_mutation = np.array(self.weight_mutation)
        self.weight_mutation = self.weight_mutation/np.sum(self.weight_mutation)
        self.index_np = np.array(range(len(self.list_mutation)), dtype=np.int)
        print(len(self.list_mutation), " mutation available")

    def mutate(self, solution: Solution):
        choice = np.random.choice(self.index_np, 
                                  size=1, 
                                  p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate(solution)

    def mutate_and_compute_obj(self, solution: Solution):
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate_and_compute_obj(solution)


class BasicPortfolioMutationTrack(Mutation):
    def __init__(self, list_mutation: List[Mutation], weight_mutation: Union[List[float], np.array]):
        self.list_mutation = list_mutation
        self.weight_mutation = weight_mutation
        if isinstance(self.weight_mutation, list):
            self.weight_mutation = np.array(self.weight_mutation)
        self.weight_mutation = self.weight_mutation/np.sum(self.weight_mutation)
        self.index_np = np.array(range(len(self.list_mutation)), dtype=np.int)

    def mutate(self, solution: Solution):
        choice = np.random.choice(self.index_np, 
                                  size=1, 
                                  p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate(solution)

    def mutate_and_compute_obj(self, solution: Solution):
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate_and_compute_obj(solution)
