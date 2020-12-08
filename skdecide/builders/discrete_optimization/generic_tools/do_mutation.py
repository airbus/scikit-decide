from enum import Enum
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem
from abc import abstractmethod
from typing import Tuple, Dict


class LocalMove:
    @abstractmethod
    def apply_local_move(self, solution: Solution) -> Solution:
        ...
    @abstractmethod
    def backtrack_local_move(self, solution: Solution) -> Solution:
        ...


class LocalMoveDefault(LocalMove):
    """
    Not clever local move
    If you're lazy or don't have the choice,
    don't do in place modification of the previous solution, so that you can retrieve it directly.
    So the backward operator is then obvious.
    """
    def __init__(self, prev_solution: Solution, new_solution: Solution):
        self.prev_solution = prev_solution
        self.new_solution = new_solution

    def apply_local_move(self, solution: Solution) -> Solution:
        return self.new_solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        return self.prev_solution


class Mutation:
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return NotImplementedError("Please implement it !")
        
    @abstractmethod
    def mutate(self, solution: Solution)->Tuple[Solution, LocalMove]:
        ...

    @abstractmethod
    def mutate_and_compute_obj(self, solution: Solution)->Tuple[Solution, LocalMove, Dict[str, float]]:
        ...









