import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, TypeAttribute
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation, LocalMove
from skdecide.builders.discrete_optimization.generic_tools.mutations.mutation_util import get_attribute_for_type
from typing import Tuple, List, Dict
import numpy as np
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import LocalMoveDefault
from skdecide.builders.discrete_optimization.knapsack.knapsack_model import KnapsackModel, KnapsackSolution
import random


class SingleBitFlipMove(LocalMove):

    def __init__(self, i, problem):
        self.i = i
        self.problem = problem

    def apply_local_move(self, solution: KnapsackSolution) -> KnapsackSolution:
        new_list_taken = solution.list_taken.copy()
        new_list_taken[self.i] = abs(new_list_taken[self.i] - 1)
        new_solution = KnapsackSolution(problem=self.problem, list_taken=new_list_taken)
        return new_solution

    def backtrack_local_move(self, solution: KnapsackSolution) -> KnapsackSolution:
        return self.apply_local_move(solution)


class KnapsackMutationSingleBitFlip(Mutation):

    def __init__(self, problem: KnapsackModel):
        self.problem = problem

    def mutate(self, solution: KnapsackSolution):
        s, m, f = self.mutate_and_compute_obj(solution)
        return s, m

    def mutate_and_compute_obj(self, solution: KnapsackSolution):
        n = len(solution.list_taken)
        i = random.randint(0, n-1)
        move = SingleBitFlipMove(i, self.problem)
        new_sol = move.apply_local_move(solution)

        if solution.weight is not None:
            if new_sol.list_taken[i] == 0:
                new_weight = solution.weight - self.problem.list_items[i].weight
                new_value = solution.value - self.problem.list_items[i].value
            else:
                new_weight = solution.weight + self.problem.list_items[i].weight
                new_value = solution.value + self.problem.list_items[i].value
            obj = {'value': new_value,
                   'weight_violation': max(0, new_weight - self.problem.max_capacity)}
        else:
            obj = self.problem.evaluate(solution)
        return new_sol, move, obj

    @staticmethod
    def build(knapsack_model: KnapsackModel, solution: Solution):
        return KnapsackMutationSingleBitFlip(knapsack_model)


class BitFlipMoveKP(LocalMove):
    def __init__(self,
                 attribute,
                 problem: KnapsackModel,
                 list_index_flip: List[int]):
        self.attribute = attribute
        self.problem = problem
        self.list_index_flip = list_index_flip

    def apply_local_move(self, solution: KnapsackSolution) -> Solution:
        if solution.weight is None:
            self.problem.evaluate(solution)
        l = getattr(solution, self.attribute)
        weight = solution.weight
        value = solution.value
        for index in self.list_index_flip:
            l[index] = (1 - l[index])
            if l[index] == 0:
                weight -= self.problem.list_items[index].weight
                value -= self.problem.list_items[index].value
            else:
                weight += self.problem.list_items[index].weight
                value += self.problem.list_items[index].value
        solution.weight = weight
        solution.value = value
        return solution

    def backtrack_local_move(self, solution: KnapsackSolution) -> Solution:
        return self.apply_local_move(solution)


class MutationKnapsack(Mutation):
    @staticmethod
    def build(knapsack_model: KnapsackModel, solution: Solution):
        return MutationKnapsack(knapsack_model)

    def __init__(self, knapsack_model: KnapsackModel, attribute: str = None):
        self.knapsack_model = knapsack_model
        self.nb_items = knapsack_model.nb_items
        self.list_items = knapsack_model.list_items
        self.profit_per_capacity = [self.list_items[i].value / self.list_items[i].weight for i in range(self.nb_items)]
        self.sum = np.sum(self.profit_per_capacity)
        self.profit_per_capacity /= self.sum
        self.sorted_by_utility = np.argsort(self.profit_per_capacity)
        self.attribute = attribute
        if attribute is None:
            attributes = get_attribute_for_type(self.knapsack_model, TypeAttribute.LIST_BOOLEAN)
            if len(attributes) > 0:
                self.attribute = attributes[0]

    def switch_on(self, variable: KnapsackSolution, come_from_outside=False):
        not_used = [i for i in range(self.nb_items) if variable.list_taken[i] == 0
                    and variable.weight + self.list_items[i].weight <= self.knapsack_model.max_capacity]
        if len(not_used) > 0:
            proba = np.array([self.profit_per_capacity[i] for i in not_used])
            proba = proba / np.sum(proba)
            index = np.random.choice(not_used, size=1, p=proba)[0]
            move = BitFlipMoveKP(self.attribute, problem=self.knapsack_model, list_index_flip=[index])
            sol = move.apply_local_move(variable)
            return sol, move, {'value': sol.value,
                               'weight_violation': max(0, sol.weight - self.knapsack_model.max_capacity)}
        else:
            if come_from_outside:
                return variable, LocalMoveDefault(variable, variable), {'value': variable.value,
                                                                        'weight_violation': max(0,
                                                                                                variable.weight - self.knapsack_model.max_capacity)}
            return self.switch_off(variable, True)

    def switch_off(self, variable: KnapsackSolution, come_from_outside=False):
        used = [i for i in range(self.nb_items) if variable.list_taken[i] == 1]
        if len(used) > 0:
            proba = np.array([1 / self.profit_per_capacity[i] for i in used])
            proba = proba / np.sum(proba)
            index = np.random.choice(used, size=1, p=proba)[0]
            move = BitFlipMoveKP(self.attribute, problem=self.knapsack_model, list_index_flip=[index])
            sol = move.apply_local_move(variable)
            return sol, move, {'value': sol.value,
                               'weight_violation': max(0, sol.weight - self.knapsack_model.max_capacity)}
        else:
            if come_from_outside:
                return variable, LocalMoveDefault(variable, variable), {'value': variable.value,
                                                                        'weight_violation': max(0,
                                                                                                variable.weight - self.knapsack_model.max_capacity)}
            return self.switch_on(variable, come_from_outside=True)

    def mutate(self, variable: KnapsackSolution) -> KnapsackSolution:
        s, m, f = self.mutate_and_compute_obj(variable)
        return s, m

    def mutate_and_compute_obj(self, variable: KnapsackSolution) -> Tuple[KnapsackSolution, LocalMove, Dict[str, float]]:
        if variable.weight is None:
            self.knapsack_model.evaluate(variable)
        r = random.random()
        if r < 0.8:
            return self.switch_on(variable)
        else:
            return self.switch_off(variable)


