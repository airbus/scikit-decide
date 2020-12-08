from skdecide.builders.discrete_optimization.generic_tools.do_mutation import LocalMove
from skdecide.builders.discrete_optimization.generic_tools.mutations.permutation_mutations import PermutationShuffleMutation, \
PermutationPartialShuffleMutation, PermutationSwap, TwoOptMutation, Mutation, Problem, Solution
from typing import Union, Tuple, Dict
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution, RCPSPModel


class PermutationMutationRCPSP(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        other_mutation = kwargs.get('other_mutation', PermutationShuffleMutation)
        other_mutation = other_mutation.build(problem, solution, **kwargs)
        return PermutationMutationRCPSP(problem, solution, other_mutation=other_mutation)

    def __init__(self, problem: Problem,
                 solution: Solution,
                 other_mutation: Mutation):
        self.problem = problem
        self.solution = solution
        self.other_mutation = other_mutation

    def mutate(self, solution: RCPSPSolution) -> Tuple[Solution, LocalMove]:
        s, lm = self.other_mutation.mutate(solution)
        try:
            s.standardised_permutation = s.generate_permutation_from_schedule()
            s._schedule_to_recompute = True
        except:
            pass
        return s, lm

    def mutate_and_compute_obj(self, solution: Solution) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        s, lm, fit = self.other_mutation.mutate_and_compute_obj(solution)
        try:
            s._schedule_to_recompute = True
            s.standardised_permutation = s.generate_permutation_from_schedule()
        except:
            pass
        return s, lm, fit

