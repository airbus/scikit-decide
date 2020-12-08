import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../"))
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation, LocalMoveDefault, LocalMove
from typing import List, Union, Tuple, Dict
import numpy as np
from skdecide.builders.discrete_optimization.tsp.tsp_model import Point, Point2D, SolutionTSP, TSPModel, \
    TSPModel2D, length, SolutionTSP, compute_length
import random


def ccw(A: Point2D, B: Point2D, C: Point2D):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)


# Return true if line segments AB and CD intersect
def intersect(A: Point2D, B: Point2D, C: Point2D, D: Point2D):
    return ccw(A, C, D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def find_intersection(variable: SolutionTSP,
                      points: List[Point2D],
                      test_all=False, nb_tests=10):
    perm = variable.permutation
    intersects = []
    its = range(len(perm)) if test_all else random.sample(range(len(perm)), min(nb_tests, len(perm)))
    jts = range(len(perm)-1) if test_all else random.sample(range(len(perm)-1), min(nb_tests, len(perm)-1))
    for i in its:
        for j in jts:
            ii = i
            jj = j
            if jj <= ii+1:
                continue
            A, B = points[perm[ii]], points[perm[ii+1]]
            C, D = points[perm[jj]], points[perm[jj+1]]
            if intersect(A, B, C, D):
                intersects += [(ii+1, jj)]
                if len(intersects)>5:
                    break
        if len(intersects)>5:
            break
    return intersects


def get_points_index(it, jt, variable: SolutionTSP, length_permutation: int):
    perm = variable.permutation
    i = perm[it]
    j = perm[jt]
    if it == 0:
        i_before = variable.start_index
    else:
        i_before = perm[it-1]
    if jt == length_permutation-1:
        j_after = variable.end_index
    else:
        j_after = perm[jt+1]
    return i_before, i, j, j_after


class Mutation2Opt(Mutation):
    node_count: int
    points: List[Point]
    @staticmethod
    def build(problem: TSPModel2D, solution: SolutionTSP, **kwargs):
        return Mutation2Opt(problem, **kwargs)

    def __init__(self, 
                 tsp_model: TSPModel2D,
                 test_all=False, 
                 nb_test=None, 
                 return_only_improvement=False, **kwargs):
        self.node_count = tsp_model.node_count
        self.length_permutation = tsp_model.length_permutation
        self.points = tsp_model.list_points
        self.test_all = test_all
        self.nb_test = min(nb_test, self.node_count-1)
        self.evaluate_function_indexes = tsp_model.evaluate_function_indexes
        self.return_only_improvement = return_only_improvement
        self.tsp_model = tsp_model
        if self.nb_test is None:
            self.nb_test = max(1, self.node_count//10)
    
    def get_points(self, it, jt, variable: SolutionTSP):
        point_before_i = None
        perm = variable.permutation
        if it == 0:
            point_before_i = self.points[variable.start_index]
        else:
            point_before_i = self.points[perm[it-1]]
        point_i = self.points[perm[it]]
        point_j = self.points[perm[jt]]
        point_after_j = None
        if jt == self.length_permutation-1:
            point_after_j = self.points[variable.end_index]
        else:
            point_after_j = self.points[perm[jt+1]]
        return point_before_i, point_i, point_j, point_after_j
    
    def get_points_index(self, it, jt, variable: SolutionTSP):
        i_before = None
        j_after = None
        perm = variable.permutation
        i = perm[it]
        j = perm[jt]
        if it == 0:
            i_before = variable.start_index
        else:
            i_before = perm[it-1]
        if jt == self.length_permutation-1:
            j_after = variable.end_index
        else:
            j_after = perm[jt+1]
        return i_before, i, j, j_after

    def mutate_and_compute_obj(self, variable: SolutionTSP):
        it = random.randint(0, self.length_permutation-2)
        jt = random.randint(it+1, self.length_permutation-1)
        point_before_it, point_it, point_jt, point_after_jt = self.get_points(it, jt, variable)
        min_change = float('inf')
        range_its = range(self.length_permutation) if self.test_all \
            else random.sample(range(self.length_permutation), min(self.nb_test, self.length_permutation))
        for i in range_its:
            #print( min(self.nb_test, self.nodeCount-i-1))
            #print(i+1, self.nodeCount-1)
            if i == self.length_permutation-1:
                range_jts = []
            else:
                range_jts = range(i+1, self.length_permutation) if self.test_all \
                    else random.sample(range(i+1, 
                                             self.length_permutation), 
                                       min(1, self.nb_test, self.length_permutation-i-1))
            for j in range_jts:
                # print("i ", i, "j ", j, "i+1", i+1, "j+1", j+1)
                # print((i,j), (i+1, j+1), (i, i+1), (j, j+1))
                i_before, i_, j_, j_after = self.get_points_index(i, j, variable)
                change = self.evaluate_function_indexes(i_before, j_)-\
                         self.evaluate_function_indexes(i_before, i_)-\
                         self.evaluate_function_indexes(j_, j_after)+\
                         self.evaluate_function_indexes(i_, j_after)
                #change = length(point_before_i, point_j)-\
                #         length(point_before_i, point_i) - \
                #         length(point_j, point_after_j)+\
                #         length(point_i, point_after_j)
                if change < min_change:
                    it = i
                    jt = j
                    min_change = change
        fitness = variable.length+min_change
        i_before, i_, j_, j_after = self.get_points_index(it, jt, variable)
        permut = variable.permutation[:(it)]+variable.permutation[it:jt+1][::-1]+variable.permutation[jt+1:]
        lengths = []
        if it > 0:
            lengths += variable.lengths[:it]
        lengths += [self.evaluate_function_indexes(i_before, j_)]
        lengths += variable.lengths[it+1:jt+1][::-1]
        lengths += [self.evaluate_function_indexes(i_, j_after)]
        if jt < self.length_permutation-1:
            lengths += variable.lengths[jt+2:]
        if False:
            print(len(variable.lengths))
            print(len(lengths))
            print(variable.lengths)
            print(min_change)
            print("original perm ; ", variable.permutation)
            print("New : ", permut)
            print("original lengths ; ", variable.lengths)
            print("New : ", lengths)
            print("original , ", sum(variable.lengths), variable.length)
            print("New : ", sum(lengths), fitness)
            print(len(permut))
        if min_change < 0 or not self.return_only_improvement:
            v = SolutionTSP(start_index=variable.start_index,
                            end_index=variable.end_index,
                            permutation=permut,
                            lengths=lengths,
                            length=fitness,
                            problem=self.tsp_model)
            return v, LocalMoveDefault(variable, v), {"length": fitness}
        else:
            return variable, LocalMoveDefault(variable, variable), {"length": variable.length}

    def mutate(self, variable: SolutionTSP):
        v, move, f = self.mutate_and_compute_obj(variable)
        return v, move


class Mutation2OptIntersection(Mutation2Opt):
    nodeCount: int
    points: List[Point]
    @staticmethod
    def build(problem: TSPModel2D, solution: SolutionTSP, **kwargs):
        return Mutation2OptIntersection(problem, **kwargs)

    def __init__(self, 
                 tsp_model: TSPModel2D,
                 test_all=True, 
                 nb_test=None, 
                 return_only_improvement=False, 
                 i_j_pairs=None, **kwargs):
        Mutation2Opt.__init__(self, tsp_model, test_all, nb_test, return_only_improvement)
        self.tsp_model = tsp_model
        self.i_j_pairs = i_j_pairs
        if self.i_j_pairs is None:
            self.i_j_pairs = None

    def mutate_and_compute_obj(self, variable: SolutionTSP):
        reset_end = False
        if True:
            reset_end = True
            ints = find_intersection(variable, self.points, 
                                     nb_tests=min(3000, self.node_count-2))
            self.i_j_pairs = ints
            if len(self.i_j_pairs) == 0:
                return variable, LocalMoveDefault(variable, variable), variable.length
        min_change = float('inf')
        perm = variable.permutation
        for i, j in self.i_j_pairs:
            i_before, i_, j_, j_after = self.get_points_index(i, j, variable)
            change = self.evaluate_function_indexes(i_before, j_)-\
                         self.evaluate_function_indexes(i_before, i_)-\
                         self.evaluate_function_indexes(j_, j_after)+\
                         self.evaluate_function_indexes(i_, j_after)
            if change < min_change:
                it = i
                jt = j
                min_change = change
        fitness = variable.length+min_change
        i_before, i_, j_, j_after = self.get_points_index(it, jt, variable)
        permut = variable.permutation[:(it)]+variable.permutation[it:jt+1][::-1]+variable.permutation[jt+1:]
        lengths = []
        if it > 0:
            lengths += variable.lengths[:it]
        lengths += [self.evaluate_function_indexes(i_before, j_)]
        lengths += variable.lengths[it+1:jt+1][::-1]
        lengths += [self.evaluate_function_indexes(i_, j_after)]
        if False:
            print(len(variable.lengths))
            print(len(lengths))
            print(variable.lengths)
            print(min_change)
            print("original perm ; ", variable.permutation)
            print("New : ", permut)
            print("original lengths ; ", variable.lengths)
            print("New : ", lengths)
            print("original , ", sum(variable.lengths), variable.length)
            print("New : ", sum(lengths), fitness)
            print(len(permut))
        if reset_end:
            self.i_j_pairs = None
        if min_change < 0 or not self.return_only_improvement:
            v = SolutionTSP(start_index=variable.start_index,
                            end_index=variable.end_index,
                            permutation=permut,
                            lengths=lengths,
                            length=fitness,
                            problem=self.tsp_model)
            return v, LocalMoveDefault(variable, v), {"length": fitness}
        else:
            return variable, LocalMoveDefault(variable, variable), {"length": variable.length}


class SwapTSPMove(LocalMove):
    def __init__(self, 
                 attribute, 
                 tsp_model: TSPModel,
                 swap: Tuple[int, int]):
        self.attribute = attribute
        self.tsp_model = tsp_model
        self.swap = swap

    def apply_local_move(self, solution: SolutionTSP) -> SolutionTSP:
        current = getattr(solution, self.attribute)
        i1, i2 = self.swap
        v1, v2 = current[i1], current[i2]
        current[i1], current[i2] = v2, v1
        i_before, i, j, j_after = get_points_index(i1, i2, solution, self.tsp_model.length_permutation)
        previous = solution.lengths[i1], solution.lengths[i1+1], solution.lengths[i2], solution.lengths[i2+1]
        solution.lengths[i1] = self.tsp_model.evaluate_function_indexes(i_before, current[i1])
        solution.lengths[i1+1] = self.tsp_model.evaluate_function_indexes(current[i1], current[i1+1])
        solution.lengths[i2] = self.tsp_model.evaluate_function_indexes(current[i2-1], current[i2])
        solution.lengths[i2+1] = self.tsp_model.evaluate_function_indexes(current[i2], j_after)
        if False:
            print(i_before, current[i1])
            print(current[i1], current[i1+1])
            print(current[i2-1], current[i2])
            print(current[i2], j_after)
            print(solution.lengths[i1]+solution.lengths[i1+1]+solution.lengths[i2]+solution.lengths[i2+1]+\
                  -sum(previous))
        solution.length = solution.length+solution.lengths[i1]+solution.lengths[i1+1]+solution.lengths[i2]+solution.lengths[i2+1]+\
                          - sum(previous)
        return solution

    def backtrack_local_move(self, solution: SolutionTSP) -> SolutionTSP:
        return self.apply_local_move(solution)


class MutationSwapTSP(Mutation):
    @staticmethod
    def build(problem: TSPModel, solution: SolutionTSP, **kwargs):
        return MutationSwapTSP(problem)

    def __init__(self, tsp_model: TSPModel):
        self.tsp_model = tsp_model
        self.length_permutation = tsp_model.length_permutation

    def mutate(self, solution: SolutionTSP)->Tuple[SolutionTSP, LocalMove]:
        i = random.randint(0, self.length_permutation-2)
        j = random.randint(i+1, min(self.length_permutation-1, i+1+3))
        two_opt_move = SwapTSPMove("permutation", self.tsp_model, (i, j))
        new_sol = two_opt_move.apply_local_move(solution)
        return new_sol, two_opt_move
        
    def mutate_and_compute_obj(self, solution: SolutionTSP)->Tuple[SolutionTSP, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        return sol, move, {"length": sol.length}


if __name__ == "__main__":
    from tsp.tsp_parser import parse_file, get_data_available
    files = get_data_available()
    files = [f for f in files if 'tsp_51_1' in f]
    model = parse_file(files[0])
    mutation = Mutation2Opt(model, False, 100, False)
    mutation = MutationSwapTSP(model)
    solution = model.get_dummy_solution()
    print("Initial : ", solution.length)
    sol = mutation.mutate_and_compute_obj(solution)
    #print(sol[0].permutation)
    lengths, obj = compute_length(model.start_index, 
                                  model.end_index,
                                  sol[0].permutation, 
                                  model.list_points, 
                                  model.node_count,
                                  model.length_permutation)
    print(sol[0].lengths)
    print(sum(sol[0].lengths))
    print(lengths)
    print(len(sol[0].lengths))
    print(len(lengths))
    print(obj, sol[0].length)
    sol_back = sol[1].backtrack_local_move(sol[0])
    print(sol_back.length, "backtrack")
