import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../"))
from skdecide.builders.discrete_optimization.ttp.ttp_model import TTPModel, TTPSolution
from skdecide.builders.discrete_optimization.generic_tools.ls.local_search import Variables, Mutation
from typing import Tuple
import numpy as np
import random

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def find_intersection(variable: TTPSolution,
                      model: TTPModel, 
                      test_all=False, nb_tests=10):
    #random.shuffle(pps)
    perm = variable.tour
    intersects = []
    its = range(1, model.numberOfNodes) if test_all else random.sample(range(1, model.numberOfNodes),
                                                                       nb_tests)
    jts = range(1, model.numberOfNodes) if test_all else random.sample(range(1, model.numberOfNodes), nb_tests)
    for i in its:
        for j in jts:
            ii = i
            jj = j
            if jj <= ii+1:
                continue
            A, B = model.nodes_array[perm[ii], 1:], model.nodes_array[perm[ii+1], 1:]
            C, D = model.nodes_array[perm[jj], 1:], model.nodes_array[perm[jj+1], 1:]
            if intersect(A, B, C, D):
                intersects += [(ii+1, jj)]
                if len(intersects)>5:
                    break
        if len(intersects)>5:
            break
    return intersects


class MutationTTPKnapsack(Mutation):
    def __init__(self, ttp_model: TTPModel):
        self.ttp_model = ttp_model
        self.items_per_city = {}
        self.item_to_city = {}
        items = self.ttp_model.items
        for i in range(len(items)):
            node = items[i]["node_index"] 
            if node not in self.items_per_city:
                self.items_per_city[node] = set()
            self.items_per_city[node].add(i)
            self.item_to_city[i] = node
        self.profit_per_capacity = self.ttp_model.profits/self.ttp_model.weights
        self.sum = np.sum(self.profit_per_capacity)
        self.profit_per_capacity /= self.sum
        self.sorted_by_utility = np.argsort(self.profit_per_capacity)
    
    def switch_on(self, variable: TTPSolution):
        not_used = [i for i in range(self.ttp_model.numberOfItems) if variable.packing[i]==-1
                    and variable.weight_used+self.ttp_model.weights[i]<=self.ttp_model.capacityOfKnapsack]
        if len(not_used)>0:
            proba = np.array([self.profit_per_capacity[i] for i in not_used])
            proba = proba/np.sum(proba)
            index = np.random.choice(not_used, size=1, p=proba)[0]
            var = TTPSolution(variable.tour, np.copy(variable.packing))
            var.packing[index] = self.item_to_city[index]
            return var
        else:
            return self.switch_off(variable)

    def switch_off(self, variable: TTPSolution):
        used = [i for i in range(self.ttp_model.numberOfItems) if variable.packing[i]!=-1]
        if len(used)>0:
            proba = np.array([1/self.profit_per_capacity[i] for i in used])
            proba = proba/np.sum(proba)
            index = np.random.choice(used, size=1, p=proba)[0]
            var = TTPSolution(variable.tour, np.copy(variable.packing))
            var.packing[index] = -1
            return var
        else:
            return self.switch_on(variable)

    def mutate(self, variable: TTPSolution)->Variables:
        r = random.random()
        if r<0.6:
            return self.switch_on(variable)
        else:
            return self.switch_off(variable)
        
    def mutate_and_compute_obj(self, variable: Variables)->Tuple[Variables, float]:
        ...

class MutationTTPKnapsackBest(Mutation):
    def __init__(self, ttp_model: TTPModel):
        self.ttp_model = ttp_model
        self.items_per_city = {}
        self.item_to_city = {}
        items = self.ttp_model.items
        for i in range(len(items)):
            node = items[i]["node_index"] 
            if node not in self.items_per_city:
                self.items_per_city[node] = set()
            self.items_per_city[node].add(i)
            self.item_to_city[i] = node
        self.profit_per_capacity = self.ttp_model.profits/self.ttp_model.weights
        self.sum = np.sum(self.profit_per_capacity)
        self.profit_per_capacity /= self.sum
        self.sorted_by_utility = np.argsort(self.profit_per_capacity)
    
    def switch_on(self, variable: TTPSolution):
        not_used = [i for i in range(self.ttp_model.numberOfItems) if variable.packing[i]==-1
                    and variable.weight_used+self.ttp_model.weights[i]<=self.ttp_model.capacityOfKnapsack]
        if len(not_used)>0:
            proba = np.array([self.profit_per_capacity[i] for i in not_used])
            proba = proba/np.sum(proba)
            cur_objective_value = variable.objective
            index = np.random.choice(not_used, size=50, p=proba)
            index_best = None
            best_value = cur_objective_value
            for i in index:
                var = TTPSolution(variable.tour, variable.packing)
                var.packing[i] = self.item_to_city[i]
                value = self.ttp_model.evaluate(var)

                if value >= best_value:
                    index_best = i
                    best_value = value
                var.packing[i] = -1
            if index_best is not None:
                var = TTPSolution(variable.tour, np.copy(variable.packing))
                var.packing[index_best] = self.item_to_city[index_best]
            else:
                var = variable
            return var
        else:
            return variable
            #print("No notused "+str(variable.weight_used))
            #return self.switch_off(variable)

    def switch_off(self, variable: TTPSolution):
        used = [i for i in range(self.ttp_model.numberOfItems) if variable.packing[i]!=-1]
        if len(used)>0:
            proba = np.array([1/self.profit_per_capacity[i] for i in used])
            proba = proba/np.sum(proba)
            # index = np.random.choice(used, size=1, p=proba)[0]
            index = np.random.choice(used, size=50, p=proba)
            index_best = None
            best_value = -float("inf")
            for i in index:
                var = TTPSolution(variable.tour, variable.packing)
                prev = var.packing[i]
                var.packing[i] = -1
                value = self.ttp_model.evaluate(var)
                if value >= best_value:
                    index_best = i
                    best_value = value
                var.packing[i] = prev
            if index_best is not None:
                var = TTPSolution(variable.tour, np.copy(variable.packing))
                var.packing[index_best] = -1
                self.ttp_model.evaluate(var)
                var = self.switch_on(var)
            else:
                var = variable
            return var
        else:
            return variable
            #print("No used "+str(variable.weight_used))
            #return self.switch_on(variable)

    def mutate(self, variable: TTPSolution)->Variables:
        r = random.random()
        if r<0.6:
            return self.switch_on(variable)
        else:
            return self.switch_off(variable)
        
    def mutate_and_compute_obj(self, variable: Variables)->Tuple[Variables, float]:
        ...


class Mutation2Opt(Mutation):
    def __init__(self, ttp_model: TTPModel, test_all: bool=False,
                 nb_test: int=1000, return_only_improvement: bool=False):
        self.ttp_model = ttp_model
        self.items_per_city = {}
        self.item_to_city = {}
        items = self.ttp_model.items
        for i in range(len(items)):
            node = items[i]["node_index"] 
            if node not in self.items_per_city:
                self.items_per_city[node] = set()
            self.items_per_city[node].add(i)
            self.item_to_city[i] = node
        self.profit_per_capacity = self.ttp_model.profits/self.ttp_model.weights
        self.sum = np.sum(self.profit_per_capacity)
        self.profit_per_capacity /= self.sum
        self.sorted_by_utility = np.argsort(self.profit_per_capacity)
        self.nb_node = self.ttp_model.numberOfNodes
        self.test_all = test_all
        self.nb_test = nb_test
        self.return_only_improvement = return_only_improvement

    def compute_i_j(self, variable: TTPSolution):
        r = random.random()
        if r < 0.1:
            ints = find_intersection(variable, self.ttp_model, 
                                     nb_tests=min(100, self.nb_node-1))
        else:
            ints = []
            range_its = range(1, self.nb_node) if self.test_all else random.sample(range(1, self.nb_node), self.nb_test)
            for i in range_its:
                if i == self.nb_node-1:
                    range_jts = []
                else:
                    range_jts = range(i+1, self.nb_node-1) if self.test_all else random.sample(range(i+1, 
                                                                                            self.nb_node-1), 
                                                                                min(1, self.nb_test, self.nb_node-i-2))
                for j in range_jts:
                    ints += [(i, j)]
        return ints


    def mutate_and_compute_obj(self, variable: TTPSolution):
        min_change = -float('inf')
        current_objective = variable.objective
        cur_var = None
        range_its = range(1, self.nb_node) if self.test_all else random.sample(range(1, self.nb_node), self.nb_test)
        ints = self.compute_i_j(variable)
        for i, j in ints:
            permut = np.concatenate((variable.tour[:i], variable.tour[i:j+1][::-1], variable.tour[j+1:]))
            sol = TTPSolution(permut, variable.packing)
            value = self.ttp_model.evaluate(sol)
            change = value-current_objective
            if change > min_change:
                it = i
                jt = j
                min_change = change
                cur_var = sol
        if min_change>0 or not self.return_only_improvement:
            if cur_var is not None:
                return cur_var, cur_var.objective
            else:
                return variable, variable.objective
        else:
            return variable, variable.objective

    def mutate(self, variable: TTPSolution):
        return self.mutate_and_compute_obj(variable)[0]

class MixingMutation(Mutation):
    def __init__(self,
                 ttp_model: TTPModel, 
                 nb_knapsack_iter: int,
                 nb_tsp_iter: int,
                 test_all: bool=False,
                 nb_test: int=1000, 
                 return_only_improvement: bool=False):
        self.mutation_knap = MutationTTPKnapsackBest(ttp_model)
        self.mutation_tsp = Mutation2Opt(ttp_model, test_all, nb_test, return_only_improvement)
        self.nb_knapsack_iter = nb_knapsack_iter
        self.nb_tsp_iter = nb_tsp_iter
        self.modes = ["tsp", "knap"]
        self.limit_count = {"tsp": nb_tsp_iter, 
                            "knap": nb_knapsack_iter}
        self.current_mode = "knap"
        self.current_iter = 0
    
    def mutate(self, variable: TTPSolution):
        r = random.random()
        if r<=0.9:
            # follow current trend.
            if self.current_mode == "tsp": 
                var = self.mutation_tsp.mutate(variable)
                self.current_iter += 1
                if self.current_iter >= self.limit_count["tsp"]:
                    self.current_iter = 0
                    self.current_mode = "knap"
                return var
            if self.current_mode == "knap": 
                var = self.mutation_knap.mutate(variable)
                self.current_iter += 1
                if self.current_iter >= self.limit_count["knap"]:
                    self.current_iter = 0
                    self.current_mode = "tsp"
                return var
        else:

            method = random.choice([self.mutation_knap, self.mutation_tsp])
            return method.mutate(variable)            

    

