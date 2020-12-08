from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, ObjectiveRegister, EncodingRegister, \
    TypeAttribute, ModeOptim, ObjectiveHandling, TypeObjective
from typing import List, Union
import numpy as np
import math
import matplotlib.pyplot as plt


class TTPSolution(Solution):
    def __init__(self, tour: Union[List[int], np.array],
                 packing: Union[List[int], np.array],
                 **kwargs):
        self.tour = tour
        self.packing = packing
        self.time_trip = kwargs.get("time_trip", 0)
        self.distance = kwargs.get("distance", 0)
        self.profit = kwargs.get("profit", 0)
        self.weight_used = kwargs.get("weight_used", 0)
        self.weight_end = kwargs.get("weight_end", 0)
        self.objective = kwargs.get("objective", 0)
        self.weight_array = kwargs.get("weight_array", 0)
    
    def copy(self):
        return TTPSolution(np.copy(self.tour), np.copy(self.packing),
                           time_trip=self.time_trip, distance=self.distance,
                           profit=self.profit, weight_used=self.weight_used,
                           weight_end=self.weight_end, objective=self.objective,
                           weight_array=self.weight_array)
    
    def __str__(self):
        return "Profit :"+str(self.profit)+"\nWeight: "+str(self.weight_used)+\
            "\nTime Trip : "+str(self.time_trip)+"\nDistance : "+str(self.distance)+"\n"+\
                "Objective : "+str(self.objective)

    def change_problem(self, new_problem):
        raise NotImplementedError

def distance(point1, point2):
    return math.sqrt((point1["x"]-point2["x"])**2+(point1["y"]-point2["y"])**2)


class TTPModel(Problem):
    def satisfy(self, variable: Solution) -> bool:
        # TODO do it
        return True

    def get_attribute_register(self) -> EncodingRegister:
        dict_attribute = {"tour":
                              {"name": "tour",
                               "type": [TypeAttribute.PERMUTATION,
                                        TypeAttribute.PERMUTATION_TSP],  # TODO, untrue at the moment..
                               "range": range(self.numberOfNodes),
                               "n": self.numberOfNodes}}
        return EncodingRegister(dict_attribute)

    def get_solution_type(self):
        return TTPSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.AGGREGATE,
                                 dict_objective_to_doc={"obj":
                                                         {"type": TypeObjective.OBJECTIVE,
                                                          "default_weight": 1}})

    def __init__(self,
                 problemName = None,
                 knapsackDataType = None,
                 numberOfNodes = None,
                 numberOfItems = None,
                 capacityOfKnapsack = None,
                 minSpeed = None,
                 maxSpeed = None,
                 rentingRatio = None,
                 edgeWeightType = None,
                 nodes = None,
                 items = None, nodes_array=None, items_array=None):
        self.problemName = problemName
        self.knapsackDataType = knapsackDataType
        self.numberOfNodes = numberOfNodes
        self.numberOfItems = numberOfItems
        self.capacityOfKnapsack = capacityOfKnapsack
        self.minSpeed = minSpeed
        self.maxSpeed = maxSpeed
        self.rentingRatio = rentingRatio
        self.edgeWeightType = edgeWeightType
        self.nodes = nodes
        self.items = items
        self.weights = np.array([item["weight"] for item in self.items])
        self.profits = np.array([item["profit"] for item in self.items])
        self.av = np.zeros((self.numberOfItems, self.numberOfNodes))
        self.nodes_array = nodes_array
        self.items_array = items_array
        for i in range(len(self.items)):
            self.av[i, self.items[i]["node_index"]] = 1
        self.evaluate_function = build_obj_function(self)
        self.evaluate_function_details = build_obj_function_details(self)

    def evaluate(self, solution: TTPSolution):  
        objective, profit, distance_tour, time_trip, weight_used, weight_end  = self.evaluate_function(solution.tour, 
                                                                                                       solution.packing)
        solution.time_trip = time_trip
        solution.distance = distance_tour
        solution.profit = profit
        solution.weight_used = weight_used
        solution.weight_end = weight_end
        solution.objective = objective
        return {"obj": solution.objective}

    def evaluate_details(self, solution: TTPSolution):  
        objective, profit, distance_tour, time_trip, weight_used, weight_end, weight_array = \
            self.evaluate_function_details(solution.tour, solution.packing)
        solution.time_trip=time_trip
        solution.distance=distance_tour
        solution.profit=profit
        solution.weight_used=weight_used
        solution.weight_end=weight_end
        solution.objective=objective
        solution.weight_array=weight_array
        return solution.objective
    

    def plot(self, solution: TTPSolution, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        pp, = ax.plot([self.nodes_array[jj, 1] for jj in solution.tour]+[self.nodes_array[solution.tour[0], 1]], 
                      [self.nodes_array[jj, 2] for jj in solution.tour]+[self.nodes_array[solution.tour[0], 2]])
        return ax
    
        
from numba import njit
from functools import partial

def build_obj_function(ttp_model):
    return partial(evaluate, 
                   capacityOfKnapsack=ttp_model.capacityOfKnapsack,
                   rentingRatio=ttp_model.rentingRatio,
                   minSpeed=ttp_model.minSpeed, 
                   maxSpeed=ttp_model.maxSpeed,
                   items_array=ttp_model.items_array, 
                   nodes_array=ttp_model.nodes_array,
                   av=ttp_model.av)

def build_obj_function_details(ttp_model):
    return partial(evaluate_with_details, 
                   capacityOfKnapsack=ttp_model.capacityOfKnapsack,
                   rentingRatio=ttp_model.rentingRatio,
                   minSpeed=ttp_model.minSpeed, 
                   maxSpeed=ttp_model.maxSpeed,
                   items_array=ttp_model.items_array, 
                   nodes_array=ttp_model.nodes_array,
                   av=ttp_model.av)

@njit
def evaluate(tour, packing, capacityOfKnapsack, rentingRatio, 
             minSpeed, maxSpeed, items_array, nodes_array, av): 
    z = packing
    weightofKnapsack = capacityOfKnapsack
    rentRate = rentingRatio
    vmin = minSpeed
    vmax = maxSpeed
    if(tour[0]!=tour[len(tour)-1]):
        print("ERROR: The last city must be the same as the first city")
        return
    wc=0
    time_trip=0
    profit=0
    distance_tour=0
    for i in range(len(tour)-1):
        selectedItem = [j for j in range(len(z)) if z[j]==tour[i]]  
        currentcitytemp = tour[i]    
        currentcity = currentcitytemp-1       
        availabilityCounter = 0
        for p in selectedItem:
            if av[p, tour[i]]!=1:
                break
        if(len(selectedItem)>0):
            for item in selectedItem:
                wc=wc+items_array[item, 2]
                profit=profit+items_array[item, 1]
        distance_i = math.sqrt((nodes_array[tour[i], 1]-nodes_array[tour[i+1], 1])**2
                              +(nodes_array[tour[i], 2]-nodes_array[tour[i+1], 2])**2)
        distance_tour += distance_i
        time_trip=time_trip+(distance_i/(vmax-wc*(vmax-vmin)/weightofKnapsack))
    weight_used = wc
    weight_end = weightofKnapsack-wc
    objective = profit-time_trip*rentRate
    return objective, profit, distance_tour, time_trip, weight_used, weight_end
    


@njit
def evaluate_with_details(tour, packing, capacityOfKnapsack, rentingRatio, 
                          minSpeed, maxSpeed, items_array, nodes_array, av): 
    z = packing
    weightofKnapsack = capacityOfKnapsack
    rentRate = rentingRatio
    vmin = minSpeed
    vmax = maxSpeed
    if(tour[0]!=tour[len(tour)-1]):
        print("ERROR: The last city must be the same as the first city")
        return
    wc=0
    time_trip=0
    profit=0
    distance_tour=0
    lll = len(tour)-1
    weight_array = np.zeros((lll), dtype=np.int32)
    for i in range(lll):
        selectedItem = [j for j in range(len(z)) if z[j]==tour[i]]  
        currentcitytemp = tour[i]    
        currentcity = currentcitytemp-1       
        availabilityCounter = 0
        for p in selectedItem:
            if av[p, tour[i]]!=1:
                break
        if(len(selectedItem)>0):
            for item in selectedItem:
                wc=wc+items_array[item, 2]
                profit=profit+items_array[item, 1]
        weight_array[i] = wc
        distance_i = math.sqrt((nodes_array[tour[i], 1]-nodes_array[tour[i+1], 1])**2
                              +(nodes_array[tour[i], 2]-nodes_array[tour[i+1], 2])**2)
        distance_tour += distance_i
        time_trip=time_trip+(distance_i/(vmax-wc*(vmax-vmin)/weightofKnapsack))
    weight_used = wc
    weight_end = weightofKnapsack-wc
    objective = profit-time_trip*rentRate
    return objective, profit, distance_tour, time_trip, weight_used, weight_end, weight_array
    
