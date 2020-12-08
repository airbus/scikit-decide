from enum import Enum
import numpy as np
from typing import List, Dict, NamedTuple
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, Solution, EncodingRegister, \
    TypeAttribute, ObjectiveRegister, TypeObjective, ObjectiveHandling, TupleFitness, ModeOptim


class Item(NamedTuple):
    index: int
    value: float
    weight: float

    def __str__(self):
        return "ind: "+str(self.index)+" weight: "+str(self.weight)+" value: "+str(self.value)


class KnapsackSolution(Solution):
    value: float
    weight: float
    list_taken: List[bool]

    def __init__(self, problem, list_taken, value=None, weight=None):
        self.problem = problem
        self.value = value
        self.weight = weight
        self.list_taken = list_taken

    def copy(self):
        return KnapsackSolution(problem=self.problem, 
                                value=self.value, 
                                weight=self.weight, 
                                list_taken=list(self.list_taken))

    def lazy_copy(self):
        return KnapsackSolution(problem=self.problem, 
                                value=self.value, weight=self.weight, 
                                list_taken=self.list_taken)

    def change_problem(self, new_problem):
        self.__init__(problem=new_problem,
                      value=self.value,
                      weight=self.weight,
                      list_taken=list(self.list_taken))

    def __str__(self):
        s = "Value="+str(self.value)+"\n"
        s += "Weight="+str(self.weight)+"\n"
        s += "Taken : "+str(self.list_taken)
        return s

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.list_taken == other.list_taken

#TODO : example : pour le tsp pour l'evaluate il faut les matrices de distance etc. et ta solution.
#fonction objectif : distance point adjacent dans la permut. 
#il faut s'assurer que le modele minizinc soit créé avec les mm objets.


class KnapsackModel(Problem):
    def __init__(self,
                 list_items: List[Item], 
                 max_capacity: float,
                 force_recompute_values: bool=False):
        self.list_items = list_items
        self.nb_items = len(list_items)
        self.max_capacity = max_capacity
        self.index_to_item = {list_items[i].index: list_items[i] for i in range(self.nb_items)}
        self.index_to_index_list = {list_items[i].index: i for i in range(self.nb_items)}
        self.force_recompute_values = force_recompute_values

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {}
        dict_register["list_taken"] = {"name": "list_taken",
                                       "type": [TypeAttribute.LIST_BOOLEAN,
                                                TypeAttribute.LIST_BOOLEAN_KNAP],
                                       "n": self.nb_items}
        return EncodingRegister(dict_register)


    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"weight_violation": {"type": TypeObjective.PENALTY,
                                               "default_weight": -100000},
                          "value": {"type": TypeObjective.OBJECTIVE,
                                    "default_weight": 1}}
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.AGGREGATE,
                                 dict_objective_to_doc=dict_objective)

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'list_taken':
            kp_sol = KnapsackSolution(problem=self, list_taken=int_vector)
        elif encoding_name == 'custom':
            kwargs = {encoding_name: int_vector, 'problem': self}
            kp_sol = KnapsackSolution(**kwargs)
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate(self, knapsack_solution: KnapsackSolution):
        if knapsack_solution.value is None or self.force_recompute_values:
            val = self.evaluate_value(knapsack_solution)
        else:
            val = knapsack_solution.value
        w_violation = self.evaluate_weight_violation(knapsack_solution)
        return {'value': val, 'weight_violation': w_violation}

    def evaluate_value(self, knapsack_solution: KnapsackSolution):
        s = sum([knapsack_solution.list_taken[i]*self.list_items[i].value
                for i in range(self.nb_items)])
        w = sum([knapsack_solution.list_taken[i]*self.list_items[i].weight
                for i in range(self.nb_items)])
        knapsack_solution.value = s
        knapsack_solution.weight = w
        return sum([knapsack_solution.list_taken[i]*self.list_items[i].value
                   for i in range(self.nb_items)])

    def evaluate_weight_violation(self, knapsack_solution: KnapsackSolution):
        return max(0,
                   knapsack_solution.weight - self.max_capacity)

    def satisfy(self, knapsack_solution: KnapsackSolution):
        if knapsack_solution.value is None:
            self.evaluate(knapsack_solution)
        return knapsack_solution.weight <= self.max_capacity

    def __str__(self):
        s = "Knapsack model with "+str(self.nb_items)+" items and capacity "+str(self.max_capacity)+'\n'
        s += "\n".join([str(item) for item in self.list_items])
        return s

    def get_dummy_solution(self):
        kp_sol = KnapsackSolution(problem=self, list_taken=[0]*self.nb_items)
        self.evaluate(kp_sol)
        return kp_sol
        
    def get_solution_type(self):
        return KnapsackSolution

    # def get_default_objective_weights(self):
    #     return [1.0, -100000]


class KnapsackModel_Mobj(KnapsackModel):
    @staticmethod
    def from_knapsack(knapsack_model: KnapsackModel):
        return KnapsackModel_Mobj(list_items=knapsack_model.list_items,
                                  max_capacity=knapsack_model.max_capacity,
                                  force_recompute_values=knapsack_model.force_recompute_values)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"weight_violation": {"type": TypeObjective.PENALTY, "default_weight": -100000},
                          "heaviest_item": {"type": TypeObjective.OBJECTIVE, "default_weight": 1},
                          "weight": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
                          "value": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}}
        return ObjectiveRegister(objective_handling=ObjectiveHandling.MULTI_OBJ,
                                 dict_objective_to_doc=dict_objective)

    def evaluate(self, knapsack_solution: KnapsackSolution):
        res = super().evaluate(knapsack_solution)
        heaviest = 0
        weight = 0
        for i in range(self.nb_items):
            if knapsack_solution.list_taken[i] == 1:
                heaviest = max(heaviest, self.list_items[i].weight)
                weight += self.list_items[i].weight
        res["heaviest_item"] = heaviest
        res["weight"] = weight
        return res

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]):
        return TupleFitness(np.array([dict_values["value"],
                                      -dict_values["heaviest_item"]]), 2)

    def evaluate_mobj(self, solution: KnapsackSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(solution))





