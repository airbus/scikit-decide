from typing import Dict, Any, List, Union
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, TypeObjective, TypeAttribute, \
    ObjectiveRegister, ObjectiveHandling, EncodingRegister, Solution, ModeOptim
from copy import deepcopy
from abc import abstractmethod
import math
import numpy as np
from numba import njit
from typing import Tuple
from functools import partial


class VrpSolution(Solution):
    def copy(self):
        return VrpSolution(problem=self.problem,
                           list_start_index=self.list_start_index,
                           list_end_index=self.list_end_index,
                           list_paths=deepcopy(self.list_paths), lengths=deepcopy(self.lengths),
                           length=self.length, capacities=deepcopy(self.capacities))

    def lazy_copy(self):
        return VrpSolution(problem=self.problem,
                           list_start_index=self.list_start_index,
                           list_end_index=self.list_end_index,
                           list_paths=self.list_paths,
                           lengths=self.lengths,
                           length=self.length,
                           capacities=self.capacities)

    def __init__(self, problem: Problem,
                 list_start_index,
                 list_end_index,
                 list_paths,
                 length: float=None,
                 lengths: List[List[float]]=None,
                 capacities: List[float]=None):
        self.problem = problem
        self.list_start_index = list_start_index
        self.list_end_index = list_end_index
        self.list_paths = list_paths
        self.length = length
        self.lengths = lengths
        self.capacities = capacities

    def __str__(self):
        return "\n".join([str(self.list_paths[i]) for i in range(len(self.list_paths))])

    def change_problem(self, new_problem):
        self.__init__(problem=new_problem,
                      list_start_index=self.list_start_index,
                      list_end_index=self.list_end_index,
                      list_paths=deepcopy(self.list_paths), lengths=deepcopy(self.lengths),
                      length=self.length, capacities=deepcopy(self.capacities))


class BasicCustomer:
    def __init__(self, name: Union[str, int], demand: float):
        self.name = name
        self.demand = demand


class VrpProblem(Problem):
    def __init__(self,
                 vehicle_count: int,
                 vehicle_capacities: List[float],
                 customer_count: int,
                 customers: List[BasicCustomer],
                 start_indexes: List[int],
                 end_indexes: List[int]):
        self.vehicle_count = vehicle_count
        self.vehicle_capacities = vehicle_capacities
        self.customer_count = customer_count
        self.customers = customers
        self.start_indexes = start_indexes  # for vehicle i : indicate what is the start index
        self.end_indexes = end_indexes  # for vehicle i : indicate what is the end index

    # for a given tsp kind of problem, you should provide a custom evaluate function, for now still abstract.
    @abstractmethod
    def evaluate_function(self, var_tsp: VrpSolution):
        ...

    @abstractmethod
    def evaluate_function_indexes(self, index_1, index_2):
        ...

    def evaluate(self, variable: VrpSolution) -> Dict[str, float]:
        if variable.lengths is None:
            lengths, obj_list, obj, capacity_list = self.evaluate_function(variable)
            variable.length = obj
            variable.lengths = lengths
            variable.capacities = capacity_list
        violation = 0
        for i in range(self.vehicle_count):
            violation += max(variable.capacities[i]-self.vehicle_capacities[i], 0)
        return {'length': variable.length, "capacity_violation": violation}

    def satisfy(self, variable: VrpSolution) -> bool:
        d = self.evaluate(variable)
        return d["capacity_violation"] == 0

    def get_attribute_register(self) -> EncodingRegister:
        dict_encoding = {"list_paths": {"name": "list_paths", "type": [TypeAttribute.LIST_INTEGER]}}
        return EncodingRegister(dict_encoding)

    def get_solution_type(self):
        return VrpSolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"length": {"type": TypeObjective.OBJECTIVE, "default_weight": -1}}
        dict_objective["capacity_violation"] = {"type": TypeObjective.PENALTY, "default_weight": -100000}
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.SINGLE,
                                 dict_objective_to_doc=dict_objective)

    def __str__(self):
        s = 'Vrp problem with \n'+str(self.customer_count) + " customers \nand "+str(self.vehicle_count)+" vehicles "
        return s

    def get_dummy_solution(self):
        s, fit = trivial_solution(self)
        return s

    def get_stupid_solution(self):
        s, fit = stupid_solution(self)
        return s


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)



class Customer2D(BasicCustomer):
    def __init__(self, name: Union[str, int], demand: float, x: float, y: float):
        super().__init__(name=name, demand=demand)
        self.x = x
        self.y = y


class VrpProblem2D(VrpProblem):
    def __init__(self,
                 vehicle_count: int,
                 vehicle_capacities: List[float],
                 customer_count: int,
                 customers: List[Customer2D],
                 start_indexes: List[int],
                 end_indexes: List[int]):
        super().__init__(vehicle_count=vehicle_count,
                         vehicle_capacities=vehicle_capacities,
                         customer_count=customer_count,
                         customers=customers,
                         start_indexes=start_indexes,
                         end_indexes=end_indexes)
        self.customers: List[Customer2D] = self.customers
        self.evaluate_function_2d = build_evaluate_function(self)

    def evaluate_function(self, vrp_sol: VrpSolution):
        return self.evaluate_function_2d(vrp_sol)

    def evaluate_function_indexes(self, index_1, index_2):
        return length(self.customers[index_1],
                      self.customers[index_2])


def trivial_solution(vrp_model: VrpProblem):
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    customers = range(vrp_model.customer_count)
    nb_vehicles = vrp_model.vehicle_count
    nb_customers = vrp_model.customer_count
    remaining_capacity_vehicle = {v: vrp_model.vehicle_capacities[v] for v in range(nb_vehicles)}
    remaining_customers = set(customers)
    for v in range(nb_vehicles):
        start = vrp_model.start_indexes[v]
        end = vrp_model.end_indexes[v]
        remaining_capacity_vehicle[v] -= vrp_model.customers[start].demand
        if end != start:
            remaining_capacity_vehicle[v] -= vrp_model.customers[end].demand
        if start in remaining_customers:
            remaining_customers.remove(start)
        if end in remaining_customers:
            remaining_customers.remove(end)
    for v in range(nb_vehicles):
        vehicle_tours.append([])
        while sum([remaining_capacity_vehicle[v] >= vrp_model.customers[customer].demand
                   for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers,
                           key=lambda x: -vrp_model.customers[x].demand * nb_customers + x)
            for customer in order:
                if remaining_capacity_vehicle[v] >= vrp_model.customers[customer].demand:
                    remaining_capacity_vehicle[v] -= vrp_model.customers[customer].demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used
    solution = VrpSolution(problem=vrp_model,
                           list_start_index=vrp_model.start_indexes,
                           list_end_index=vrp_model.end_indexes,
                           list_paths=vehicle_tours, length=None, lengths=None)
    fit = vrp_model.evaluate(solution)
    return solution, fit


def stupid_solution(vrp_model: VrpProblem):
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    customers = range(vrp_model.customer_count)
    nb_vehicles = vrp_model.vehicle_count
    remaining_capacity_vehicle = {v: vrp_model.vehicle_capacities[v] for v in range(nb_vehicles)}
    remaining_customers = set(customers)
    for v in range(nb_vehicles):
        start = vrp_model.start_indexes[v]
        end = vrp_model.end_indexes[v]
        remaining_capacity_vehicle[v] -= vrp_model.customers[start].demand
        if end != start:
            remaining_capacity_vehicle[v] -= vrp_model.customers[end].demand
        if start in remaining_customers:
            remaining_customers.remove(start)
        if end in remaining_customers:
            remaining_customers.remove(end)
    for v in range(nb_vehicles):
        vehicle_tours.append([])
    vehicle_tours[0] = list(sorted(remaining_customers))
    print(vehicle_tours)
    solution = VrpSolution(problem=vrp_model,
                           list_start_index=vrp_model.start_indexes,
                           list_end_index=vrp_model.end_indexes,
                           list_paths=vehicle_tours, length=None, lengths=None)
    fit = vrp_model.evaluate(solution)
    return solution, fit


def compute_length(start_index,
                   end_index,
                   solution: List[int],
                   list_customers: List[BasicCustomer],
                   method):
    if len(solution) > 0:
        obj = method(start_index, solution[0])
        lengths = [obj]
        capacity = list_customers[start_index].demand
        capacity += list_customers[solution[0]].demand
        for index in range(0, len(solution)-1):
            ll = method(solution[index], solution[index+1])
            obj += ll
            lengths += [ll]
            capacity += list_customers[solution[index+1]].demand
        lengths += [method(end_index, solution[-1])]
        if end_index != start_index:
            capacity += list_customers[end_index].demand
        obj += lengths[-1]
    else:
        obj = method(start_index, end_index)
        lengths = [obj]
        capacity = list_customers[start_index].demand
        if end_index != start_index:
            capacity += list_customers[end_index].demand
    return lengths, obj, capacity


# More efficient implementation
@njit
def compute_length_np(start_index,
                      end_index,
                      solution: Union[List[int], np.array],
                      np_points)->Tuple[Union[List[float], np.array], float]:
    obj = np.sqrt((np_points[start_index, 0]-np_points[solution[0], 0])**2+\
                  (np_points[start_index, 1]-np_points[solution[0], 1])**2)
    len_sol = len(solution)
    lengths = np.zeros((len_sol+1))
    lengths[0] = obj
    for index in range(0, len_sol-1):
        ll = math.sqrt((np_points[solution[index], 0]-np_points[solution[index+1], 0])**2+\
                       (np_points[solution[index], 1]-np_points[solution[index+1], 1])**2)
        obj += ll
        lengths[index+1] = ll
    lengths[len_sol] = np.sqrt((np_points[end_index, 0]-np_points[solution[-1], 0])**2+\
                               (np_points[end_index, 1]-np_points[solution[-1], 1])**2)
    obj += lengths[len_sol]
    return lengths, obj


def sequential_computing(vrp_sol: VrpSolution, vrp_model: VrpProblem):
    lengths_list = []
    obj_list = []
    capacity_list = []
    sum_obj = 0
    for i in range(len(vrp_sol.list_paths)):
        lengths, obj, capacity = compute_length(start_index=vrp_sol.list_start_index[i],
                                                end_index=vrp_sol.list_end_index[i],
                                                solution=vrp_sol.list_paths[i],
                                                list_customers=vrp_model.customers,
                                                method=vrp_model.evaluate_function_indexes)
        lengths_list += [lengths]
        obj_list += [obj]
        capacity_list += [capacity]
        sum_obj += obj
    return lengths_list, obj_list, sum_obj, capacity_list


def build_evaluate_function(vrp_model: VrpProblem):
    return partial(sequential_computing,
                   vrp_model=vrp_model)




