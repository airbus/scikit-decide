from typing import Dict, List, Any
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, Solution, ObjectiveRegister, EncodingRegister, \
    TypeObjective, ObjectiveHandling, TypeAttribute, ModeOptim
import math
from collections import namedtuple
from abc import abstractmethod
from copy import deepcopy

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


class FacilitySolution(Solution):

    def __init__(self, problem: Problem,
                 facility_for_customers: List[int], dict_details: Dict[str, Any]=None):
        """
        :param problem: a FacilityProblem object.
        :param facility_for_customers: list of size "nb_customers"
        forall i in [0, nb_customers-1], facility_for_customers[i] \in [0, nb_facilities-1]
        is the index of the facility location for the i-th customer.
        :param dict_details: giving more metrics of the solution such as the capacities used, the setup cost etc.
        See problem.evaluate(sol) implementation for FacilityProblem
        """
        self.problem = problem
        self.facility_for_customers = facility_for_customers
        self.dict_details = dict_details

    def copy(self):
        return FacilitySolution(self.problem, facility_for_customers=list(self.facility_for_customers),
                                dict_details=deepcopy(self.dict_details))

    def lazy_copy(self):
        return FacilitySolution(self.problem,
                                facility_for_customers=self.facility_for_customers,
                                dict_details=self.dict_details)
    
    def change_problem(self, new_problem):
        self.__init__(problem=new_problem,
                      facility_for_customers=list(self.facility_for_customers),
                      dict_details=deepcopy(self.dict_details))

class FacilityProblem(Problem):
    def __init__(self, facility_count: int,
                 customer_count: int,
                 facilities: List[Facility],
                 customers: List[Customer]):
        self.facility_count = facility_count
        self.customer_count = customer_count
        self.facilities = facilities
        self.customers = customers

    @abstractmethod
    def evaluate_customer_facility(self, facility: Facility, customer: Customer)->float:
        ...

    def evaluate(self, variable: FacilitySolution) -> Dict[str, float]:
        if variable.dict_details is not None:
            return variable.dict_details
        d = self.evaluate_cost(variable)
        capacity_constraint_violation = 0
        for f in d["details"]:
            capacity_constraint_violation = max(d["details"][f]["capacity_used"]-self.facilities[f].capacity, 0)
        d["capacity_constraint_violation"] = capacity_constraint_violation
        return d

    def evaluate_from_encoding(self, int_vector, encoding_name):
        kp_sol = None
        if encoding_name == 'facility_for_customers':
            kp_sol = FacilitySolution(problem=self, facility_for_customers=int_vector)
        elif encoding_name == 'custom':
            kwargs = {encoding_name: int_vector, 'problem': self}
            kp_sol = FacilitySolution(**kwargs)
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate_cost(self, variable: FacilitySolution):
        facility_details = {}
        cost = 0.
        setup_cost = 0.
        for i in range(self.customer_count):
            f = variable.facility_for_customers[i]
            if f not in facility_details:
                facility_details[f] = {"capacity_used": 0.,
                                       "customers": set(),
                                       "cost": 0.,
                                       "setup_cost": self.facilities[f].setup_cost}
                setup_cost += facility_details[f]["setup_cost"]
            facility_details[f]["capacity_used"] += self.customers[i].demand
            facility_details[f]["customers"].add(i)
            c = self.evaluate_customer_facility(facility=self.facilities[f],
                                                customer=self.customers[i])
            facility_details[f]["cost"] += c
            cost += c
        return {"cost": cost, "setup_cost": setup_cost, "details": facility_details}

    def satisfy(self, variable: FacilitySolution) -> bool:
        d = self.evaluate(variable)
        return d["capacity_constraint_violation"] == 0.

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = dict()
        dict_register["facility_for_customers"] = {"name": "facility_for_customers",
                                                   "type": [TypeAttribute.LIST_INTEGER],
                                                   "n": self.customer_count,
                                                   "arrity": self.facility_count}
        return EncodingRegister(dict_register)

    def get_dummy_solution(self):
        return FacilitySolution(self, [0]*self.customer_count)

    def get_solution_type(self):
        return FacilitySolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"cost": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
                          "setup_cost": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
                          "capacity_constraint_violation": {"type": TypeObjective.OBJECTIVE,
                                                            "default_weight": -10000}}
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.AGGREGATE,
                                 dict_objective_to_doc=dict_objective)


def length(point1: Point, point2: Point):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class FacilityProblem2DPoints(FacilityProblem):

    def evaluate_customer_facility(self, facility: Facility, customer: Customer) -> float:
        return length(facility.location, customer.location)




