from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from skdecide.builders.discrete_optimization.vrp.vrp_model import length, BasicCustomer, VrpProblem, \
    VrpProblem2D, VrpSolution, Customer2D
from skdecide.builders.discrete_optimization.vrp.vrp_toolbox import compute_length_matrix, build_graph
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction,\
    build_aggreg_function_and_params_objective
from enum import Enum


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [
        [
            0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354,
            468, 776, 662
        ],
        [
            548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
            1016, 868, 1210
        ],
        [
            776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164,
            1130, 788, 1552, 754
        ],
        [
            696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
            1164, 560, 1358
        ],
        [
            582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
            1050, 674, 1244
        ],
        [
            274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628,
            514, 1050, 708
        ],
        [
            502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856,
            514, 1278, 480
        ],
        [
            194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320,
            662, 742, 856
        ],
        [
            308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662,
            320, 1084, 514
        ],
        [
            194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388,
            274, 810, 468
        ],
        [
            536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764,
            730, 388, 1152, 354
        ],
        [
            502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114,
            308, 650, 274, 844
        ],
        [
            388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194,
            536, 388, 730
        ],
        [
            354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0,
            342, 422, 536
        ],
        [
            468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536,
            342, 0, 764, 194
        ],
        [
            776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274,
            388, 422, 764, 0, 798
        ],
        [
            662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730,
            536, 194, 798, 0
        ],
    ]
    data['demands'] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    data['vehicle_capacities'] = [15, 15, 15, 15]
    data['num_vehicles'] = 4
    data['depot'] = 0
    return data


class FirstSolutionStrategy(Enum):
    SAVINGS = 0
    PATH_MOST_CONSTRAINED_ARC = 1


class LocalSearchMetaheuristic(Enum):
    GUIDED_LOCAL_SEARCH = 0
    SIMULATED_ANNEALING = 1


first_solution_map = {FirstSolutionStrategy.SAVINGS: routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
                      FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC: routing_enums_pb2.FirstSolutionStrategy.SAVINGS}
metaheuristic_map = {LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH:
                     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
                     LocalSearchMetaheuristic.SIMULATED_ANNEALING:
                     routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING}


class VrpORToolsSolver(SolverDO):
    def __init__(self, problem: VrpProblem,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.problem = problem
        self.manager = None
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **kwargs):
        first_solution_strategy = kwargs.get("first_solution_strategy",
                                             FirstSolutionStrategy.SAVINGS)
        local_search_metaheuristic = kwargs.get("local_search_metaheuristic",
                                                LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        first_solution_strategy = first_solution_map[first_solution_strategy]
        local_search_metaheuristic = metaheuristic_map[local_search_metaheuristic]
        G, matrix_distance = build_graph(self.problem)
        matrix_distance_int = np.array(10 ** 5 * matrix_distance, dtype=np.int)
        demands = [self.problem.customers[i].demand for i in range(self.problem.customer_count)]
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(self.problem.customer_count,
                                               self.problem.vehicle_count,
                                               self.problem.start_indexes,
                                               self.problem.end_indexes)
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return matrix_distance_int[from_node, to_node]
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_callback_index,
                                                0,  # null capacity slack
                                                self.problem.vehicle_capacities,  # vehicle maximum capacities
                                                True,  # start cumul to zero
                                                'Capacity')
        # initial_solution = routing.ReadAssignmentFromRoutes(vehicle_tours,
        #                                                     True)
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = first_solution_strategy
        search_parameters.local_search_metaheuristic = local_search_metaheuristic
        search_parameters.time_limit.seconds = 100
        # Solve the problem.
        self.manager = manager
        self.routing = routing
        self.search_parameters = search_parameters
        print("Initialized ...")

    def retrieve(self, solution):
        vehicle_tours = []
        vehicle_tours_all = []
        vehicle_count = self.problem.vehicle_count
        objective = 0
        route_distance = 0
        for vehicle_id in range(vehicle_count):
            vehicle_tours.append([])
            vehicle_tours_all.append([])
            index = self.routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_load = 0
            cnt = 0
            print(vehicle_id)
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                if cnt != 0:
                    vehicle_tours[-1] += [node_index]
                vehicle_tours_all[-1] += [node_index]
                cnt += 1
                route_load += self.problem.customers[node_index].demand
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                objective += self.problem.evaluate_function_indexes(node_index,
                                                                    self.manager.IndexToNode(index))
            vehicle_tours_all[-1] += [self.manager.IndexToNode(index)]
        print("Route distance : ", route_distance)
        print("Vehicle tours : ", vehicle_tours)
        print("Objective : ", objective)
        print("Vehicle tours all : ", vehicle_tours_all)
        variable_vrp = VrpSolution(problem=self.problem,
                                   list_start_index=self.problem.start_indexes,
                                   list_end_index=self.problem.end_indexes,
                                   list_paths=vehicle_tours,
                                   length=None,
                                   lengths=None,
                                   capacities=None)
        return variable_vrp

    def solve(self, **kwargs):
        if self.manager is None:
            self.init_model(**kwargs)
        limit_time_s = kwargs.get("limit_time_s", 100)
        self.search_parameters.time_limit.seconds = limit_time_s
        print("Solving")
        solution = self.routing.SolveWithParameters(self.search_parameters)
        print(solution)
        variable_vrp = self.retrieve(solution)
        fit = self.aggreg_sol(variable_vrp)
        return ResultStorage(list_solution_fits=[(variable_vrp, fit)],
                             mode_optim=self.params_objective_function.sense_function)








