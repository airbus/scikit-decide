import os
from minizinc import Instance, Model, Solver, Status
from skdecide.builders.discrete_optimization.facility.facility_model import FacilityProblem, FacilitySolution
from skdecide.builders.discrete_optimization.facility.solvers.greedy_solvers import GreedySolverFacility
from skdecide.builders.discrete_optimization.facility.solvers.facility_lp_solver import compute_length_matrix
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ParamsObjectiveFunction, \
    get_default_objective_setup
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
import random
from datetime import timedelta
from enum import Enum
path_minizinc = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "../minizinc/"))


class FacilityCPModel(Enum):
    DEFAULT_INT = 0
    DEFAULT_INT_LNS = 1


file_dict = {FacilityCPModel.DEFAULT_INT: "facility_int.mzn",
             FacilityCPModel.DEFAULT_INT_LNS: "facility_int_lns.mzn"}


class FacilityCP(SolverDO):
    def __init__(self, facility_problem: FacilityProblem,
                 params_objective_function: ParamsObjectiveFunction):
        self.facility_problem = facility_problem
        self.params_objective_function = params_objective_function
        if self.params_objective_function is None:
            self.params_objective_function = get_default_objective_setup(self.facility_problem)
        _, self.aggreg = build_evaluate_function_aggregated(problem=self.facility_problem,
                                                            params_objective_function=self.params_objective_function)
        self.model = None
        self.instance = None

    def init_model(self, **kwargs):
        verbose = kwargs.get("verbose", False)
        model_type = kwargs.get("cp_model", FacilityCPModel.DEFAULT_INT)
        path = os.path.join(path_minizinc, file_dict[model_type])
        self.model = Model(path)
        solver = Solver.lookup("chuffed")
        instance = Instance(solver, self.model)
        # Assign 4 to n
        instance["nb_facilities"] = self.facility_problem.facility_count
        instance["nb_customers"] = self.facility_problem.customer_count
        setup_costs, closests, distances = compute_length_matrix(self.facility_problem)
        if model_type in [FacilityCPModel.DEFAULT_INT, FacilityCPModel.DEFAULT_INT_LNS]:
            distances = [[int(distances[f, c]) for c in range(distances.shape[1])]
                         for f in range(distances.shape[0])]
            instance["distance"] = distances
            instance["setup_cost_vector"] = [int(s) for s in setup_costs]
            instance["demand"] = [int(self.facility_problem.customers[c].demand)
                                  for c in range(self.facility_problem.customer_count)]
            instance["capacity"] = [int(self.facility_problem.facilities[f].capacity)
                                    for f in range(self.facility_problem.facility_count)]
        else:
            distances = [[distances[f, c] for c in range(distances.shape[1])]
                         for f in range(distances.shape[0])]
            instance["distance"] = distances
            instance["setup_cost_vector"] = [s for s in setup_costs]
            instance["demand"] = [self.facility_problem.customers[c].demand
                                  for c in range(self.facility_problem.customer_count)]
            instance["capacity"] = [self.facility_problem.facilities[f].capacity
                                    for f in range(self.facility_problem.facility_count)]
        self.instance = instance

    def solve(self, **kwargs):
        if self.model is None:
            self.init_model(**kwargs)
            print(self.model)
        limit_time_s = kwargs.get("limit_time_s", 100)
        result = self.instance.solve(timeout=timedelta(seconds=limit_time_s))
        # Output the array q
        opt: Status = result.status
        facility_sol = FacilitySolution(self.facility_problem,
                                        [f-1 for f in result["facility_for_customer"]])
        fit = self.facility_problem.evaluate(facility_sol)
        return facility_sol, fit

    def get_solution(self, **kwargs):
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedySolverFacility(self.facility_problem)
            solution, f = greedy_solver.solve()
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.facility_problem.get_dummy_solution()
        print("Greedy Done")
        return solution

    def solve_lns(self, fraction_to_fix: float=0.9,
                  nb_iteration: int=10,
                  **kwargs):
        first_solution = self.get_solution(**kwargs)
        dict_color = {i+1: first_solution.facility_for_customers[i]+1
                      for i in range(self.facility_problem.customer_count)}
        self.init_model(**kwargs)
        limit_time_s = kwargs.get("limit_time_s", 100)
        range_node = range(1, self.facility_problem.customer_count+1)
        iteration = 0
        current_solution = first_solution
        current_best_solution = current_solution.copy()
        current_objective = self.aggreg(self.facility_problem.evaluate(current_best_solution))
        while iteration < nb_iteration:
            with self.instance.branch() as child:
                subpart_color = set(random.sample(range_node, int(fraction_to_fix * self.facility_problem.customer_count)))
                for i in range_node:
                    if i in subpart_color:
                        # print("constraint color_graph["+str(i)+"] == "+ str(dict_color[i])+";\n")
                        child.add_string("constraint facility_for_customer["+str(i)+"] == " + str(dict_color[i])+";\n")
                child.add_string(f"solve :: int_search(facility_for_customer,"
                                 f" input_order, indomain_min, complete) minimize(objective);\n")
                print("Solving... ", iteration)
                res = child.solve(timeout=timedelta(seconds=limit_time_s))
                print(res.status)
                if res.solution is not None and -res["objective"] > current_objective:
                    current_objective = -res["objective"]
                    current_best_solution = FacilitySolution(self.facility_problem,
                                                             [f-1 for f in res["facility_for_customer"]])
                    fit = self.facility_problem.evaluate(current_best_solution)
                    dict_color = {i + 1: current_best_solution.facility_for_customers[i]+1
                                  for i in range(self.facility_problem.customer_count)}
                    print(iteration, " : , ", res["objective"])
                    print('IMPROVED : ')
                else:
                    try:
                        print(iteration, " :  ", res["objective"])
                    except:
                        print(iteration, " failed ")
                    # print({i: res["color_graph"][i-1] for i in range_node})
                iteration += 1
        fit = self.facility_problem.evaluate(current_best_solution)
        return current_best_solution, fit


