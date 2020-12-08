from skdecide.builders.discrete_optimization.facility.facility_model import FacilityProblem, FacilitySolution
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction,\
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage


class GreedySolverFacility(SolverDO):
    """
    build a trivial solution
    pack the facilities one by one until all the customers are served
    """
    def __init__(self, facility_problem: FacilityProblem,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.facility_problem = facility_problem
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.facility_problem,
                                                       params_objective_function=params_objective_function)

    def solve(self, **kwargs)->ResultStorage:
        solution = [-1]*self.facility_problem.customer_count
        capacity_remaining = [f.capacity for f in self.facility_problem.facilities]
        facility_index = 0
        for index in range(len(self.facility_problem.customers)):
            customer = self.facility_problem.customers[index]
            if capacity_remaining[facility_index] >= customer.demand:
                solution[index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert capacity_remaining[facility_index] >= customer.demand
                solution[index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
        sol = FacilitySolution(problem=self.facility_problem,
                               facility_for_customers=solution)
        fit = self.aggreg_sol(sol)
        return ResultStorage(list_solution_fits=[(sol, fit)],
                             best_solution=sol,
                             mode_optim=self.params_objective_function.sense_function)
