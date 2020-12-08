from skdecide.builders.discrete_optimization.knapsack.knapsack_model import KnapsackModel, KnapsackSolution
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction


def compute_density(knapsack_model: KnapsackModel):
    dd = sorted([l for l in knapsack_model.list_items 
                 if l.weight<=knapsack_model.max_capacity], 
                key=lambda x: x.value/x.weight, 
                reverse=True)
    return dd


def compute_density_and_penalty(knapsack_model: KnapsackModel):
    dd = sorted([l for l in knapsack_model.list_items  if l.weight<=knapsack_model.max_capacity], 
                key=lambda x: x.value/x.weight-x.weight, 
                reverse=True)
    return dd


def greedy_using_queue(knapsack_model: KnapsackModel, method_queue=None)->KnapsackSolution:
    if method_queue is None:
        method_queue = compute_density
    value = 0
    weight = 0
    taken = [0]*knapsack_model.nb_items
    sorted_per_density = method_queue(knapsack_model)
    for i in range(len(taken)):
        if sorted_per_density[i].weight+weight<=knapsack_model.max_capacity:
            taken[sorted_per_density[i].index] = 1
            value += sorted_per_density[i].value
            weight += sorted_per_density[i].weight
        else:
            continue
    return KnapsackSolution(problem=knapsack_model, value=value, weight=weight, list_taken=taken)


def best_of_greedy(knapsack_model: KnapsackModel)->KnapsackSolution:
    result1 = greedy_using_queue(knapsack_model, compute_density)
    result2 = greedy_using_queue(knapsack_model, compute_density_and_penalty)
    return result1 if result1.value>result2.value else result2


class GreedyBest(SolverDO):
    def __init__(self, knapsack_model: KnapsackModel,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.knapsack_model = knapsack_model
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, *args):
        pass

    def solve(self):
        res = best_of_greedy(self.knapsack_model)
        fit = self.aggreg_sol(res)
        return ResultStorage(list_solution_fits=[(res, fit)],
                             best_solution=res,
                             mode_optim=self.params_objective_function.sense_function)


class GreedyDummy(SolverDO):
    def __init__(self, knapsack_model: KnapsackModel,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.knapsack_model = knapsack_model
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, *args):
        pass

    def solve(self):
        sol = KnapsackSolution(problem=self.knapsack_model,
                               value=0,
                               weight=0,
                               list_taken=[0]*self.knapsack_model.nb_items)
        fit = self.aggreg_sol(sol)
        return ResultStorage(list_solution_fits=[(sol, fit)],
                             best_solution=sol,
                             mode_optim=self.params_objective_function.sense_function)
