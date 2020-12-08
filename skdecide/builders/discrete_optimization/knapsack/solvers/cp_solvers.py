from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP, CPSolverName, map_cp_solver_name
from skdecide.builders.discrete_optimization.knapsack.knapsack_model import KnapsackModel, KnapsackSolution
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from minizinc import Instance, Model, Solver, Status, Result
from datetime import timedelta
import os, random
this_path = os.path.dirname(os.path.abspath(__file__))


class CPKnapsackMZN(CPSolver):
    def __init__(self, knapsack_model: KnapsackModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.knapsack_model = knapsack_model
        self.instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["list_items"]
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        l_items = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                l_items += [result[i, "list_items"]]
                objectives += [result[i, "objective"]]
        else:
            l_items += [result["list_items"]]
            objectives += [result["objective"]]
        list_solutions_fit = []
        for items, objective in zip(l_items, objectives):
            taken = [0] * self.knapsack_model.nb_items
            weight = 0
            value = 0
            for i in range(len(items)):
                if items[i] != 0:
                    taken[self.knapsack_model.list_items[items[i] - 1].index] = 1
                    weight += self.knapsack_model.list_items[items[i] - 1].weight
                    value += self.knapsack_model.list_items[items[i] - 1].value
            sol = KnapsackSolution(problem=self.knapsack_model,
                                   value=value,
                                   weight=weight,
                                   list_taken=taken)
            fit = self.aggreg_sol(sol)
            list_solutions_fit += [(sol, fit)]
        return ResultStorage(list_solution_fits=list_solutions_fit, best_solution=None,
                             mode_optim=self.params_objective_function.sense_function)

    def init_model(self, **args):
        # Load n-Queens model from file
        model = Model(os.path.join(this_path, "../minizinc/knapsack_mzn.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["nb_items"] = self.knapsack_model.nb_items
        instance["values"] = [0]+[self.knapsack_model.list_items[i].value 
                                  for i in range(self.knapsack_model.nb_items)]
        instance["weights"] = [0]+[self.knapsack_model.list_items[i].weight 
                                   for i in range(self.knapsack_model.nb_items)]
        instance["max_capacity"] = self.knapsack_model.max_capacity
        self.instance = instance
        
    def solve(self, parameters_cp: ParametersCP=ParametersCP.default(), **args):
        result = self.instance.solve(timeout=timedelta(seconds=parameters_cp.TimeLimit),
                                     intermediate_solutions=parameters_cp.intermediate_solution)
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)


class CPKnapsackMZN2(CPSolver):
    def __init__(self, knapsack_model: KnapsackModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.knapsack_model = knapsack_model
        self.instance = None
        self.cp_solver_name = cp_solver_name
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        # Load n-Queens model from file
        model = Model(os.path.join(this_path, "../minizinc/knapsack_global.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["nb_items"] = self.knapsack_model.nb_items
        instance["values"] = [self.knapsack_model.list_items[i].value 
                                  for i in range(self.knapsack_model.nb_items)]
        instance["weights"] = [self.knapsack_model.list_items[i].weight 
                               for i in range(self.knapsack_model.nb_items)]
        instance["max_capacity"] = self.knapsack_model.max_capacity
        self.instance = instance

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        l_items_taken = []
        intermediate_solution = parameters_cp.intermediate_solution
        if intermediate_solution:
            for i in range(len(result)):
                l_items_taken += [result[i, "taken"]]
        else:
            l_items_taken += [result["taken"]]
        list_solution_fits = []
        for i in range(len(l_items_taken)):
            items_taken = l_items_taken[i]
            taken = [0] * self.knapsack_model.nb_items
            weight = 0
            value = 0
            for i in range(len(items_taken)):
                if items_taken[i] != 0:
                    taken[self.knapsack_model.list_items[i].index] = 1
                    weight += self.knapsack_model.list_items[i].weight
                    value += self.knapsack_model.list_items[i].value
            sol = KnapsackSolution(problem=self.knapsack_model,
                                   value=value, weight=weight, list_taken=taken)
            fit = self.aggreg_sol(sol)
            list_solution_fits += [(sol, fit)]
        return ResultStorage(list_solution_fits=list_solution_fits, best_solution=None,
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameter_cp: ParametersCP=ParametersCP.default(), **args):
        result = self.instance.solve(timeout=timedelta(seconds=parameter_cp.TimeLimit),
                                     intermediate_solutions=parameter_cp.intermediate_solution)
        return self.retrieve_solutions(result=result, parameters_cp=parameter_cp)

    def retrieve(self, items_taken):
        taken = [0]*self.knapsack_model.nb_items
        weight = 0
        value = 0
        for i in range(len(items_taken)):
            if items_taken[i] != 0:
                taken[self.knapsack_model.list_items[i].index] = 1
                weight += self.knapsack_model.list_items[i].weight
                value += self.knapsack_model.list_items[i].value
        return [KnapsackSolution(problem=self.knapsack_model,
                                 value=value, weight=weight, list_taken=taken)]

    def solve_lns(self, 
                  init_solution: KnapsackSolution, 
                  fraction_decision_fixed: float, 
                  nb_iteration_max: int, 
                  max_time_per_iteration_s: float, save_results=True):
        taken_current_solution = [init_solution.list_taken[item.index] for item in self.knapsack_model.list_items]
        current_objective = init_solution.value
        nb_items = self.knapsack_model.nb_items
        range_item = list(range(nb_items))
        current_solution = init_solution
        iteration = 0
        results = {"taken": [init_solution.list_taken], 
                   "objective": [init_solution.value],
                   "weight": [init_solution.weight]}
        while iteration < nb_iteration_max:
            with self.instance.branch() as child:
                subpart_item = set(random.sample(range_item, int(fraction_decision_fixed*nb_items)))
                for i in range_item:
                    if i in subpart_item:
                        child.add_string("constraint taken["+str(i+1)+"] == "+ str(taken_current_solution[i])+";\n")
                res = child.solve(timeout=timedelta(seconds=max_time_per_iteration_s))
                if res.solution is not None:
                    solution = self.retrieve(res["taken"])[0]
                    print(res.status)
                if res.solution is not None and res["objective"] > current_objective:
                    current_objective = res["objective"]
                    iteration += 1
                    current_solution = solution
                    taken_current_solution = [solution.list_taken[item.index]
                                              for item in self.knapsack_model.list_items]
                    print("Improved ", current_objective)
                    if save_results:
                        results["taken"] += [taken_current_solution]
                        results["objective"] += [current_objective]
                        results["weight"] += [res["weight"]]
                else:
                    iteration += 1
                    if res.solution is not None:
                        objective = res["objective"]
                        taken_solution = [solution.list_taken[item.index] for item in self.knapsack_model.list_items]
                        if save_results:
                            results["taken"] += [taken_solution]
                            results["objective"] += [objective]
                            results["weight"] += [res["weight"]]
        return [current_solution], results
