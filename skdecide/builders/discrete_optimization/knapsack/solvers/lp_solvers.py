from skdecide.builders.discrete_optimization.knapsack.knapsack_model import Item, KnapsackModel, KnapsackSolution
from typing import Dict, Iterable
from gurobi import LinExpr, Model, GRB, quicksum
from ortools.linear_solver import pywraplp
from ortools.algorithms import pywrapknapsack_solver
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import ParametersMilp, MilpSolverName
import matplotlib.pyplot as plt
from tqdm import trange
import random
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.mip.pymip_tools import MyModelMilp, CBC, \
    MAXIMIZE, BINARY, CONTINUOUS, INTEGER, xsum
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolver
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolver, ParametersMilp, MilpSolverName, map_solver


class LPKnapsackGurobi(SolverDO):
    def __init__(self, knapsack_model: KnapsackModel,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.knapsack_model = knapsack_model
        self.model = None
        self.variable_decision = {}
        self.constraints_dict = {}
        self.description_variable_description = {}
        self.description_constraint = {}
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self,  **args):
        warm_start = args.get('warm_start', {}) 
        self.model = Model("Knapsack")
        self.variable_decision = {"x": {}}
        self.description_variable_description = {"x": {"shape": self.knapsack_model.nb_items,
                                                       "type": bool, 
                                                       "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not"}}
        self.description_constraint["weight"] = {"descr": "sum of weight of used items doesn't exceed max capacity"}
        weight = {}
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        x = {}
        for item in list_item:
            i = item.index
            x[i] = self.model.addVar(vtype=GRB.BINARY, 
                                     obj=item.value,
                                     name="x_"+str(i))
            if i in warm_start:
                x[i].start = warm_start[i]
                x[i].varhinstval = warm_start[i]
            weight[i] = item.weight
        self.variable_decision["x"] = x
        self.model.update()
        self.constraints_dict["weight"] = self.model.addConstr(quicksum([weight[i]*x[i] for i in x])<=max_capacity)
        self.model.update()
        self.model.setParam("TimeLimit", 200)
        self.model.modelSense = GRB.MAXIMIZE
        self.model.setParam(GRB.Param.PoolSolutions, 10000)
        self.model.setParam("MIPGapAbs", 0.00001)
        self.model.setParam("MIPGap", 0.00000001)
    
    def retrieve_solutions(self, range_solutions: Iterable[int]):
        # nObjectives = S.NumObj
        solutions = []
        fits = []
        # x = S.getVars()
        for s in range_solutions:
            weight = 0
            xs = {}
            self.model.params.SolutionNumber = s
            obj = self.model.getAttr("ObjVal")
            for e in self.variable_decision["x"]:
                value = self.variable_decision["x"][e].getAttr('Xn')
                if value <= 0.1:
                    xs[e] = 0
                    continue
                xs[e] = 1
                weight += self.knapsack_model.index_to_item[e].weight
            solutions += [KnapsackSolution(problem=self.knapsack_model,
                                           value=obj,
                                           weight=weight, 
                                           list_taken=[xs[e] for e in sorted(xs)])]
            fits += [self.aggreg_sol(solutions[-1])]
        return ResultStorage(list_solution_fits=[(s, f) for s, f in zip(solutions, fits)],
                             mode_optim=self.params_objective_function.sense_function)
    
    def solve(self, parameter_gurobi: ParametersMilp):
        self.model.setParam("TimeLimit", parameter_gurobi.TimeLimit)
        self.model.modelSense = GRB.MAXIMIZE
        self.model.setParam(GRB.Param.PoolSolutions, parameter_gurobi.PoolSolutions)
        self.model.setParam("MIPGapAbs", parameter_gurobi.MIPGapAbs)
        self.model.setParam("MIPGap", parameter_gurobi.MIPGap)
        print("optimizing...")
        self.model.optimize()
        nSolutions = self.model.SolCount
        nObjectives = self.model.NumObj
        objective = self.model.getObjective().getValue()
        print('Problem has', nObjectives, 'objectives')
        print('Gurobi found', nSolutions, 'solutions')
        if parameter_gurobi.retrieve_all_solution:
            solutions = self.retrieve_solutions(list(range(nSolutions)))
        else:
            solutions = self.retrieve_solutions([0])
        return solutions

    def solve_lns(self,
                  parameter_gurobi: ParametersMilp,
                  init_solution: KnapsackSolution,
                  fraction_decision_fixed: float,
                  nb_iteration_max: int):
        self.model.setParam("TimeLimit", parameter_gurobi.TimeLimit)
        self.model.setParam("OutputFlag", 0)
        self.model.modelSense = GRB.MAXIMIZE
        self.model.setParam(GRB.Param.PoolSolutions, parameter_gurobi.PoolSolutions)
        self.model.setParam("MIPGapAbs", parameter_gurobi.MIPGapAbs)
        self.model.setParam("MIPGap", parameter_gurobi.MIPGap)
        current_solution = init_solution
        constraints = {}
        list_solutions = [current_solution]
        list_objective = [current_solution.value]
        objective = init_solution.value
        for k in trange(nb_iteration_max):
            for c in constraints:
                self.model.remove(constraints[c])
            self.add_init_solution(current_solution)
            fixed_variable = set(random.sample(self.variable_decision["x"].keys(),
                                              int(fraction_decision_fixed*len(self.variable_decision["x"]))))
            constraints = self.fix_decision(current_solution, fixed_variable)
            self.model.optimize()
            nSolutions = self.model.SolCount
            nObjectives = self.model.NumObj
            objective = self.model.getObjective().getValue()
            if parameter_gurobi.retrieve_all_solution:
                solutions = self.retrieve_solutions(list(range(nSolutions)))
            else:
                solutions = self.retrieve_solutions([0])
            current_solution = solutions[0]
            list_solutions += [solutions[0]]
            list_objective += [solutions[0].value]
        print("Last obj : ", list_objective[-1])
        fig, ax = plt.subplots(1)
        ax.plot(list_objective)
        plt.show()

    def add_init_solution(self, init_solution: KnapsackSolution):
        for i in self.variable_decision["x"]:
            self.variable_decision["x"][i].start = init_solution.list_taken[i]
            self.variable_decision["x"][i].varhintval = init_solution.list_taken[i]

    def fix_decision(self,  init_solution: KnapsackSolution, fixed_variable_keys):
        constraints = {}
        for i in fixed_variable_keys:
            constraints[i] = self.model.addConstr(self.variable_decision["x"][i] == init_solution.list_taken[i])
        return constraints

    def describe_the_model(self):
        return str(self.description_variable_description)+"\n"+str(self.description_constraint)


class LPKnapsackCBC(SolverDO):
    def __init__(self, knapsack_model: KnapsackModel,
                 params_objective_function: ParamsObjectiveFunction):
        self.knapsack_model = knapsack_model
        self.model = None
        self.variable_decision = {}
        self.constraints_dict = {}
        self.description_variable_description = {}
        self.description_constraint = {}
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)
    
    def init_model(self, warm_start: Dict[int, int]=None):
        if warm_start is None:
            warm_start = {}
        self.variable_decision = {"x": {}}
        self.description_variable_description = {"x": {"shape": self.knapsack_model.nb_items,
                                                       "type": bool, 
                                                       "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not"}}
        self.description_constraint["weight"] = {"descr": "sum of weight of used items doesn't exceed max capacity"}
        S = pywraplp.Solver('knapsack',
                            pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        #S.EnableOutput()
        x = {}
        weight = {}
        value = {}
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        for item in list_item:
            i = item.index
            x[i] = S.BoolVar("x_"+str(i))
            if i in warm_start:
                S.SetHint([x[i]], [warm_start[i]])
            weight[i] = item.weight
            value[i] = item.value
        self.constraints_dict["weight"] = S.Add(S.Sum([x[i] * weight[i]
                                                       for i in x])<=max_capacity)
        value_knap = S.Sum([x[i] * value[i]
                            for i in x])
        S.Maximize(value_knap)
        self.model = S
        self.variable_decision["x"] = x
    
    def solve(self):
        self.model.SetTimeLimit(60000)
        res = self.model.Solve()
        resdict = {0: 'OPTIMAL', 1: 'FEASIBLE', 2: 'INFEASIBLE', 3: 'UNBOUNDED',
                   4: 'ABNORMAL', 5: 'MODEL_INVALID', 6: 'NOT_SOLVED'}
        print('Result:', resdict[res])
        objective = self.model.Objective().Value()
        xs = {}
        x = self.variable_decision["x"]
        weight = 0
        for i in x:
            sv = x[i].solution_value()
            if sv>=0.5:
                xs[i] = 1
                weight += self.knapsack_model.index_to_item[i].weight
            else:
                xs[i] = 0
        sol = KnapsackSolution(problem=self.knapsack_model,
                                 value=objective,
                                 weight=weight, 
                                 list_taken=[xs[e] for e in sorted(xs)])
        fit = self.aggreg_sol(sol)
        return ResultStorage(list_solution_fits=[(sol, fit)],
                             mode_optim=self.params_objective_function.sense_function)

    def describe_the_model(self):
        return str(self.description_variable_description)+"\n"+str(self.description_constraint)


# Can use GRB or CBC
class LPKnapsack(MilpSolver):
    def __init__(self, knapsack_model: KnapsackModel,
                 milp_solver_name: MilpSolverName,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.knapsack_model = knapsack_model
        self.model: MyModelMilp = None
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[milp_solver_name]
        self.variable_decision = {}
        self.constraints_dict = {}
        self.description_variable_description = {}
        self.description_constraint = {}
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        warm_start = args.get('warm_start', {})
        solver_name = args.get("solver_name", CBC)
        self.model = MyModelMilp("Knapsack",
                                 solver_name=solver_name,
                                 sense=MAXIMIZE)
        self.variable_decision = {"x": {}}
        self.description_variable_description = {"x": {"shape": self.knapsack_model.nb_items,
                                                       "type": bool,
                                                       "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not"}}
        self.description_constraint["weight"] = {"descr": "sum of weight of used items doesn't exceed max capacity"}
        weight = {}
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        x = {}
        start = []
        for item in list_item:
            i = item.index
            x[i] = self.model.add_var(var_type=BINARY,
                                      obj=item.value,
                                      name="x_" + str(i))
            if i in warm_start:
                start += [(x[i], warm_start[i])]
            weight[i] = item.weight
        self.model.start = start
        self.variable_decision["x"] = x
        self.model.update()
        self.constraints_dict["weight"] = self.model.add_constr(xsum([weight[i] * x[i] for i in x]) <= max_capacity,
                                                                name="capacity_constr")
        self.model.update()

    def retrieve_solutions(self, range_solutions: Iterable[int]):
        # nObjectives = S.NumObj
        solutions = []
        fits = []
        # x = S.getVars()
        for s in range_solutions:
            weight = 0
            xs = {}
            obj = self.model.objective_values[s]
            value_kp = 0
            for e in self.variable_decision["x"]:
                value = self.variable_decision["x"][e].xi(s)
                if value <= 0.1:
                    xs[e] = 0
                    continue
                xs[e] = 1
                weight += self.knapsack_model.index_to_item[e].weight
                value_kp += self.knapsack_model.index_to_item[e].value
            solutions += [KnapsackSolution(problem=self.knapsack_model,
                                           value=value_kp,
                                           weight=weight,
                                           list_taken=[xs[e] for e in sorted(xs)])]
            fits += [self.aggreg_sol(solutions[-1])]
            print(self.aggreg_sol)

        return ResultStorage(list_solution_fits=[(sol, fit) for sol, fit in zip(solutions, fits)],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_milp: ParametersMilp, **args):
        print("optimizing...")
        self.model.optimize(max_seconds=parameters_milp.TimeLimit,
                            max_solutions=parameters_milp.PoolSolutions)
        nSolutions = self.model.num_solutions
        objective = self.model.objective_value
        print('Solver found', nSolutions, 'solutions')
        print("Objective : ", objective)
        if parameters_milp.retrieve_all_solution:
            solutions = self.retrieve_solutions(list(range(nSolutions)))
        else:
            solutions = self.retrieve_solutions([0])
        return solutions

    def solve_lns(self,
                  parameter_gurobi: ParametersMilp,
                  init_solution: KnapsackSolution,
                  fraction_decision_fixed: float,
                  nb_iteration_max: int):
        current_solution = init_solution
        constraints = {}
        list_solutions = [current_solution]
        list_objective = [current_solution.value]
        objective = init_solution.value
        for k in trange(nb_iteration_max):
            for c in constraints:
                self.model.remove(constraints[c])
            self.add_init_solution(current_solution)
            fixed_variable = set(random.sample(self.variable_decision["x"].keys(),
                                               int(fraction_decision_fixed * len(self.variable_decision["x"]))))
            constraints = self.fix_decision(current_solution, fixed_variable)
            self.model.optimize(max_seconds=parameter_gurobi.TimeLimit,
                                max_solutions=parameter_gurobi.PoolSolutions)
            nSolutions = self.model.num_solutions
            objective = self.model.objective_value
            if parameter_gurobi.retrieve_all_solution:
                solutions = self.retrieve_solutions(list(range(nSolutions)))
            else:
                solutions = self.retrieve_solutions([0])
            current_solution = solutions.get_best_solution()
            list_solutions += [solutions.get_best_solution()]
            list_objective += [solutions.get_best_solution().value]
        print("Last obj : ", list_objective[-1])
        fig, ax = plt.subplots(1)
        ax.plot(list_objective)
        plt.show()

    def add_init_solution(self, init_solution: KnapsackSolution):
        start = []
        for i in self.variable_decision["x"]:
            start += [(self.variable_decision["x"][i], init_solution.list_taken[i])]
        self.model.start = start

    def fix_decision(self, init_solution: KnapsackSolution, fixed_variable_keys):
        constraints = {}
        for i in fixed_variable_keys:
            constraints[i] = self.model.add_constr(self.variable_decision["x"][i] == init_solution.list_taken[i])
        return constraints

    def describe_the_model(self):
        return str(self.description_variable_description) + "\n" + str(self.description_constraint)


class KnapsackORTools(SolverDO):
    def __init__(self, knapsack_model: KnapsackModel,
                 params_objective_function: ParamsObjectiveFunction):
        self.knapsack_model = knapsack_model
        self.model = None
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.knapsack_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **kwargs):
        solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        values = [item.value for item in list_item]
        weights = [[item.weight for item in list_item]]
        capacities = [max_capacity]
        solver.Init(values, weights, capacities)
        self.model = solver
    
    def solve(self, **kwargs):
        computed_value = self.model.Solve()
        print('Total value =', computed_value)
        xs = {}
        weight = 0
        value = 0
        for i in range(self.knapsack_model.nb_items):
            if self.model.BestSolutionContains(i):
                #packed_weights.append(self.knapsack_model.list_items[i].weight)
                weight += self.knapsack_model.list_items[i].weight
                value += self.knapsack_model.list_items[i].value
                xs[self.knapsack_model.list_items[i].index] = 1
            else:
                xs[self.knapsack_model.list_items[i].index] = 0
        sol = KnapsackSolution(problem=self.knapsack_model,
                               value=value,
                               weight=weight,
                               list_taken=[xs[e] for e in sorted(xs)])
        fit = self.aggreg_sol(sol)
        return ResultStorage(list_solution_fits=[(sol, fit)],
                             mode_optim=self.params_objective_function.sense_function)

