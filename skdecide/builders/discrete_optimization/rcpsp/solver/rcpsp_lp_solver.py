from itertools import product
from typing import List, Dict, Union

from mip import Model, xsum, BINARY, MINIMIZE, CBC, GRB, Var, INTEGER

from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ParamsObjectiveFunction, \
    ModeOptim, get_default_objective_setup, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolver
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import ParametersMilp
from skdecide.builders.discrete_optimization.generic_tools.mip.pymip_tools import MyModelMilp
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution, SingleModeRCPSPModel,\
    MultiModeRCPSPModel, RCPSPModelCalendar, PartialSolution
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_pile import PileSolverRCPSP, GreedyChoice, PileSolverRCPSP_Calendar
from enum import Enum
import random
import gurobi


class LP_RCPSP_Solver(Enum):
    GRB = 0
    CBC = 1


class LP_RCPSP(MilpSolver):
    def __init__(self, rcpsp_model: SingleModeRCPSPModel,
                 lp_solver=LP_RCPSP_Solver.CBC,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.model: Model = None
        self.lp_solver = CBC
        if lp_solver == LP_RCPSP_Solver.GRB:
            self.lp_solver = GRB
        elif lp_solver == LP_RCPSP_Solver.CBC:
            self.lp_solver = CBC
        self.variable_decision = {}
        self.constraints_dict = {}
        self.constraints_dict["lns"] = []
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.rcpsp_model,
                                                       params_objective_function=
                                                       params_objective_function)
        # self.description_variable_description = {}
        # self.description_constraint = {}

    def init_model(self,  **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        verbose = args.get("verbose", False)
        if start_solution is None:
            if greedy_start:
                if verbose:
                    print("Computing greedy solution")
                greedy_solver = PileSolverRCPSP(self.rcpsp_model)
                store_solution = greedy_solver.solve(greedy_choice=GreedyChoice.MOST_SUCCESSORS)
                self.start_solution = store_solution.get_best_solution_fit()[0]
                makespan = self.rcpsp_model.evaluate(self.start_solution)["makespan"]
            else:
                if verbose:
                    print("Get dummy solution")
                solution = self.rcpsp_model.get_dummy_solution()
                self.start_solution = solution
                makespan = self.rcpsp_model.evaluate(solution)["makespan"]
        else:
            self.start_solution = start_solution
            makespan = self.rcpsp_model.evaluate(start_solution)["makespan"]
        # p = [0, 3, 2, 5, 4, 2, 3, 4, 2, 4, 6, 0]
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        print(sorted_tasks)
        p = [int(self.rcpsp_model.mode_details[key][1]['duration'])
             for key in sorted_tasks]
        # print('p:', p)

        # u = [[0, 0], [5, 1], [0, 4], [1, 4], [1, 3], [3, 2], [3, 1], [2, 4],
        #      [4, 0], [5, 2], [2, 5], [0, 0]]
        u = []
        for task in sorted_tasks:
            tmp = []
            for r in self.rcpsp_model.resources.keys():
                tmp.append(self.rcpsp_model.mode_details[task][1][r])
            u.append(tmp)
        # print('u: ', u)

        # c = [6, 8]
        c = [x for x in self.rcpsp_model.resources.values()]
        # print('c: ', c)

        # S = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6],
        #      [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]]
        S = []
        print('successors: ', self.rcpsp_model.successors)
        for task in sorted_tasks:
            for suc in self.rcpsp_model.successors[task]:
                S.append([task-1, suc-1])
        # print('S: ', S)
        (R, self.J, self.T) = (range(len(c)), range(len(p)), range(sum(p)))
        # we have a better self.T to limit the number of variables :
        self.T = range(int(makespan+1))
        # model = Model()
        self.model = Model(sense=MINIMIZE,
                           solver_name=self.lp_solver)

        self.x: List[List[Var]] = [[self.model.add_var(name="x({},{})".format(j, t),
                                                       var_type=BINARY) for t in self.T]
                                   for j in self.J]

        self.model.objective = xsum(self.x[len(self.J) - 1][t] * t for t in self.T)

        for j in self.J:
            self.model += xsum(self.x[j][t] for t in self.T) == 1

        for (r, t) in product(R, self.T):
            self.model += (
                    xsum(u[j][r] * self.x[j][t2] for j in self.J for t2 in range(max(0, t - p[j] + 1), t + 1))
                    <= c[r])

        for (j, s) in S:
            self.model += xsum(t * self.x[s][t] - t * self.x[j][t] for t in self.T) >= p[j]
        start = []
        for j in self.J:
            for t in self.T:
                if self.start_solution.rcpsp_schedule[j+1]["start_time"] == t:
                    start += [(self.x[j][t], 1)]
                else:
                    start += [(self.x[j][t], 0)]

        self.model.start = start
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)
        self.constraints_partial_solutions = []
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    constraints += [self.model.add_constr(xsum([j*self.x[task-1][j]
                                                                for j in range(len(self.x[task-1]))]) == p_s.start_times[task])]
                    constraints += [self.model.add_constr(self.x[task-1][p_s.start_times[task]] == 1)]

            if p_s.partial_permutation is not None:
                for t1, t2 in zip(p_s.partial_permutation[:-1], p_s.partial_permutation[1:]):
                    constraints += [self.model.add_constr(xsum([t * self.x[t1-1][t]-t*self.x[t2-1][t]
                                                                for t in self.T]) <= 0)]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        constraints += [self.model.add_constr(xsum([t * self.x[t1-1][t]-t*self.x[t2-1][t]
                                                                   for t in self.T]) <= 0)]
            self.starts = {}
            for j in range(len(self.x)):
                self.starts[j] = self.model.add_var(name="start_" + str(j),
                                                    lb=0, ub=makespan)
                self.model.add_constr(xsum(t * self.x[j][t] for t in self.T) == self.starts[j])
            if p_s.start_at_end is not None:
                for i, j in p_s.start_at_end:
                    constraints += [self.model.add_constr(self.starts[j-1] == self.starts[i-1]+p[i-1])]
            if p_s.start_together is not None:
                for i, j in p_s.start_together:
                    constraints += [self.model.add_constr(self.starts[j-1] == self.starts[i-1])]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    constraints += [self.model.add_constr(self.starts[t2-1] >= self.starts[t1-1]+delta)]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    constraints += [self.model.add_constr(self.starts[t2-1] >= self.starts[t1-1]+delta+p[t1-1])]
            self.constraints_partial_solutions = constraints

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        retrieve_all_solution = parameters_milp.retrieve_all_solution
        nb_solutions_max = parameters_milp.n_solutions_max
        nb_solution = min(nb_solutions_max, self.model.num_solutions)
        if not retrieve_all_solution:
            nb_solution = 1
        list_solution_fits = []
        print(nb_solution, " solutions found")
        for s in range(nb_solution):
            rcpsp_schedule = {}
            objective = self.model.objective_values[s]
            for (j, t) in product(self.J, self.T):
                value = self.x[j][t].xi(s)
                if value >= 0.5:
                    rcpsp_schedule[j + 1] = {'start_time': t,
                                             'end_time': t + self.rcpsp_model.mode_details[j + 1][1]['duration']}
            print("Size schedule : ", len(rcpsp_schedule.keys()))
            try:
                solution = RCPSPSolution(problem=self.rcpsp_model,
                                         rcpsp_schedule=rcpsp_schedule,
                                         rcpsp_schedule_feasible=True)
                fit = self.aggreg_from_sol(solution)
                list_solution_fits += [(solution, fit)]
            except:
                print("Problem =", rcpsp_schedule, len(rcpsp_schedule))
                pass
        return ResultStorage(list_solution_fits=list_solution_fits,
                             best_solution=min(list_solution_fits,
                                               key=lambda x: x[1])[0],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_milp: ParametersMilp=ParametersMilp.default(), **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model()
        limit_time_s = parameters_milp.TimeLimit
        self.model.sol_pool_size = parameters_milp.PoolSolutions
        self.model.max_mip_gap_abs = parameters_milp.MIPGapAbs
        self.model.max_mip_gap = parameters_milp.MIPGap
        self.model.optimize(max_seconds=limit_time_s,
                            max_solutions=parameters_milp.n_solutions_max)
        return self.retrieve_solutions(parameters_milp)


class LP_MRCPSP(MilpSolver):
    def __init__(self,
                 rcpsp_model: MultiModeRCPSPModel,
                 lp_solver=LP_RCPSP_Solver.CBC,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.model: Model = None
        self.lp_solver = CBC
        if lp_solver == LP_RCPSP_Solver.GRB:
            self.lp_solver = GRB
        elif lp_solver == LP_RCPSP_Solver.CBC:
            self.lp_solver = CBC
        self.variable_decision = {}
        self.constraints_dict = {}
        self.constraints_dict["lns"] = []
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.rcpsp_model,
                                                       params_objective_function=
                                                       params_objective_function)
        # self.description_variable_description = {}
        # self.description_constraint = {}

    def init_model(self,  **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        verbose = args.get("verbose", False)
        if start_solution is None:
            if greedy_start:
                if verbose:
                    print("Computing greedy solution")
                greedy_solver = PileSolverRCPSP(self.rcpsp_model)
                store_solution = greedy_solver.solve(greedy_choice=GreedyChoice.MOST_SUCCESSORS)
                self.start_solution = store_solution.get_best_solution_fit()[0]
                makespan = self.rcpsp_model.evaluate(self.start_solution)["makespan"]
            else:
                if verbose:
                    print("Get dummy solution")
                solution = self.rcpsp_model.get_dummy_solution()
                self.start_solution = solution
                makespan = self.rcpsp_model.evaluate(solution)["makespan"]
        else:
            self.start_solution = start_solution
            makespan = self.rcpsp_model.evaluate(start_solution)["makespan"]

        # p = [0, 3, 2, 5, 4, 2, 3, 4, 2, 4, 6, 0]
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        p = [int(max([self.rcpsp_model.mode_details[key][mode]['duration']
                      for mode in self.rcpsp_model.mode_details[key]]))
             for key in sorted_tasks]
        # c = [6, 8]
        c = [x for x in self.rcpsp_model.resources.values()]
        renewable = {r: self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources
                     if r not in self.rcpsp_model.non_renewable_resources}
        non_renewable = {r: self.rcpsp_model.resources[r] for r in self.rcpsp_model.non_renewable_resources}
        # print('c: ', c)
        # S = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6],
        #      [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]]
        S = []
        print('successors: ', self.rcpsp_model.successors)
        for task in sorted_tasks:
           for suc in self.rcpsp_model.successors[task]:
               S.append([task, suc])
        # print('S: ', S)
        (R, self.J, self.T) = (range(len(c)), range(len(p)), range(sum(p)))
        # we have a better self.T to limit the number of variables :
        if self.start_solution.rcpsp_schedule_feasible:
            self.T = range(int(makespan+1))
        # model = Model()
        self.model = Model(sense=MINIMIZE,
                           solver_name=self.lp_solver)
        self.x: Dict[Var] = {}
        last_task = max(self.rcpsp_model.mode_details.keys())
        variable_per_task = {}
        for task in sorted_tasks:
            if task not in variable_per_task:
                variable_per_task[task] = []
            for mode in self.rcpsp_model.mode_details[task]:
                for t in self.T:
                    self.x[(task, mode, t)] = self.model.add_var(name="x({},{}, {})".format(task, mode, t),
                                                                 var_type=BINARY)
                    variable_per_task[task] += [(task, mode, t)]
        self.model.objective = xsum(self.x[key] * key[2] for key in variable_per_task[last_task])
        for j in variable_per_task:
            self.model += xsum(self.x[key] for key in variable_per_task[j]) == 1

        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            renewable_quantity = {r: renewable[r] for r in renewable}
        else:
            renewable_quantity = {r: [renewable[r]]*len(self.T) for r in renewable}

        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            non_renewable_quantity = {r: non_renewable[r] for r in non_renewable}
        else:
            non_renewable_quantity = {r: [non_renewable[r]]*len(self.T) for r in non_renewable}

        for (r, t) in product(renewable, self.T):
            self.model.add_constr(xsum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.x[key]
                                  for key in self.x
                                  if key[2] <= t < key[2]+int(self.rcpsp_model.mode_details[key[0]][key[1]]["duration"]))
                                   <= renewable_quantity[r][t])
            print(r, t)
        for r in non_renewable:
            self.model.add_constr(xsum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.x[key]
                                       for key in self.x) <= non_renewable_quantity[r][0])
        durations = {j: self.model.add_var(name="duration_"+str(j),
                                           var_type=INTEGER)
                     for j in variable_per_task}
        self.durations = durations
        self.variable_per_task = variable_per_task
        for j in variable_per_task:
            self.model.add_constr(xsum(self.rcpsp_model.mode_details[key[0]][key[1]]["duration"]*self.x[key]
                                       for key in variable_per_task[j]) == durations[j])
        for (j, s) in S:
            self.model.add_constr(xsum([key[2] * self.x[key] for key in variable_per_task[s]]
                                        + [- key[2] * self.x[key]
                                           for key in variable_per_task[j]]) >=
                                       durations[j])

        start = []
        for j in self.start_solution.rcpsp_schedule:
            start_time_j = self.start_solution.rcpsp_schedule[j]["start_time"]
            mode_j = 1 if j == 1 or j == self.rcpsp_model.n_jobs + 2 else self.start_solution.rcpsp_modes[j - 2]
            start += [(self.durations[j], self.rcpsp_model.mode_details[j][mode_j]["duration"])]
            for k in self.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == mode_j:
                    start += [(self.x[k], 1)]
                else:
                    start += [(self.x[k], 0)]
        self.model.start = start
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)
        self.constraints_partial_solutions = []
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    constraints += [self.model.add_constr(xsum([self.x[k] for k in self.variable_per_task[task]
                                                                if k[2] == p_s.start_times[task]]) == 1)]
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(p_s.partial_permutation[:-1], p_s.partial_permutation[1:]):
                    constraints += [self.model.add_constr(xsum([key[2] * self.x[key] for key in variable_per_task[t1]]
                                                               + [- key[2] * self.x[key]
                                                                  for key in variable_per_task[t2]]) <= 0)]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        constraints += [self.model.add_constr(xsum([key[2] * self.x[key] for key in variable_per_task[t1]]
                                                               + [- key[2] * self.x[key]
                                                                  for key in variable_per_task[t2]]) <= 0)]
            self.constraints_partial_solutions = constraints
            print('Partial solution constraints : ', self.constraints_partial_solutions)

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        retrieve_all_solution = parameters_milp.retrieve_all_solution
        nb_solutions_max = parameters_milp.n_solutions_max
        nb_solution = min(nb_solutions_max, self.model.num_solutions)
        if not retrieve_all_solution:
            nb_solution = 1
        list_solution_fits = []
        print(nb_solution, " solutions found")
        for s in range(nb_solution):
            rcpsp_schedule = {}
            modes = {}
            objective = self.model.objective_values[s]
            for (task, mode, t) in self.x:
                value = self.x[(task, mode, t)].xi(s)
                if value >= 0.5:
                    rcpsp_schedule[task] = {'start_time': t,
                                            'end_time': t + self.rcpsp_model.mode_details[task][mode]['duration']}
                    modes[task] = mode
            print("Size schedule : ", len(rcpsp_schedule.keys()))
            try:
                modes.pop(1)
                modes.pop(self.rcpsp_model.n_jobs+2)
                modes_vec = [modes[k] for k in sorted(modes)]
                solution = RCPSPSolution(problem=self.rcpsp_model,
                                         rcpsp_schedule=rcpsp_schedule,
                                         rcpsp_modes=modes_vec,
                                         rcpsp_schedule_feasible=True)
                fit = self.aggreg_from_sol(solution)
                list_solution_fits += [(solution, fit)]
            except:
                pass
        return ResultStorage(list_solution_fits=list_solution_fits,
                             best_solution=min(list_solution_fits,
                                               key=lambda x: x[1])[0],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_milp: ParametersMilp=ParametersMilp.default(), **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        limit_time_s = parameters_milp.TimeLimit
        self.model.sol_pool_size = parameters_milp.PoolSolutions
        self.model.max_mip_gap_abs = parameters_milp.MIPGapAbs
        self.model.max_mip_gap = parameters_milp.MIPGap
        self.model.optimize(max_seconds=limit_time_s,
                            max_solutions=parameters_milp.n_solutions_max)
        return self.retrieve_solutions(parameters_milp)


class LP_MRCPSP_GUROBI(MilpSolver):
    def __init__(self,
                 rcpsp_model: MultiModeRCPSPModel,
                 lp_solver=LP_RCPSP_Solver.CBC,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.model: gurobi.Model = None
        self.lp_solver = CBC
        if lp_solver == LP_RCPSP_Solver.GRB:
            self.lp_solver = GRB
        elif lp_solver == LP_RCPSP_Solver.CBC:
            self.lp_solver = CBC
        self.variable_decision = {}
        self.constraints_dict = {}
        self.constraints_dict["lns"] = []
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.rcpsp_model,
                                                       params_objective_function=
                                                       params_objective_function)
        # self.description_variable_description = {}
        # self.description_constraint = {}

    def init_model(self,  **args):
        greedy_start = args.get("greedy_start", True)
        start_solution = args.get("start_solution", None)
        max_horizon = args.get("max_horizon", None)
        verbose = args.get("verbose", False)
        if start_solution is None:
            if greedy_start:
                if verbose:
                    print("Computing greedy solution")
                if isinstance(self.rcpsp_model, RCPSPModelCalendar):
                    greedy_solver = PileSolverRCPSP_Calendar(self.rcpsp_model)
                else:
                    greedy_solver = PileSolverRCPSP(self.rcpsp_model)
                store_solution = greedy_solver.solve(greedy_choice=GreedyChoice.MOST_SUCCESSORS)
                self.start_solution = store_solution.get_best_solution_fit()[0]
                makespan = self.rcpsp_model.evaluate(self.start_solution)["makespan"]
            else:
                if verbose:
                    print("Get dummy solution")
                solution = self.rcpsp_model.get_dummy_solution()
                self.start_solution = solution
                makespan = self.rcpsp_model.evaluate(solution)["makespan"]
        else:
            self.start_solution = start_solution
            makespan = self.rcpsp_model.evaluate(start_solution)["makespan"]

        # p = [0, 3, 2, 5, 4, 2, 3, 4, 2, 4, 6, 0]
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        p = [int(max([self.rcpsp_model.mode_details[key][mode]['duration']
                      for mode in self.rcpsp_model.mode_details[key]]))
             for key in sorted_tasks]
        # c = [6, 8]
        c = [x for x in self.rcpsp_model.resources.values()]
        renewable = {r: self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources
                     if r not in self.rcpsp_model.non_renewable_resources}
        non_renewable = {r: self.rcpsp_model.resources[r] for r in self.rcpsp_model.non_renewable_resources}
        # print('c: ', c)
        # S = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6],
        #      [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]]
        S = []
        print('successors: ', self.rcpsp_model.successors)
        for task in sorted_tasks:
           for suc in self.rcpsp_model.successors[task]:
               S.append([task, suc])
        # print('S: ', S)
        (R, self.J, self.T) = (range(len(c)), range(len(p)), list(range(sum(p))))
        # we have a better self.T to limit the number of variables :
        if self.start_solution.rcpsp_schedule_feasible:
            self.T = list(range(int(makespan+1)))
        if max_horizon is not None:
            self.T = list(range(max_horizon+1))
            print("Hey")
            print(self.T)
        # model = Model()
        self.model = gurobi.Model("MRCPSP")
        self.x: Dict[gurobi.Var] = {}
        last_task = max(self.rcpsp_model.mode_details.keys())
        variable_per_task = {}
        keys_for_t = {}
        for task in sorted_tasks:
            if task not in variable_per_task:
                variable_per_task[task] = []
            for mode in self.rcpsp_model.mode_details[task]:
                for t in self.T:
                    self.x[(task, mode, t)] = self.model.addVar(name="x({},{}, {})".format(task, mode, t),
                                                                vtype=gurobi.GRB.BINARY)
                    for tt in range(t, t+self.rcpsp_model.mode_details[task][mode]["duration"]):
                        if tt not in keys_for_t:
                            keys_for_t[tt] = set()
                        keys_for_t[tt].add((task, mode, t))
                    variable_per_task[task] += [(task, mode, t)]
        self.model.update()
        self.model.setObjective(gurobi.quicksum(self.x[key] * key[2] for key in variable_per_task[last_task]))
        self.model.addConstrs(gurobi.quicksum(self.x[key] for key in variable_per_task[j]) == 1
                              for j in variable_per_task)

        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            renewable_quantity = {r: renewable[r] for r in renewable}
        else:
            renewable_quantity = {r: [renewable[r]]*len(self.T) for r in renewable}

        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            non_renewable_quantity = {r: non_renewable[r] for r in non_renewable}
        else:
            non_renewable_quantity = {r: [non_renewable[r]]*len(self.T) for r in non_renewable}

        #for r, t in product(renewable, self.T):
        #    self.model.addConstr(gurobi.quicksum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.x[key]
        #                                          for key in keys_for_t[t])
        #                          <= renewable_quantity[r][t])
        #    print(r, t)

        self.model.addConstrs(gurobi.quicksum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.x[key]
                                              for key in keys_for_t[t])
                                              <= renewable_quantity[r][t]
                              for (r,t) in product(renewable, self.T))

        self.model.addConstrs(gurobi.quicksum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.x[key]
                                       for key in self.x) <= non_renewable_quantity[r][0]
                              for r in non_renewable)
        self.model.update()
        durations = {j: self.model.addVar(name="duration_"+str(j),
                                          vtype=gurobi.GRB.INTEGER)
                     for j in variable_per_task}
        self.durations = durations
        self.variable_per_task = variable_per_task
        self.model.addConstrs(gurobi.quicksum(self.rcpsp_model.mode_details[key[0]][key[1]]["duration"]*self.x[key]
                                       for key in variable_per_task[j]) == durations[j]
                              for j  in variable_per_task)
        self.model.addConstrs(gurobi.quicksum([key[2] * self.x[key] for key in variable_per_task[s]]
                                              + [- key[2] * self.x[key]
                                                 for key in variable_per_task[j]]) >=
                              durations[j]
                              for (j, s) in S)

        start = []
        self.starts = {}
        for task in sorted_tasks:
            self.starts[task] = self.model.addVar(name="start({})".format(task),
                                                  vtype=gurobi.GRB.INTEGER,
                                                  lb=0, ub=self.T[-1])
            self.starts[task].start = self.start_solution.rcpsp_schedule[task]["start_time"]
            self.model.addConstr(gurobi.quicksum([self.x[key]*key[2] for key in variable_per_task[task]])
                                 == self.starts[task])
        for j in self.start_solution.rcpsp_schedule:
            start_time_j = self.start_solution.rcpsp_schedule[j]["start_time"]
            mode_j = 1 if j == 1 or j == self.rcpsp_model.n_jobs + 2 else self.start_solution.rcpsp_modes[j - 2]
            start += [(self.durations[j], self.rcpsp_model.mode_details[j][mode_j]["duration"])]
            for k in self.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == mode_j:
                    start += [(self.x[k], 1)]
                    self.x[k].start = 1
                else:
                    start += [(self.x[k], 0)]
                    self.x[k].start = 0

        # self.model.start = start
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)
        self.constraints_partial_solutions = []
        self.model.update()
        if p_s is not None:
            constraints = []
            if p_s.start_times is not None:
                constraints = self.model.addConstrs(gurobi.quicksum([self.x[k] for k in self.variable_per_task[task]
                                                                if k[2] == p_s.start_times[task]]) == 1
                                                    for task in p_s.start_times)
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(p_s.partial_permutation[:-1], p_s.partial_permutation[1:]):
                    constraints += [self.model.addConstr(gurobi.quicksum([key[2] * self.x[key] for key in variable_per_task[t1]]
                                                               + [- key[2] * self.x[key]
                                                                  for key in variable_per_task[t2]]) <= 0)]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        constraints += [self.model.addConstr(gurobi.quicksum([key[2] * self.x[key] for key in variable_per_task[t1]]
                                                             + [- key[2] * self.x[key]
                                                                for key in variable_per_task[t2]]) <= 0)]
            if p_s.start_at_end is not None:
                for i, j in p_s.start_at_end:
                    constraints += [self.model.addConstr(self.starts[j] == self.starts[i]+durations[i])]
            if p_s.start_together is not None:
                for i, j in p_s.start_together:
                    constraints += [self.model.addConstr(self.starts[j] == self.starts[i])]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    constraints += [self.model.addConstr(self.starts[t2] >= self.starts[t1]+delta)]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    constraints += [self.model.addConstr(self.starts[t2] >= self.starts[t1]+delta+durations[t1])]
            self.constraints_partial_solutions = constraints
            print('Partial solution constraints : ', self.constraints_partial_solutions)
            self.model.update()

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        retrieve_all_solution = parameters_milp.retrieve_all_solution
        nb_solutions_max = parameters_milp.n_solutions_max
        nb_solution = min(nb_solutions_max, self.model.SolCount)
        if not retrieve_all_solution:
            nb_solution = 1
        list_solution_fits = []
        for s in range(nb_solution):
            self.model.params.SolutionNumber = s
            rcpsp_schedule = {}
            modes = {}
            objective = self.model.getAttr("ObjVal")
            for (task, mode, t) in self.x:
                value = self.x[(task, mode, t)].getAttr('Xn')
                if value >= 0.5:
                    rcpsp_schedule[task] = {'start_time': t,
                                            'end_time': t + self.rcpsp_model.mode_details[task][mode]['duration']}
                    modes[task] = mode
            print("Size schedule : ", len(rcpsp_schedule.keys()))
            try:
                modes.pop(1)
                modes.pop(self.rcpsp_model.n_jobs+2)
                modes_vec = [modes[k] for k in sorted(modes)]
                solution = RCPSPSolution(problem=self.rcpsp_model,
                                         rcpsp_schedule=rcpsp_schedule,
                                         rcpsp_modes=modes_vec,
                                         rcpsp_schedule_feasible=True)
                fit = self.aggreg_from_sol(solution)
                list_solution_fits += [(solution, fit)]
            except:
                pass
        return ResultStorage(list_solution_fits=list_solution_fits,
                             best_solution=min(list_solution_fits,
                                               key=lambda x: x[1])[0],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_milp: ParametersMilp=ParametersMilp.default(), **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        self.model.setParam("TimeLimit", parameters_milp.TimeLimit)
        self.model.setParam("MIPGapAbs", parameters_milp.MIPGapAbs)
        self.model.setParam("PoolSolutions", parameters_milp.PoolSolutions)
        self.model.modelSense = gurobi.GRB.MINIMIZE
        self.model.optimize()
        return self.retrieve_solutions(parameters_milp)




