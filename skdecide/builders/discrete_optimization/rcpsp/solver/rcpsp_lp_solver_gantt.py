from itertools import product
from typing import List, Dict, Union, Tuple, Set
from mip import Model, xsum, BINARY, MINIMIZE, CBC, GRB, Var, INTEGER
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ParamsObjectiveFunction, \
    ModeOptim, get_default_objective_setup, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolver, map_solver
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import ParametersMilp, MilpSolverName, MilpSolver
from skdecide.builders.discrete_optimization.generic_tools.mip.pymip_tools import MyModelMilp
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution, SingleModeRCPSPModel,\
    MultiModeRCPSPModel, RCPSPModelCalendar, PartialSolution
from skdecide.builders.discrete_optimization.rcpsp.solver.rcpsp_pile import PileSolverRCPSP, GreedyChoice
from enum import Enum
import random
import networkx as nx

import gurobi as gurobi

# from gurobi import LinExpr, Model, GRB, quicksum

# TODO : modelize the optimisation problem behind this.


def intersect(i1, i2):
    if (i2[0] >= i1[1]
         or i1[0] >= i2[1]):
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


class ConstraintTaskIndividual:
    list_tuple: List[Tuple[str, int, int, bool]]
    # task, ressource, ressource_individual, has or has not to do a task
    # indicates constraint for a given resource individual that has to do a tas
    def __init__(self, list_tuple):
        self.list_tuple = list_tuple


class ConstraintWorkDuration:
    ressource: str
    individual: int
    time_bounds: Tuple[int, int]
    working_time_upper_bound: int

    def __init__(self, ressource, individual, time_bounds, working_time_upper_bound):
        self.ressource = ressource
        self.individual = individual
        self.time_bounds = time_bounds
        self.working_time_upper_bound = working_time_upper_bound


class LP_MRCPSP_GANTT(MilpSolver):
    def __init__(self,
                 rcpsp_model: RCPSPModelCalendar,
                 rcpsp_solution: RCPSPSolution,
                 lp_solver=MilpSolverName.CBC,
                 **kwargs):
        self.rcpsp_model = rcpsp_model
        self.lp_solver = lp_solver
        self.rcpsp_solution = rcpsp_solution
        self.jobs = sorted(list(self.rcpsp_model.mode_details.keys()))
        self.modes_dict = {i+2: self.rcpsp_solution.rcpsp_modes[i] for i in range(len(self.rcpsp_solution.rcpsp_modes))}
        self.modes_dict[1] = 1
        self.modes_dict[self.jobs[-1]] = 1
        self.rcpsp_schedule = self.rcpsp_solution.rcpsp_schedule
        #self.set_start_times = set(self.rcpsp_schedule.values())
        self.start_times_dict = {}
        for task in self.rcpsp_schedule:
            t = self.rcpsp_schedule[task]["start_time"]
            if t not in self.start_times_dict:
                self.start_times_dict[t] = set()
            self.start_times_dict[t].add((task, t))
        self.graph_intersection_time = nx.Graph()
        for t in self.jobs:
            self.graph_intersection_time.add_node(t)
        for t in self.jobs:
            intersected_jobs = [task for task in self.rcpsp_schedule
                                if intersect([self.rcpsp_schedule[task]["start_time"],
                                              self.rcpsp_schedule[task]["end_time"]],
                                             [self.rcpsp_schedule[t]["start_time"],
                                              self.rcpsp_schedule[t]["end_time"]]) is not None
                                and t != task]
            for tt in intersected_jobs:
                self.graph_intersection_time.add_edge(t, tt)
        cliques = [c for c in nx.find_cliques(self.graph_intersection_time)]
        self.cliques = cliques

    def init_model(self, **args):
        self.model = Model(sense=MINIMIZE,
                           solver_name=map_solver[self.lp_solver])
        self.ressource_id_usage = {k: {i: {} for i in range(len(self.rcpsp_model.calendar_details[k]))}
                                   for k in self.rcpsp_model.calendar_details.keys()}
        variables_per_task = {}
        variables_per_individual = {}
        constraints_ressource_need = {}

        for task in self.jobs:
            start = self.rcpsp_schedule[task]["start_time"]
            end = self.rcpsp_schedule[task]["end_time"]
            for k in self.ressource_id_usage:  # typically worker
                needed_ressource = self.rcpsp_model.mode_details[task][self.modes_dict[task]][k] > 0
                if needed_ressource:
                    for individual in self.ressource_id_usage[k]:
                        available = all([self.rcpsp_model.calendar_details[k][individual][time]
                                         for time in range(start, end)])
                        if available:
                            key_variable = (k, individual, task)
                            self.ressource_id_usage[k][individual][task] = self.model.add_var(name=str(key_variable),
                                                                                              var_type=BINARY,
                                                                                              obj=random.random())
                            if task not in variables_per_task:
                                variables_per_task[task] = set()
                            if k not in variables_per_individual:
                                variables_per_individual[k] = {}
                            if individual not in variables_per_individual[k]:
                                variables_per_individual[k][individual] = set()
                            variables_per_task[task].add(key_variable)
                            variables_per_individual[k][individual].add(key_variable)
                    ressource_needed = self.rcpsp_model.mode_details[task][self.modes_dict[task]][k]
                    if k not in constraints_ressource_need:
                        constraints_ressource_need[k] = {}
                    constraints_ressource_need[k][task] = self.model.add_constr(xsum([self.ressource_id_usage[k]
                                                                                      [key[1]]
                                                                                      [key[2]]
                                                                                     for key in variables_per_task[task]
                                                                                     if key[0] == k])
                                                                                == ressource_needed, name="ressource_"
                                                                                                          + str(k)
                                                                                                          + "_"
                                                                                                          + str(task))
        overlaps_constraints = {}

        for i in range(len(self.cliques)):
            tasks = set(self.cliques[i])
            for k in variables_per_individual:
                for individual in variables_per_individual[k]:
                    keys_variable = [variable
                                     for variable in variables_per_individual[k][individual]
                                     if variable[2] in tasks]
                    if len(keys_variable) > 0:
                        overlaps_constraints[(i, k, individual)] = \
                            self.model.add_constr(xsum([self.ressource_id_usage[key[0]][key[1]][key[2]]
                                                        for key in keys_variable]) <= 1)

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
            resource_id_usage = {k: {individual:
                                         {task: self.ressource_id_usage[k][individual][task].xi(s)
                                          for task in self.ressource_id_usage[k][individual]}
                                     for individual in self.ressource_id_usage[k]
                                     }
                                 for k in self.ressource_id_usage}
            print(resource_id_usage)

    def solve(self, parameters_milp: ParametersMilp=ParametersMilp.default(),
              **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        limit_time_s = parameters_milp.TimeLimit
        self.model.sol_pool_size = parameters_milp.PoolSolutions
        self.model.max_mip_gap_abs = parameters_milp.MIPGapAbs
        self.model.max_mip_gap = parameters_milp.MIPGap
        self.model.optimize(max_seconds=limit_time_s,
                            max_solutions=parameters_milp.n_solutions_max)
        return self.retrieve_solutions(parameters_milp)

# gurobi solver which is ussefull to get a pool of solution (indeed, using the other one we dont have usually a lot of
# ssolution since we converge rapidly to the "optimum" (we don't have an objective value..)
class LP_MRCPSP_GANTT_GUROBI(MilpSolver):
    def __init__(self,
                 rcpsp_model: RCPSPModelCalendar,
                 rcpsp_solution: RCPSPSolution,
                 lp_solver=MilpSolverName.CBC,
                 **kwargs):
        self.rcpsp_model = rcpsp_model
        self.lp_solver = lp_solver
        self.rcpsp_solution = rcpsp_solution
        self.jobs = sorted(list(self.rcpsp_model.mode_details.keys()))
        self.modes_dict = {i+2: self.rcpsp_solution.rcpsp_modes[i] for i in range(len(self.rcpsp_solution.rcpsp_modes))}
        self.modes_dict[1] = 1
        self.modes_dict[self.jobs[-1]] = 1
        self.rcpsp_schedule = self.rcpsp_solution.rcpsp_schedule
        #self.set_start_times = set(self.rcpsp_schedule.values())
        self.start_times_dict = {}
        for task in self.rcpsp_schedule:
            t = self.rcpsp_schedule[task]["start_time"]
            if t not in self.start_times_dict:
                self.start_times_dict[t] = set()
            self.start_times_dict[t].add((task, t))
        self.graph_intersection_time = nx.Graph()
        for t in self.jobs:
            self.graph_intersection_time.add_node(t)
        for t in self.jobs:
            intersected_jobs = [task for task in self.rcpsp_schedule
                                if intersect([self.rcpsp_schedule[task]["start_time"],
                                              self.rcpsp_schedule[task]["end_time"]],
                                             [self.rcpsp_schedule[t]["start_time"],
                                              self.rcpsp_schedule[t]["end_time"]]) is not None
                                and t != task]
            for tt in intersected_jobs:
                self.graph_intersection_time.add_edge(t, tt)
        cliques = [c for c in nx.find_cliques(self.graph_intersection_time)]
        self.cliques = cliques
        self.constraint_additionnal = {}

    def init_model(self, **args):
        self.model = gurobi.Model("Gantt")
        self.ressource_id_usage = {k: {i: {} for i in range(len(self.rcpsp_model.calendar_details[k]))}
                                   for k in self.rcpsp_model.calendar_details.keys()}
        variables_per_task = {}
        variables_per_individual = {}
        constraints_ressource_need = {}

        for task in self.jobs:
            start = self.rcpsp_schedule[task]["start_time"]
            end = self.rcpsp_schedule[task]["end_time"]
            for k in self.ressource_id_usage:  # typically worker
                needed_ressource = self.rcpsp_model.mode_details[task][self.modes_dict[task]][k] > 0
                if needed_ressource:
                    for individual in self.ressource_id_usage[k]:
                        available = all([self.rcpsp_model.calendar_details[k][individual][time]
                                         for time in range(start, end)])
                        if available:
                            key_variable = (k, individual, task)
                            self.ressource_id_usage[k][individual][task] = self.model.addVar(name=str(key_variable),
                                                                                             vtype=gurobi.GRB.BINARY)
                            if task not in variables_per_task:
                                variables_per_task[task] = set()
                            if k not in variables_per_individual:
                                variables_per_individual[k] = {}
                            if individual not in variables_per_individual[k]:
                                variables_per_individual[k][individual] = set()
                            variables_per_task[task].add(key_variable)
                            variables_per_individual[k][individual].add(key_variable)
                    ressource_needed = self.rcpsp_model.mode_details[task][self.modes_dict[task]][k]
                    if k not in constraints_ressource_need:
                        constraints_ressource_need[k] = {}
                    constraints_ressource_need[k][task] = self.model.addConstr(gurobi.quicksum([self.ressource_id_usage[k]
                                                                                                [key[1]]
                                                                                                [key[2]]
                                                                                                for key in variables_per_task[task]
                                                                                                if key[0] == k])
                                                                                == ressource_needed, name="ressource_"
                                                                                                          + str(k)
                                                                                                          + "_"
                                                                                                          + str(task))
        overlaps_constraints = {}

        for i in range(len(self.cliques)):
            tasks = set(self.cliques[i])
            for k in variables_per_individual:
                for individual in variables_per_individual[k]:
                    keys_variable = [variable
                                     for variable in variables_per_individual[k][individual]
                                     if variable[2] in tasks]
                    if len(keys_variable) > 0:
                        overlaps_constraints[(i, k, individual)] = \
                            self.model.addConstr(gurobi.quicksum([self.ressource_id_usage[key[0]][key[1]][key[2]]
                                                                  for key in keys_variable]) <= 1)

    def adding_constraint(self, constraint_description: Union[ConstraintTaskIndividual, ConstraintWorkDuration],
                          constraint_name: str=""):
        if isinstance(constraint_description, ConstraintTaskIndividual):
            if constraint_name == "":
                constraint_name = str(ConstraintTaskIndividual.__name__)
            for tupl in constraint_description.list_tuple:
                ressource, ressource_individual, task, has_to_do = tupl
                if ressource in self.ressource_id_usage:
                    if ressource_individual in self.ressource_id_usage[ressource]:
                        if task in self.ressource_id_usage[ressource][ressource_individual]:
                            if constraint_name not in self.constraint_additionnal:
                                self.constraint_additionnal[constraint_name] = []
                            self.constraint_additionnal[constraint_name] += [self.model
                                                                                 .addConstr(self.ressource_id_usage[ressource][ressource_individual][task] == has_to_do)]
        if isinstance(constraint_description, ConstraintWorkDuration):
            if constraint_name == "":
                constraint_name = str(ConstraintWorkDuration.__name__)
            if constraint_name not in self.constraint_additionnal:
                self.constraint_additionnal[constraint_name] = []
            tasks_of_interest = [t for t in self.rcpsp_schedule
                                 if t in self.ressource_id_usage.get(constraint_description.ressource, {})
                                                                .get(constraint_description.individual, {}) and
                                 (constraint_description.time_bounds[0] <= self.rcpsp_schedule[t]["start_time"]
                                  <= constraint_description.time_bounds[1]
                                  or constraint_description.time_bounds[0] <= self.rcpsp_schedule[t]["end_time"]
                                  <= constraint_description.time_bounds[1])]
            print(tasks_of_interest)
            self.constraint_additionnal[constraint_name] += \
                [self.model.addConstr(gurobi.quicksum([self.ressource_id_usage[constraint_description.ressource]
                                                       [constraint_description.individual][t] *
                                                       (min(constraint_description.time_bounds[1],
                                                            self.rcpsp_schedule[t]['end_time'])
                                                        - max(constraint_description.time_bounds[0],
                                                              self.rcpsp_schedule[t]["start_time"]))
                                                       for t in tasks_of_interest]) <=
                                      constraint_description.working_time_upper_bound)]
            self.model.update()


    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        retrieve_all_solution = parameters_milp.retrieve_all_solution
        nb_solutions_max = parameters_milp.n_solutions_max
        nb_solution = self.model.getAttr("SolCount")
        print(nb_solution, " solutions found")
        solutions = []
        for s in range(nb_solution):
            self.model.params.SolutionNumber = s
            objective = self.model.getAttr("poolObjVal")
            print("Objective : ", objective)
            solutions += [{k: {individual:
                                   {task: self.ressource_id_usage[k][individual][task].getAttr('Xn')
                                    for task in self.ressource_id_usage[k][individual]}
                               for individual in self.ressource_id_usage[k]
                               }
                           for k in self.ressource_id_usage}]
        return solutions

    def solve(self, parameters_milp: ParametersMilp=ParametersMilp.default(),
              **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model(greedy_start=False, **kwargs)
        self.model.modelSense = kwargs.get("sense", gurobi.GRB.MINIMIZE)
        self.model.setParam(gurobi.GRB.Param.PoolSolutions, parameters_milp.PoolSolutions)
        self.model.setParam("MIPGapAbs", parameters_milp.MIPGapAbs)
        self.model.setParam("MIPGap", parameters_milp.MIPGap)
        self.model.setParam("TimeLimit", parameters_milp.TimeLimit)
        self.model.setParam("PoolSearchMode", parameters_milp.pool_search_mode)
        self.model.optimize()
        return self.retrieve_solutions(parameters_milp)

    def build_objective_function_from_a_solution(self,
                                                 ressource_usage: Dict[str,
                                                                       Dict[int, Dict[int, bool]]],
                                                 ignore_tuple: Set[Tuple[str, int, int]]=None):
        objective = gurobi.LinExpr(0.)
        if ignore_tuple is None:
            ignore_tuple = set()
        for k in ressource_usage:
            for individual in ressource_usage[k]:
                for task in ressource_usage[k][individual]:
                    if (k, individual, task) in ignore_tuple:
                        pass
                    if ressource_usage[k][individual][task] >= 0.5:
                        objective.add(1-self.ressource_id_usage[k][individual][task])
        print("Setting new objectives = Change task objective")
        self.model.setObjective(objective)











