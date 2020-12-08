from typing import Dict, Tuple

from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction

from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, tree, MS_RCPSPSolution
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import SolverDO, MilpSolver, ParametersMilp, MilpSolverName, map_solver
from mip import Model, MINIMIZE, xsum, BINARY, INTEGER, Var, LinExpr, MAXIMIZE


class LP_Solver_MRSCPSP(MilpSolver):
    def __init__(self, rcpsp_model: MS_RCPSPModel,
                 lp_solver: MilpSolverName=MilpSolverName.CBC,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.model: Model = None
        self.lp_solver = lp_solver
        self.variable_decision = {}
        self.constraints_dict = {}
        self.constraints_dict["lns"] = []
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.rcpsp_model,
                                                       params_objective_function=
                                                       params_objective_function)
    #
    # def init_model(self, **args):
    #     self.model = Model(name="mrcpsp",
    #                        sense=MINIMIZE,
    #                        solver_name=map_solver[self.lp_solver])
    #     sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
    #     max_duration = sum([int(max([self.rcpsp_model.mode_details[key][mode]['duration']
    #                         for mode in self.rcpsp_model.mode_details[key]]))
    #                         for key in sorted_tasks])+10
    #
    #     # c = [6, 8]
    #     renewable = {r: self.rcpsp_model.resources_availability[r]
    #                  for r in self.rcpsp_model.resources_availability
    #                  if r not in self.rcpsp_model.non_renewable_resources}
    #     non_renewable = {r: self.rcpsp_model.resources_availability[r]
    #                      for r in self.rcpsp_model.non_renewable_resources}
    #     # print('c: ', c)
    #     # S = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6],
    #     #      [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]]
    #     list_edges = []
    #     print('successors: ', self.rcpsp_model.successors)
    #     for task in sorted_tasks:
    #         for suc in self.rcpsp_model.successors[task]:
    #             list_edges.append([task, suc])
    #     self.x: Dict[Tuple, Var] = {}
    #     self.employee_usage: Dict[Tuple, Var] = {}
    #     self.task_mode: Dict[Tuple, Var] = {}
    #     self.task_mode_without_t: Dict[Tuple, Var] = {}
    #     last_task = max(self.rcpsp_model.mode_details.keys())
    #     all_vars = set()
    #     all_vars_employee = set()
    #     variable_per_task = {}
    #     variable_per_task_mode = {}
    #     variable_per_employee = {}
    #     variable_per_employee_time = {}
    #     variable_employee_usage = {}
    #     tasks_only_ressource = [task for task in self.rcpsp_model.mode_details
    #                             if all([s not in self.rcpsp_model.skills_set
    #                                     or (s in self.rcpsp_model.skills_set
    #                                         and self.rcpsp_model.mode_details[task][mode][s] == 0)
    #                                     for mode in self.rcpsp_model.mode_details[task]
    #                                     for s in self.rcpsp_model.mode_details[task][mode]])]
    #     for employee in self.rcpsp_model.employees:
    #         skills_employee = [skill
    #                            for skill in self.rcpsp_model.employees[employee].dict_skill.keys()
    #                            if self.rcpsp_model.employees[employee].dict_skill[skill].skill_value > 0]
    #         for task in sorted_tasks:
    #             if task in tasks_only_ressource:
    #                 continue
    #             for mode in self.rcpsp_model.mode_details[task]:
    #                 required_skills = [s
    #                                    for s in self.rcpsp_model.mode_details[task][mode]
    #                                    if s in self.rcpsp_model.skills_set
    #                                    and self.rcpsp_model.mode_details[task][mode][s] > 0
    #                                    and s in skills_employee]
    #                 if len(required_skills) == 0:
    #                     # this employee will be useless anyway, pass
    #                     continue
    #                 for t in range(max_duration+1):
    #                     if (task, mode, t) not in self.task_mode:
    #                         if task not in variable_per_task_mode:
    #                             variable_per_task_mode[task] = []
    #                         self.task_mode[(task, mode, t)] = self.model.add_var(name="mode_{},{},{}".format(task,
    #                                                                                                          mode, t),
    #                                                                              var_type=BINARY)
    #                         variable_per_task_mode[task].append((task, mode, t))
    #                     self.x[(task,
    #                             employee,
    #                             mode,
    #                             t)] = self.model.add_var(name="x({},{},{},{})".format(task, employee, mode, t),
    #                                                      var_type=BINARY)
    #                     if task not in variable_per_task:
    #                         variable_per_task[task] = []
    #                     if employee not in variable_per_employee:
    #                         variable_per_employee[employee] = []
    #                     if employee not in variable_per_employee_time:
    #                         variable_per_employee_time[employee] = {}
    #                     if t not in variable_per_employee_time[employee]:
    #                         variable_per_employee_time[employee][t] = []
    #                     variable_per_task[task] += [(task, employee,
    #                                                  mode, t)]
    #                     all_vars.add((task, employee,
    #                                   mode, t))
    #                     variable_per_employee[employee] += [(task, employee,
    #                                                          mode, t)]
    #                     variable_per_employee_time[employee][t] += [(task,
    #                                                                  employee,
    #                                                                  mode,
    #                                                                  t)]
    #                 for s in required_skills:
    #                     if employee not in variable_employee_usage:
    #                         variable_employee_usage[employee] = []
    #                     variable_employee_usage[employee] += [(task, employee, s)]
    #                     all_vars_employee.add((task, employee, s))
    #                     self.employee_usage[(task, employee, s)] = \
    #                         self.model.add_var(name="usage({},{},{})".format(task, employee, s),
    #                                            var_type=BINARY)
    #
    #     employee = "fake-employee"
    #     for task in tasks_only_ressource:
    #         for mode in self.rcpsp_model.mode_details[task]:
    #             for t in range(max_duration + 1):
    #                 if (task, mode, t) not in self.task_mode:
    #                     if task not in variable_per_task_mode:
    #                         variable_per_task_mode[task] = []
    #                     self.task_mode[(task, mode, t)] = self.model.add_var(name="mode_{},{},{}".format(task, mode, t),
    #                                                                          var_type=BINARY)
    #                     variable_per_task_mode[task].append((task, mode, t))
    #                 self.x[(task, employee, mode, t)] = self.model.add_var(name="x({},{},{},{})"
    #                                                                        .format(task, employee, mode, t),
    #                                                                        var_type=BINARY)
    #                 if task not in variable_per_task:
    #                     variable_per_task[task] = []
    #                 if employee not in variable_per_employee:
    #                     variable_per_employee[employee] = []
    #                 if employee not in variable_per_employee_time:
    #                     variable_per_employee_time[employee] = {}
    #                 if t not in variable_per_employee_time[employee]:
    #                     variable_per_employee_time[employee][t] = []
    #                 variable_per_task[task] += [(task, employee,
    #                                              mode, t)]
    #                 variable_per_employee[employee] += [(task, employee,
    #                                                      mode, t)]
    #                 all_vars.add((task, employee,
    #                               mode, t))
    #                 variable_per_employee_time[employee][t] += [(task,
    #                                                              employee,
    #                                                              mode,
    #                                                              t)]
    #     for task, mode, t in self.task_mode:
    #         # all variables in self.x, with same task, mode, time
    #         var_of_interest = [v for v in all_vars if v[0] == task and v[2] == mode and v[3] == t]
    #         for v in var_of_interest:
    #             self.model.add_constr(self.task_mode[(task, mode, t)] >= self.x[v])
    #
    #     tm = set([(x[0], x[1]) for x in self.task_mode])
    #     for t, m in tm:
    #         self.task_mode_without_t[(t, m)] = self.model.add_var(name="mode_{},{}".format(t, m),
    #                                                               var_type=BINARY)
    #     for x in self.task_mode:
    #         self.model.add_constr(self.task_mode_without_t[(x[0], x[1])] >= self.task_mode[x])
    #     for t in sorted_tasks:
    #         self.model.add_constr(xsum(self.task_mode_without_t[x] for x in self.task_mode_without_t
    #                                    if x[0] == t) == 1)
    #
    #     # at least one mode !!!
    #     for task in sorted_tasks:
    #         vars_task_mode = [v for v in self.task_mode if v[0] == task]
    #         self.model.add_constr(xsum(self.task_mode[v] for v in vars_task_mode) >= 1)
    #
    #     self.durations = {j: self.model.add_var(name="duration_" + str(j),
    #                                             var_type=INTEGER)
    #                       for j in variable_per_task}
    #
    #     for task in variable_per_task_mode.keys():
    #         self.model.add_constr(xsum(self.task_mode_without_t[key]
    #                                    * self.rcpsp_model.mode_details[key[0]][key[1]]["duration"]
    #                                    for key in self.task_mode_without_t if key[0] == task)
    #                               == self.durations[task])
    #     for task in variable_per_task_mode.keys():
    #        self.model.add_constr(xsum(self.task_mode[key] for key in variable_per_task_mode[task])
    #                               == self.durations[task]+1)
    #     self.start_time_bin: Dict[Tuple, Var] = {}
    #     self.end_time_bin: Dict[Tuple, Var] = {}
    #
    #     self.start_time: Dict[Tuple, Var] = {}
    #     self.end_time: Dict[Tuple, Var] = {}
    #
    #     for task in variable_per_task:
    #         self.start_time[task] = self.model.add_var(name="start_{}".format(task),
    #                                                             var_type=INTEGER,
    #                                                             lb=0,
    #                                                             ub=max_duration)
    #         self.end_time[task] = self.model.add_var(name="end_{}".format(task),
    #                                                  var_type=INTEGER,
    #                                                  lb=0,
    #                                                  ub=max_duration+10)
    #         self.model.add_constr(self.start_time[task]+self.durations[task]-self.end_time[task] == 0)
    #         for variable in variable_per_task_mode[task]:
    #              time = variable[-1]
    #              # Setting starting date.
    #              self.model.add_constr(self.start_time[task] <= self.task_mode[variable]*time
    #                                    + (1-self.task_mode[variable])*10000)
    #
    #     # constraints on the employees activity :
    #     for employee in variable_per_employee:
    #         if employee == "fake-employee":
    #             # no constraint on this.
    #             continue
    #         for time in variable_per_employee_time[employee]:
    #             # the employee do only one stuff at a time.
    #             self.model.add_constr(xsum(self.x[variable]
    #                                        for variable in variable_per_employee_time[employee][time]) <= 1)
    #             self.model.add_constr(xsum(self.x[variable]
    #                                        for variable in variable_per_employee_time[employee][time])
    #                                   <= 1 if self.rcpsp_model.employees[employee].calendar_employee[time] else 0)
    #             # (makes redundant the precedent one)
    #
    #     for task in variable_per_task:
    #         for employee in variable_per_employee:
    #             if employee == "fake-employee":
    #                 continue
    #             variable_task_employee = [v for v in variable_per_employee[employee]
    #                                       if v[0] == task]
    #             if len(variable_task_employee) > 0:
    #                 variable_skills = [v for v in variable_employee_usage[employee] if v[0] == task]
    #                 skills_dict = {}
    #                 for v in variable_skills:
    #                     if v[2] not in skills_dict:
    #                         skills_dict[v[2]] = []
    #                     skills_dict[v[2]] += [v]
    #                 # for skill in skills_dict:
    #                 #    self.model.add_constr(xsum([self.x[v] for v in variable_task_employee])
    #                 #                          <= self.durations[task]) # Need to check that :/
    #                 for v in variable_task_employee:
    #                     self.model.add_constr(xsum([self.employee_usage[v]
    #                                                 for v in variable_skills]) >= self.x[v])
    #                 self.model.add_constr(xsum(self.x[v] for v in variable_task_employee)
    #                                       >= xsum([self.employee_usage[v]
    #                                                for v in variable_skills]))
    #     # constraints on ressource usage :
    #     from itertools import product
    #
    #     # for (r, t) in product(renewable, range(max_duration+1)):
    #     #     self.model += (xsum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.task_mode[key]
    #     #                         for key in self.task_mode)
    #     #                    <= renewable[r][t])
    #     #
    #     # for r in non_renewable:
    #     #     self.model.add_constr(xsum(int(self.rcpsp_model.mode_details[key[0]][key[1]][r]) * self.task_mode[key]
    #     #                                for key in self.task_mode) <= non_renewable[r][0])
    #
    #     # constraint on skill:
    #     done = set()
    #     for (task, mode, time) in self.task_mode:
    #         if (task, mode) not in done:
    #             skills = [s
    #                       for s in self.rcpsp_model.mode_details[task][mode]
    #                       if s in self.rcpsp_model.skills_set
    #                       and self.rcpsp_model.mode_details[task][mode][s]>0]
    #             print("Skilles : ", skills)
    #
    #             if len(skills) > 0:
    #                 for s in skills:
    #                     print("Needs : ", [self.rcpsp_model.mode_details[task][mode][s]
    #                                        for tmt in self.task_mode
    #                                        if tmt[0] == task and tmt[1] == mode])
    #                     variable = [v for v in all_vars_employee if v[0] == task and v[2] == s]
    #                     self.model.add_constr(xsum(self.employee_usage[v]
    #                                                * self.rcpsp_model.employees[v[1]].dict_skill[v[2]].skill_value
    #                                                for v in variable) >=
    #                                           xsum(self.task_mode[tmt]*self.rcpsp_model.mode_details[task][mode][s]
    #                                                for tmt in self.task_mode
    #                                                if tmt[0] == task and tmt[1] == mode))
    #         else:
    #             continue
    #         done.add((task, mode))
    #     # Precedence =
    #     for (j, s) in list_edges:
    #         self.model.add_constr(self.start_time[s]-self.end_time[j]
    #                               >= self.durations[j])
    #     self.model.objective = self.start_time[max(self.start_time)]

    def init_model(self, **args):
        self.model = Model(name="mrcpsp",
                           sense=MINIMIZE,
                           solver_name=map_solver[self.lp_solver])
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        max_duration = sum([int(max([self.rcpsp_model.mode_details[key][mode]['duration']
                            for mode in self.rcpsp_model.mode_details[key]]))
                            for key in sorted_tasks])+10
        # c = [6, 8]
        renewable = {r: self.rcpsp_model.resources_availability[r]
                     for r in self.rcpsp_model.resources_availability
                     if r not in self.rcpsp_model.non_renewable_resources}
        non_renewable = {r: self.rcpsp_model.resources_availability[r]
                         for r in self.rcpsp_model.non_renewable_resources}
        # print('c: ', c)
        # S = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6],
        #      [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]]
        list_edges = []
        print('successors: ', self.rcpsp_model.successors)
        for task in sorted_tasks:
            for suc in self.rcpsp_model.successors[task]:
                list_edges.append([task, suc])
        times = range(max_duration+1)
        self.modes = {task: {mode: self.model.add_var(name="mode_{},{}".format(task, mode),
                                                      var_type=BINARY)
                             for mode in self.rcpsp_model.mode_details[task]}
                      for task in self.rcpsp_model.mode_details}

        self.start_times = {task:
                                {mode:
                                    {t: self.model.add_var(name="start_{},{},{}".format(task, mode, t),
                                                           var_type=BINARY)
                                     for t in times}
                                 for mode in self.rcpsp_model.mode_details[task]}
                            for task in self.rcpsp_model.mode_details}
        # you have to choose one starting date :
        for task in self.start_times:
            self.model.add_constr(xsum(self.start_times[task][mode][t]
                                       for mode in self.start_times[task]
                                       for t in self.start_times[task][mode]) == 1)
            for mode in self.modes[task]:
                self.model.add_constr(self.modes[task][mode] == xsum(self.start_times[task][mode][t]
                                                                     for t in self.start_times[task][mode]))
        self.durations = {task: self.model.add_var(name="duration_" + str(task),
                                                   var_type=INTEGER)
                          for task in self.modes}
        self.start_times_task = {task: self.model.add_var(name="start_time_{}".format(task),
                                                          var_type=INTEGER)
                                 for task in self.start_times}
        self.end_times_task = {task: self.model.add_var(name="end_time_{}".format(task),
                                                        var_type=INTEGER)
                               for task in self.start_times}

        for task in self.start_times:
            self.model.add_constr(xsum(self.start_times[task][mode][t]*t
                                       for mode in self.start_times[task]
                                       for t in self.start_times[task][mode]) == self.start_times_task[task])
            self.model.add_constr(self.end_times_task[task]-self.start_times_task[task]-self.durations[task] == 0)

        for task in self.durations:
            self.model.add_constr(xsum(self.rcpsp_model.mode_details[task][mode]["duration"]
                                       * self.modes[task][mode]
                                       for mode in self.modes[task]) == self.durations[task])
        self.employee_usage = tree()
        task_in_employee_usage = set()
        for employee in self.rcpsp_model.employees:
            skills_employee = [skill
                               for skill in self.rcpsp_model.employees[employee].dict_skill.keys()
                               if self.rcpsp_model.employees[employee].dict_skill[skill].skill_value > 0]
            for task in sorted_tasks:
                for mode in self.rcpsp_model.mode_details[task]:
                    required_skills = [s
                                       for s in self.rcpsp_model.mode_details[task][mode]
                                       if s in self.rcpsp_model.skills_set
                                       and self.rcpsp_model.mode_details[task][mode][s] > 0
                                       and s in skills_employee]
                    if len(required_skills) == 0:
                        # this employee will be useless anyway, pass
                        continue
                    for s in required_skills:
                        for t in range(max_duration+1):
                            self.employee_usage[(employee, task, mode, t, s)] = \
                                self.model.add_var(name="employee_{}{}{}{}{}".format(employee, task, mode, t, s),
                                                   var_type=BINARY)
                            task_in_employee_usage.add(task)
                            self.model.add_constr(self.employee_usage[(employee, task, mode, t, s)]
                                                  - self.modes[task][mode] <= 0)
                            self.model.add_constr(self.employee_usage[(employee, task, mode, t, s)]
                                                  - self.start_times[task][mode][t] <= 0)
                            if any(not self.rcpsp_model.employees[employee].calendar_employee[tt]
                                   for tt in range(t, t+self.rcpsp_model.mode_details[task][mode]["duration"])):
                                # print((employee, task, mode, t, s), "will be 0")
                                # print(self.rcpsp_model.employees[employee].calendar_employee)
                                self.model.add_constr(self.employee_usage[(employee, task, mode, t, s)] == 0)
        employees = set([x[0] for x in self.employee_usage])
        # for emp in employees:
        #     all = [x for x in self.employee_usage if x[0] == emp]
        #     for x1 in all:
        #         for x2 in all:
        #             if x1[0] == x2[0]:
        #                 self.model.add_constr(self.employee_usage[x1]+self.employee_usage[x2] <= 1)
        #             if x1[0] != x2[0]:
        #                 if x1[3]+self.rcpsp_model.mode_details[x1[0]][x1[2]]["duration"]>=x2[3]:
        #                     self.model.add_constr(self.employee_usage[x1] + self.employee_usage[x2] <= 1)
        #     print("Employee ", emp)
        from itertools import product

        # can't work on overlapping tasks.
        for emp, t in product(employees, times):
            self.model.add_constr(xsum(self.employee_usage[x]
                                       for x in self.employee_usage
                                       if x[0] == emp and
                                       x[3] <= t < x[3]+int(self.rcpsp_model.mode_details[x[1]][x[2]]["duration"])) <= 1)
        # ressource usage limit
        for (r, t) in product(renewable, times):
            self.model.add_constr(xsum(int(self.rcpsp_model.mode_details[task][mode][r])
                                           * self.start_times[task][mode][time]
                                           for task in self.start_times
                                           for mode in self.start_times[task]
                                           for time in self.start_times[task][mode]
                                           if time <= t < time+int(self.rcpsp_model.mode_details[task][mode]["duration"]))
                                  <= renewable[r][t])
        # for non renewable ones.
        for r in non_renewable:
            self.model.add_constr(xsum(int(self.rcpsp_model.mode_details[task][mode][r])
                                       * self.start_times[task][mode][time]
                                       for task in self.start_times
                                       for mode in self.start_times[task]
                                       for time in self.start_times[task][mode]) <= non_renewable[r][0])
        for task in self.start_times_task:
            required_skills = [(s,
                                mode,
                                self.rcpsp_model.mode_details[task][mode][s])
                               for mode in self.rcpsp_model.mode_details[task]
                               for s in self.rcpsp_model.mode_details[task][mode]
                               if s in self.rcpsp_model.skills_set
                               and self.rcpsp_model.mode_details[task][mode][s] > 0]
            skills = set([s[0] for s in required_skills])
            for s in skills:
                employee_usage_keys = [v for v in self.employee_usage if v[1] == task and v[4] == s]
                self.model.add_constr(xsum(self.employee_usage[x]*self.rcpsp_model.employees[x[0]].dict_skill[s].skill_value
                                           for x in employee_usage_keys)
                                      >= xsum(self.modes[task][mode]*self.rcpsp_model.mode_details[task][mode].get(s, 0)
                                              for mode in self.modes[task]))
        for (j, s) in list_edges:
            self.model.add_constr(self.start_times_task[s]-self.end_times_task[j]
                                  >= 0)
        self.model.objective = self.start_times_task[max(self.start_times_task)]

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
            results = {}
            employee_usage = {}
            employee_usage_solution = {}
            for task in self.start_times:
                for mode in self.start_times[task]:
                    for t in self.start_times[task][mode]:
                        value = self.start_times[task][mode][t].xi(s)
                        results[(task, mode, t)] = value
                        if value >= 0.5:
                            rcpsp_schedule[task] = {'start_time': int(t),
                                                    'end_time': int(t
                                                                    + self.rcpsp_model.mode_details[task][mode]['duration'])}
                            modes[task] = mode
            for t in self.employee_usage:
                employee_usage[t] = self.employee_usage[t].xi(s)
                if employee_usage[t] >= 0.5:
                    if t[1] not in employee_usage_solution:
                        employee_usage_solution[t[1]] = {}
                    if t[0] not in employee_usage_solution[t[1]]:
                        employee_usage_solution[t[1]][t[0]] = set()
                    employee_usage_solution[t[1]][t[0]].add(t[4])
                    # (employee, task, mode, time, skill)

            modes = {}
            modes_task = {}
            for t in self.modes:
                for m in self.modes[t]:
                    modes[(t, m)] = self.modes[t][m].xi(s)
                    if modes[(t, m)] >= 0.5:
                        modes_task[t] = m
            durations = {}
            for t in self.durations:
                durations[t] = self.durations[t].xi(s)
            start_time = {}
            for t in self.start_times_task:
                start_time[t] = self.start_times_task[t].xi(s)
            end_time = {}
            for t in self.start_times_task:
                end_time[t] = self.end_times_task[t].xi(s)
            print("Size schedule : ", len(rcpsp_schedule.keys()))
            print("results", "(task, mode, time)",
                  {x: results[x] for x in results if results[x] == 1.})
            print("Employee usage : ",
                  "(employee, task, mode, time, skill)",
                  {x: employee_usage[x] for x in employee_usage if employee_usage[x] == 1.})
            print("task mode : ", "(task, mode)",
                  {t: modes[t]
                   for t in modes
                   if modes[t] == 1.})
            print("durations : ", durations)
            print("Start time ", start_time)
            print("End time ", end_time)
            solution = MS_RCPSPSolution(problem=self.rcpsp_model,
                                        modes=modes_task,
                                        schedule=rcpsp_schedule,
                                        employee_usage=employee_usage_solution)
            fit = self.aggreg_from_sol(solution)
            list_solution_fits += [(solution, fit)]
        return ResultStorage(list_solution_fits=list_solution_fits,
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_milp: ParametersMilp, **args) -> ResultStorage:
        if self.model is None:
            import time
            print("Init LP model ")
            t = time.time()
            self.init_model(greedy_start=False)
            print("LP model initialized...in ", time.time()-t, " seconds")
        limit_time_s = parameters_milp.TimeLimit
        self.model.sol_pool_size = parameters_milp.PoolSolutions
        self.model.max_mip_gap_abs = parameters_milp.MIPGapAbs
        self.model.max_mip_gap = parameters_milp.MIPGap
        self.model.optimize(max_seconds=limit_time_s,
                            max_solutions=parameters_milp.n_solutions_max)
        return self.retrieve_solutions(parameters_milp)
