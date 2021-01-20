from dataclasses import InitVar
from typing import Union

from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP, CPSolverName, map_cp_solver_name
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ObjectiveHandling, \
    ParamsObjectiveFunction, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution,\
    SingleModeRCPSPModel, MultiModeRCPSPModel, \
    RCPSPModelCalendar, PartialSolution

from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel,\
    MS_RCPSPSolution, MS_RCPSPSolution_Variant
from minizinc import Instance, Model, Solver, Status, Result
from datetime import timedelta
import os, random
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {"multi-calendar": os.path.join(this_path, "../minizinc/ms_rcpsp_multi_mode_mzn_calendar.mzn"),
             "multi-calendar-no-ressource": os.path.join(this_path,
                                                         "../minizinc/ms_rcpsp_multi_mode_mzn_calendar_no_ressource.mzn")}


class MS_RCPSPSolCP:
    objective: int
    __output_item: InitVar[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        print("One solution ", self.objective)

    def check(self) -> bool:
        return True


class CP_MS_MRCPSP_MZN(CPSolver):
    def __init__(self,
                 rcpsp_model: MS_RCPSPModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.instance: Instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["start", "mrun"]  # For now, I've put the var names of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)
        self.calendar = True
        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            self.calendar = True
        self.one_ressource_per_task = kwargs.get('one_ressource_per_task', False)
        self.resources_index = None

    def init_model(self, **args):
        no_ressource = len(self.rcpsp_model.resources_list) == 0
        # model_type = "multi-calendar" if not no_ressource else "multi-calendar-no-ressource"
        model_type = "multi-calendar"
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type", False)
        exact_skills_need = args.get("exact_skills_need", True)
        if custom_output_type:
            model.output_type = MS_RCPSPSolCP
            self.custom_output_type = True

        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        # solver = Solver.lookup("")
        resources_list = sorted(list(self.rcpsp_model.resources_availability.keys()))
        self.resources_index = resources_list
        instance = Instance(solver, model)
        n_res = len(resources_list)
        # print('n_res: ', n_res)
        keys = []
        if not no_ressource:
            instance["n_res"] = n_res
            keys += ["n_res"]
        if model_type == "multi-calendar":
            instance["exact_skills_need"] = exact_skills_need
            keys += ["exact_skills_need"]
            instance["add_calendar_constraint_unit"] = args.get("add_calendar_constraint_unit", True)
            keys += ["add_calendar_constraint_unit"]
        instance["exact_skills_need"] = exact_skills_need
        instance["one_ressource_per_task"] = self.one_ressource_per_task
        keys += ["one_ressource_per_task"]
        # rc = [val for val in self.rcpsp_model.resources.values()]
        # # print('rc: ', rc)
        # instance["rc"] = rc
        n_tasks = self.rcpsp_model.n_jobs_non_dummy + 2
        # print('n_tasks: ', n_tasks)
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        # print('mode_details: ', self.rcpsp_model.mode_details)
        n_opt = sum([len(list(self.rcpsp_model.mode_details[key].keys())) for key in sorted_tasks])
        # print('n_opt: ', n_opt)
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
        modes = []
        dur = []
        counter = 0
        self.modeindex_map = {}
        for act in sorted_tasks:
            tmp = list(sorted(self.rcpsp_model.mode_details[act].keys()))
            # tmp = [counter + x for x in tmp]
            for i in range(len(tmp)):
                original_mode_index = tmp[i]
                mod_index = counter+tmp[i]
                tmp[i] = mod_index
                self.modeindex_map[mod_index] = {'task': act, 'original_mode_index': original_mode_index}
            modes.append(set(tmp))
            counter = tmp[-1]
            dur = dur + [self.rcpsp_model.mode_details[act][key]['duration']
                         for key in sorted(self.rcpsp_model.mode_details[act].keys())]

        # print('modes: ', modes)
        instance['modes'] = modes
        keys += ["modes"]
        # print('dur: ', dur)
        instance['dur'] = dur
        keys += ["dur"]

        if not no_ressource:
            rreq = []
            index = 0
            for res in resources_list:
                rreq.append([])
                for task in sorted_tasks:
                    for mod in sorted(self.rcpsp_model.mode_details[task].keys()):
                        rreq[index].append(int(self.rcpsp_model.mode_details[task][mod][res]))
                index += 1

        skills_set = sorted(list(self.rcpsp_model.skills_set))
        nb_units = len(self.rcpsp_model.employees)
        skill_required = []
        index = 0
        for skill in skills_set:
            skill_required.append([])
            for task in sorted_tasks:
                for mod in sorted(self.rcpsp_model.mode_details[task].keys()):
                    skill_required[index].append(int(self.rcpsp_model.mode_details[task][mod].get(skill, 0)))
            index += 1
        if True:
            if no_ressource:
                instance["max_time"] = self.rcpsp_model.horizon
                keys += ["max_time"]
            else:
                one_ressource = list(self.rcpsp_model.resources_availability.keys())[0]
                instance["max_time"] = min(len(self.rcpsp_model.resources_availability[one_ressource])-1,
                                               self.rcpsp_model.horizon)
                # instance["max_time"] = 2842
                keys += ["max_time"]
                ressource_capacity_time = [[int(x)
                                            for x in
                                            self.rcpsp_model.resources_availability[res][:(instance["max_time"]+1)]]
                                           for res in resources_list]
                # print(instance["max_time"])
                # print(len(ressource_capacity_time))
                # print([len(x) for x in ressource_capacity_time])
                instance["ressource_capacity_time"] = ressource_capacity_time
                keys += ["ressource_capacity_time"]
        instance["nb_skill"] = len(self.rcpsp_model.skills_set)
        instance["skillreq"] = skill_required
        instance["nb_units"] = nb_units
        keys += ["nb_skill", "skillreq", "nb_units"]
        ressource_unit_capacity_time = [[1 if x else 0
                                         for x
                                         in self.rcpsp_model.employees[j].calendar_employee[:(instance["max_time"]+1)]]
                                         for j in sorted(self.rcpsp_model.employees)]

        import math
        skillunits = [[int(math.floor(self.rcpsp_model.employees[j].dict_skill[s].skill_value))
                       if s in self.rcpsp_model.employees[j].dict_skill else 0
                       for s in skills_set]
                      for j in sorted(self.rcpsp_model.employees)]
        self.employees_position = sorted(self.rcpsp_model.employees)
        instance["ressource_unit_capacity_time"] = ressource_unit_capacity_time
        instance["skillunits"] = skillunits
        keys += ["skillunits", "ressource_unit_capacity_time"]
        print("Employee position CP ", self.employees_position)

        # print('rreq: ', rreq)
        if not no_ressource:
            instance["rreq"] = rreq
            keys += ["rreq"]

            rcap = [int(max(self.rcpsp_model.resources_availability[x])) for x in resources_list]
            # print('rcap: ', rcap)
            instance["rcap"] = rcap
            keys += ["rcap"]
            # print('non_renewable_resources:', self.rcpsp_model.non_renewable_resources)
            rtype = [2 if res in self.rcpsp_model.non_renewable_resources else 1
                     for res in resources_list]

            # print('rtype: ', rtype)
            instance["rtype"] = rtype
            keys += ["rtype"]

        succ = [set(self.rcpsp_model.successors[task]) for task in sorted_tasks]
        # print('succ: ', succ)

        instance["succ"] = succ
        keys += ["succ"]

        # import pymzn
        # pymzn.dict2dzn({key: instance[key] for key in keys},
        #                fout='ms_rcpsp_example_imopse.dzn')
        self.instance = instance
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)
        if p_s is not None:
            constraint_strings = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    string = "constraint start[" + str(task) + "] == " + str(p_s.start_times[task]) + ";\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(p_s.partial_permutation[:-1], p_s.partial_permutation[1:]):
                    string = "constraint start[" + str(t1) + "] <= start[" + str(t2) + "];\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        string = "constraint start[" + str(t1) + "] <= start[" + str(t2) + "];\n"
                        self.instance.add_string(string)
                        constraint_strings += [string]
            if p_s.task_mode is not None:
                for task in p_s.start_times:
                    indexes = [i for i in self.modeindex_map if self.modeindex_map[i]["task"] == task
                               and self.modeindex_map[i]["original_mode_index"] == p_s.task_mode[task]]
                    if len(indexes) >= 0:
                        string = "constraint mrun["+str(indexes[0])+"] == 1;"
                        self.instance.add_string(string)
                        constraint_strings += [string]
            if p_s.start_together is not None:
                for t1, t2 in p_s.start_together:
                    string = "constraint start[" + str(t1) + "] == start[" + str(t2) + "];\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    string = "constraint start[" + str(t2) + "] >= start[" + str(t1) + "]+"+str(delta)+";\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    string = "constraint start[" + str(t2) + "] >= start[" + str(t1) + "]+adur["+str(t1)+"]+"+str(delta)+";\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.start_at_end is not None:
                for t1, t2 in p_s.start_at_end:
                    string = "constraint start[" + str(t2) + "] == start[" + str(t1) + "]+adur["+str(t1)+"];\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]

    def retrieve_solutions(self, result, parameters_cp: ParametersCP=ParametersCP.default()):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        mruns = []
        units_used = []
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], MS_RCPSPSolCP):
                    starts += [result[i].dict["start"]]
                    mruns += [result[i].dict["mrun"]]
                    units_used += [result[i].dict["unit_used"]]
                else:
                    starts += [result[i, "start"]]
                    mruns += [result[i, "mrun"]]
                    units_used += [result[i, "unit_used"]]
                # array_skill = result[i, "array_skills_required"]
                # print("Objective : ", result[i, "objective"])
        else:
            if isinstance(result, MS_RCPSPSolCP):
                starts += [result.dict["start"]]
                mruns += [result.dict["mrun"]]
                units_used += [result.dict["unit_used"]]
            else:
                starts += [result["start"]]
                mruns += [result["mrun"]]
                units_used += [result["unit_used"]]
            # array_skill = result["array_skills_required"]
        for start_times, mrun, unit_used in zip(starts, mruns, units_used):
            #print("New Solution")
            modes = []
            usage = {}
            for i in range(len(mrun)):
                if mrun[i] and (self.modeindex_map[i + 1]['task'] != 1) and (
                       self.modeindex_map[i + 1]['task'] != self.rcpsp_model.n_jobs_non_dummy + 2):
                    modes.append(self.modeindex_map[i + 1]['original_mode_index'])
                elif (self.modeindex_map[i + 1]['task'] == 1) or (
                        self.modeindex_map[i + 1]['task'] == self.rcpsp_model.n_jobs_non_dummy + 2):
                    modes.append(1)
            for w in range(len(unit_used)):
                for task in range(len(unit_used[w])):
                    if unit_used[w][task] == 1:
                        task_id = task+1
                        #print("Unit used : , ", w, "taskid : ",
                        #      task_id, unit_used[w][task])
                        mode = modes[task_id-1]
                        skills_needed = set([s #, self.rcpsp_model.mode_details[task_id][mode][s])
                                             for s in self.rcpsp_model.skills_set
                                             if s in self.rcpsp_model.mode_details[task_id][mode]
                                             and self.rcpsp_model.mode_details[task_id][mode][s] > 0])

                        skills_worker = set([s
                                             # self.rcpsp_model.employees[self.employees_position[w]].dict_skill[s].skill_value)
                                             for s in self.rcpsp_model.employees[self.employees_position[w]].dict_skill
                                             if self.rcpsp_model.employees[self.employees_position[w]].dict_skill[s].skill_value > 0])
                        intersection = skills_needed.intersection(skills_worker)
                        if len(intersection) > 0:
                            if task_id not in usage:
                                usage[task_id] = {}
                            usage[task_id][self.employees_position[w]] = intersection
            rcpsp_schedule = {}
            for i in range(len(start_times)):
                rcpsp_schedule[i + 1] = {'start_time': start_times[i],
                                         'end_time': start_times[i]
                                         + self.rcpsp_model.mode_details[i + 1][modes[i]]['duration']}
            sol = MS_RCPSPSolution(problem=self.rcpsp_model,
                                   modes={i+1: modes[i] for i in range(len(modes))},
                                   schedule=rcpsp_schedule,
                                   employee_usage=usage)
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            list_solutions_fit += [(sol, objective)]
        result_storage = ResultStorage(list_solution_fits=list_solutions_fit,
                                       best_solution=best_solution,
                                       mode_optim=self.params_objective_function.sense_function,
                                       limit_store=False)
        return result_storage

    def solve(self, parameters_cp: ParametersCP=ParametersCP.default(), **args):
        if self.instance is None:
            self.init_model(**args)
        timeout = parameters_cp.TimeLimit
        intermediate_solutions = parameters_cp.intermediate_solution
        result = self.instance.solve(timeout=timedelta(seconds=timeout),
                                     intermediate_solutions=intermediate_solutions)
        verbose = args.get("verbose", True)
        if verbose:
            print(result.status)
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)
