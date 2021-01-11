from dataclasses import InitVar
from typing import Union, List
from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP, CPSolverName, map_cp_solver_name
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ObjectiveHandling, \
    ParamsObjectiveFunction, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution,\
    SingleModeRCPSPModel, MultiModeRCPSPModel, \
    RCPSPModelCalendar, PartialSolution
from minizinc import Instance, Model, Solver, Status, Result
import json
from datetime import timedelta
import os, random
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {"single": os.path.join(this_path, "../minizinc/rcpsp_single_mode_mzn.mzn"),
             "single-preemptive": os.path.join(this_path, "../minizinc/rcpsp_single_mode_mzn_preemptive.mzn"),
             "multi": os.path.join(this_path, "../minizinc/rcpsp_multi_mode_mzn.mzn"),
             "multi-no-bool": os.path.join(this_path, "../minizinc/rcpsp_multi_mode_mzn_no_bool.mzn"),
             "multi-calendar": os.path.join(this_path, "../minizinc/rcpsp_multi_mode_mzn_calendar.mzn"),
             "multi-calendar-boxes": os.path.join(this_path, "../minizinc/rcpsp_mzn_calendar_boxes.mzn"),
             "modes": os.path.join(this_path, "../minizinc/mrcpsp_mode_satisfy.mzn")}


class RCPSPSolCP:
    objective: int
    __output_item: InitVar[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        print("One solution ", self.objective)

    def check(self) -> bool:
        return True


class CP_RCPSP_MZN(CPSolver):
    def __init__(self, rcpsp_model: RCPSPModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.instance: Instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["s"]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        model_type = args.get("model_type", "single")
        if model_type == "single-preemptive":
            nb_preemptive = args.get("nb_preemptive", 2)
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type",  False)
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)
        if model_type == "single-preemptive":
            instance["nb_preemptive"] = nb_preemptive
            # TODO : make this as options.
            instance["possibly_preemptive"] = [True for task in self.rcpsp_model.mode_details]
            instance["max_preempted"] = 3
        n_res = len(list(self.rcpsp_model.resources.keys()))
        # print('n_res: ', n_res)
        instance["n_res"] = n_res
        sorted_resources = sorted(self.rcpsp_model.resources_list)
        self.resources_index = sorted_resources
        rc = [int(self.rcpsp_model.resources[r])
              for r in sorted_resources]
        # print('rc: ', rc)
        instance["rc"] = rc
        n_tasks = self.rcpsp_model.n_jobs + 2
        # print('n_tasks: ', n_tasks)
        instance["n_tasks"] = n_tasks
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        d = [int(self.rcpsp_model.mode_details[key][1]['duration']) for key in sorted_tasks]
        # print('d: ', d)
        instance["d"] = d
        rr = []
        index = 0
        for res in sorted_resources:
            rr.append([])
            for task in sorted_tasks:
                rr[index].append(int(self.rcpsp_model.mode_details[task][1][res]))
            index += 1
        instance["rr"] = rr
        suc = [set(self.rcpsp_model.successors[task]) for task in sorted_tasks]
        instance["suc"] = suc
        self.instance = instance
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)
        if p_s is not None:
            constraint_strings = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    string = "constraint s[" + str(task) + "] == " + str(p_s.start_times[task]) + ";\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(p_s.partial_permutation[:-1], p_s.partial_permutation[1:]):
                    string = "constraint s[" + str(t1) + "] <= s[" + str(t2) + "];\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        string = "constraint s[" + str(t1) + "] <= s[" + str(t2) + "];\n"
                        self.instance.add_string(string)
                        constraint_strings += [string]
            if p_s.start_together is not None:
                for t1, t2 in p_s.start_together:
                    string = "constraint s[" + str(t1) + "] == s[" + str(t2) + "];\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.start_after_nunit is not None:
                for t1, t2, delta in p_s.start_after_nunit:
                    string = "constraint s[" + str(t2) + "] >= s[" + str(t1) + "]+"+str(delta)+";\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.start_at_end_plus_offset is not None:
                for t1, t2, delta in p_s.start_at_end_plus_offset:
                    string = "constraint s[" + str(t2) + "] >= s[" + str(t1) + "]+d["+str(t1)+"]+"+str(delta)+";\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.start_at_end is not None:
                for t1, t2 in p_s.start_at_end:
                    string = "constraint s[" + str(t2) + "] == s[" + str(t1) + "]+d["+str(t1)+"];\n"
                    self.instance.add_string(string)
                    constraint_strings += [string]

    def retrieve_solutions(self, result, parameters_cp: ParametersCP=ParametersCP.default())->ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts += [result[i].dict["s"]]
                else:
                    starts += [result[i, "s"]]
        else:
            if isinstance(result, RCPSPSolCP):
                starts += [result.dict["s"]]
            else:
                starts = [result["s"]]

        for start_times in starts:
            rcpsp_schedule = {}
            for k in range(len(start_times)):
                rcpsp_schedule[k + 1] = {'start_time': start_times[k],
                                         'end_time': start_times[k]
                                                     + self.rcpsp_model.mode_details[k + 1][1]['duration']}
            sol = RCPSPSolution(problem=self.rcpsp_model,
                                rcpsp_schedule=rcpsp_schedule, rcpsp_schedule_feasible=True)
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

    def solve(self, parameters_cp: ParametersCP=ParametersCP.default(), **args):  # partial_solution: PartialSolution=None, **args):
        if self.instance is None:
            self.init_model(**args)
        timeout = parameters_cp.TimeLimit
        intermediate_solutions = parameters_cp.intermediate_solution
        try:
            result = self.instance.solve(timeout=timedelta(seconds=timeout),
                                         intermediate_solutions=intermediate_solutions)
        except Exception as e:
            print(e)
            return None
        verbose = args.get("verbose", False)
        if verbose:
            print(result.status)
            print(result.statistics["solveTime"])
        return self.retrieve_solutions(result, parameters_cp=parameters_cp)


class CP_MRCPSP_MZN(CPSolver):
    def __init__(self,
                 rcpsp_model: RCPSPModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["start", "mrun"]  # For now, I've put the var names of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)
        self.calendar = False
        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            self.calendar = True

    def init_model(self, **args):
        model_type = args.get("model_type", None)
        if model_type is None:
            model_type = "multi" if not self.calendar else "multi-calendar"
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type", False)
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        resources_list = list(self.rcpsp_model.resources.keys())
        self.resources_index = resources_list
        instance = Instance(solver, model)
        n_res = len(resources_list)
        # print('n_res: ', n_res)
        keys = []

        instance["n_res"] = n_res
        keys += ["n_res"]

        # rc = [val for val in self.rcpsp_model.resources.values()]
        # # print('rc: ', rc)
        # instance["rc"] = rc

        n_tasks = self.rcpsp_model.n_jobs + 2
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
        self.modeindex_map = {}
        general_counter = 1
        for act in sorted_tasks:
            tmp = sorted(self.rcpsp_model.mode_details[act].keys())
            # tmp = [counter + x for x in tmp]
            set_mode_task = set()
            for i in range(len(tmp)):
                original_mode_index = tmp[i]
                set_mode_task.add(general_counter)
                self.modeindex_map[general_counter] = {'task': act, 'original_mode_index': original_mode_index}
                general_counter += 1
            modes.append(set_mode_task)
            dur = dur + [self.rcpsp_model.mode_details[act][key]['duration']
                         for key in tmp]
        # print('modes: ', modes)
        instance['modes'] = modes
        keys += ["modes"]

        # print('dur: ', dur)
        instance['dur'] = dur
        keys += ["dur"]


        rreq = []
        index = 0
        for res in resources_list:
            rreq.append([])
            for task in sorted_tasks:
                for mod in sorted(self.rcpsp_model.mode_details[task].keys()):
                    rreq[index].append(int(self.rcpsp_model.mode_details[task][mod][res]))
            index += 1

        # print('rreq: ', rreq)
        instance["rreq"] = rreq
        keys += ["rreq"]

        if not self.calendar:
            rcap = [int(self.rcpsp_model.resources[x]) for x in resources_list]
        else:
            rcap = [int(max(self.rcpsp_model.resources[x])) for x in resources_list]
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

        if self.calendar:
            one_ressource = list(self.rcpsp_model.resources.keys())[0]
            instance["max_time"] = len(self.rcpsp_model.resources[one_ressource])
            print(instance["max_time"])
            keys += ["max_time"]
            ressource_capacity_time = [[int(x) for x in self.rcpsp_model.resources[res]]
                                       for res in resources_list]
            # print(instance["max_time"])
            # print(len(ressource_capacity_time))
            # print([len(x) for x in ressource_capacity_time])
            instance["ressource_capacity_time"] = ressource_capacity_time
            keys += ["ressource_capacity_time"]

        # import pymzn
        # pymzn.dict2dzn({key: instance[key] for key in keys},
        #                 fout='rcpsp_.dzn')
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
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts += [result[i].dict["start"]]
                    mruns += [result[i].dict["mrun"]]
                else:
                    starts += [result[i, "start"]]
                    mruns += [result[i, "mrun"]]

        else:
            if isinstance(result, RCPSPSolCP):
                starts += [result.dict["start"]]
                mruns += [result.dict["mrun"]]
            else:
                starts = [result["start"]]
                mruns = [result["mrun"]]
        for start_times, mrun in zip(starts, mruns):
            modes = []
            for i in range(len(mrun)):
                if mrun[i] and (self.modeindex_map[i + 1]['task'] != 1) and (
                        self.modeindex_map[i + 1]['task'] != self.rcpsp_model.n_jobs + 2):
                    modes.append(self.modeindex_map[i + 1]['original_mode_index'])
                elif (self.modeindex_map[i + 1]['task'] == 1) or (
                        self.modeindex_map[i + 1]['task'] == self.rcpsp_model.n_jobs + 2):
                    modes.append(1)
            rcpsp_schedule = {}
            for i in range(len(start_times)):
                rcpsp_schedule[i + 1] = {'start_time': start_times[i],
                                         'end_time': start_times[i]
                                         + self.rcpsp_model.mode_details[i + 1][modes[i]]['duration']}
            sol = RCPSPSolution(problem=self.rcpsp_model,
                                rcpsp_schedule=rcpsp_schedule,
                                rcpsp_modes=modes[1:-1],
                                rcpsp_schedule_feasible=True)
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


class MRCPSP_Result:
    objective: int
    __output_item: InitVar[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        self.mode_chosen = json.loads(_output_item)

    def check(self) -> bool:
        return True


class CP_MRCPSP_MZN_NOBOOL(CPSolver):
    def __init__(self,
                 rcpsp_model: RCPSPModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["start", "mrun"]  # For now, I've put the var names of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)
        self.calendar = False
        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            self.calendar = True

    def init_model(self, **args):
        model = Model(files_mzn["multi-no-bool"])

        model.output_type = MRCPSP_Result
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        resources_list = list(self.rcpsp_model.resources.keys())
        instance = Instance(solver, model)
        n_res = len(resources_list)
        # print('n_res: ', n_res)
        keys = []

        instance["n_res"] = n_res
        keys += ["n_res"]

        # rc = [val for val in self.rcpsp_model.resources.values()]
        # # print('rc: ', rc)
        # instance["rc"] = rc

        n_tasks = self.rcpsp_model.n_jobs + 2
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
        general_counter = 1
        for act in sorted_tasks:
            tmp = sorted(self.rcpsp_model.mode_details[act].keys())
            # tmp = [counter + x for x in tmp]
            set_mode_task = set()
            for i in range(len(tmp)):
                original_mode_index = tmp[i]
                set_mode_task.add(general_counter)
                self.modeindex_map[general_counter] = {'task': act, 'original_mode_index': original_mode_index}
                general_counter += 1
            modes.append(set_mode_task)
            dur = dur + [self.rcpsp_model.mode_details[act][key]['duration']
                         for key in tmp]

        # print('modes: ', modes)
        instance['modes'] = modes
        keys += ["modes"]


        # print('dur: ', dur)
        instance['dur'] = dur
        keys += ["dur"]


        rreq = []
        index = 0
        for res in resources_list:
            rreq.append([])
            for task in sorted_tasks:
                for mod in sorted(self.rcpsp_model.mode_details[task].keys()):
                    rreq[index].append(int(self.rcpsp_model.mode_details[task][mod][res]))
            index += 1

        # print('rreq: ', rreq)
        instance["rreq"] = rreq
        keys += ["rreq"]

        if not self.calendar:
            rcap = [self.rcpsp_model.resources[x] for x in resources_list]
        else:
            rcap = [int(max(self.rcpsp_model.resources[x])) for x in resources_list]
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

        if self.calendar:
            one_ressource = list(self.rcpsp_model.resources.keys())[0]
            instance["max_time"] = len(self.rcpsp_model.resources[one_ressource])
            print(instance["max_time"])
            keys += ["max_time"]
            ressource_capacity_time = [[int(x) for x in self.rcpsp_model.resources[res]]
                                       for res in resources_list]
            # print(instance["max_time"])
            # print(len(ressource_capacity_time))
            # print([len(x) for x in ressource_capacity_time])
            instance["ressource_capacity_time"] = ressource_capacity_time
            keys += ["ressource_capacity_time"]

        # import pymzn
        # pymzn.dict2dzn({key: instance[key] for key in keys},
        #                fout='rcpsp_.dzn')
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

    def retrieve_solutions(self, result, parameters_cp: ParametersCP=ParametersCP.default()):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        mruns = []
        object_result: List[MRCPSP_Result] = []
        if intermediate_solutions:
            for i in range(len(result)):
                object_result += [result[i]]
                # print("Objective : ", result[i, "objective"])
        else:
            object_result += [result]
        for res in object_result:
            modes = []
            for j in range(len(res.mode_chosen)):
                if (self.modeindex_map[j + 1]['task'] != 1) and (self.modeindex_map[j + 1]['task'] != self.rcpsp_model.n_jobs + 2):
                    modes.append(self.modeindex_map[res.mode_chosen[j]]['original_mode_index'])
                elif (self.modeindex_map[j + 1]['task'] == 1) or (
                        self.modeindex_map[j + 1]['task'] == self.rcpsp_model.n_jobs + 2):
                    modes.append(1)
            rcpsp_schedule = {}
            start_times = res.dict["start"]
            for i in range(len(start_times)):
                rcpsp_schedule[i + 1] = {'start_time': start_times[i],
                                         'end_time': start_times[i]
                                         + self.rcpsp_model.mode_details[i + 1][modes[i]]['duration']}
            sol = RCPSPSolution(problem=self.rcpsp_model,
                                rcpsp_schedule=rcpsp_schedule,
                                rcpsp_modes=modes[1:-1],
                                rcpsp_schedule_feasible=True)
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


class CP_MRCPSP_MZN_MODES:
    def __init__(self, rcpsp_model: RCPSPModel,
                 cp_solver_name: CPSolverName=CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.rcpsp_model = rcpsp_model
        self.instance: Instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["start", "mrun"]  # For now, I've put the var names of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        model = Model(files_mzn["modes"])
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)

        n_res = len(list(self.rcpsp_model.resources.keys()))
        # print('n_res: ', n_res)
        instance["n_res"] = n_res

        # rc = [val for val in self.rcpsp_model.resources.values()]
        # # print('rc: ', rc)
        # instance["rc"] = rc

        n_tasks = self.rcpsp_model.n_jobs + 2
        # print('n_tasks: ', n_tasks)
        instance["n_tasks"] = n_tasks
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        # print('mode_details: ', self.rcpsp_model.mode_details)
        n_opt = sum([len(list(self.rcpsp_model.mode_details[key].keys())) for key in sorted_tasks])
        # print('n_opt: ', n_opt)
        instance["n_opt"] = n_opt

        modes = []
        counter = 0
        self.modeindex_map = {}

        for act in sorted_tasks:
            tmp = list(self.rcpsp_model.mode_details[act].keys())
            # tmp = [counter + x for x in tmp]
            for i in range(len(tmp)):
                original_mode_index = tmp[i]
                mod_index = counter+tmp[i]
                tmp[i] = mod_index
                self.modeindex_map[mod_index] = {'task': act, 'original_mode_index': original_mode_index}
            modes.append(set(tmp))
            counter = tmp[-1]

        # print('modes: ', modes)
        instance['modes'] = modes


        rreq = []
        index = 0
        for res in self.rcpsp_model.resources.keys():
            rreq.append([])
            for task in sorted_tasks:
                for mod in self.rcpsp_model.mode_details[task].keys():
                    rreq[index].append(int(self.rcpsp_model.mode_details[task][mod][res]))
            index += 1

        # print('rreq: ', rreq)
        instance["rreq"] = rreq

        rcap = [val for val in self.rcpsp_model.resources.values()]
        # print('rcap: ', rcap)
        if isinstance(rcap[0], list):
            rcap = [int(max(r)) for r in rcap]
        instance["rcap"] = rcap

        # print('non_renewable_resources:', self.rcpsp_model.non_renewable_resources)
        rtype = [2 if res in self.rcpsp_model.non_renewable_resources else 1 for res in self.rcpsp_model.resources.keys()]

        # print('rtype: ', rtype)
        instance["rtype"] = rtype
        self.instance: Instance = instance
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)
        if p_s is not None:
            constraint_strings = []
            if p_s.task_mode is not None:
                for task in p_s.start_times:
                    indexes = [i for i in self.modeindex_map if self.modeindex_map[i]["task"] == task
                               and self.modeindex_map[i]["original_mode_index"] == p_s.task_mode[task]]
                    if len(indexes) >= 0:
                        print("Index found : ", len(indexes))
                        string = "constraint mrun[" + str(indexes[0]) + "] == 1;"
                        self.instance.add_string(string)
                        constraint_strings += [string]


    def retrieve_solutions(self, result, parameters_cp: ParametersCP=ParametersCP.default()):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        mruns = []
        if intermediate_solutions:
            for i in range(len(result)):
                mruns += [result[i, "mrun"]]
        else:
            mruns += [result["mrun"]]
        all_modes = []
        for mrun in mruns:
            modes = [1]*(self.rcpsp_model.n_jobs+2)
            for i in range(len(mrun)):
                if mrun[i] == 1 and (self.modeindex_map[i + 1]['task'] != 1) and (
                                     self.modeindex_map[i + 1]['task'] != self.rcpsp_model.n_jobs + 2):
                    modes[self.modeindex_map[i+1]['task']-1] = self.modeindex_map[i + 1]['original_mode_index']
            all_modes += [modes]
        return all_modes

    def solve(self, parameters_cp: ParametersCP=ParametersCP.default(), **args):
        if self.instance is None:
            self.init_model(**args)
        timeout = parameters_cp.TimeLimit
        intermediate_solutions = parameters_cp.intermediate_solution
        result = self.instance.solve(timeout=timedelta(seconds=timeout),
                                     nr_solutions=parameters_cp.nr_solutions
                                     if not parameters_cp.all_solutions else None,
                                     all_solutions=parameters_cp.all_solutions,
                                     intermediate_solutions=intermediate_solutions)
        verbose = args.get("verbose", False)
        if verbose:
            print(result.status)
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)
