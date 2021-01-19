import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, EncodingRegister, TypeAttribute, \
    ObjectiveRegister, TypeObjective, ObjectiveHandling, TupleFitness, RobustProblem, MethodAggregating, ModeOptim
from skdecide.builders.discrete_optimization.generic_tools.graph_api import Graph
from typing import List, Union, NamedTuple, Tuple, Dict
from abc import abstractmethod
import numpy as np
from enum import Enum
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import poisson, rv_discrete, randint
from collections import defaultdict


def tree():
    return defaultdict(tree)


class ScheduleGenerationScheme(Enum):
    SERIAL_SGS = 0
    PARALLEL_SGS = 1


class RCPSPSolution(Solution):
    rcpsp_permutation: Union[List[int], np.array]
    rcpsp_schedule: Dict[int, Dict]  # {task_id: {'start': start_time, 'end': end_time, 'resources': list_of_resource_ids}}
    rcpsp_modes: List[int]  # {task_id: mode_id}
    standardised_permutation: Union[List[int], np.array]

    def __init__(self, problem,
                 rcpsp_permutation=None,
                 rcpsp_schedule=None,
                 rcpsp_modes=None,
                 rcpsp_schedule_feasible=None,
                 standardised_permutation=None):
        self.problem = problem
        self.rcpsp_permutation = rcpsp_permutation
        self.rcpsp_schedule = rcpsp_schedule
        self._schedule_to_recompute = rcpsp_schedule is None
        self.rcpsp_modes = rcpsp_modes
        self.rcpsp_schedule_feasible = rcpsp_schedule_feasible
        self.standardised_permutation = standardised_permutation

        if self.rcpsp_modes is None:
            if not self.problem.is_rcpsp_multimode():
            # if isinstance(self.problem, SingleModeRCPSPModel):
                self.rcpsp_modes = [1 for i in range(self.problem.n_jobs)]
            else:
            # elif isinstance(self.problem, MultiModeRCPSPModel):
                self.rcpsp_modes = self.problem.fixed_modes

        if self.rcpsp_permutation is None:
            if isinstance(self.problem, MultiModeRCPSPModel):
                self.rcpsp_permutation = self.problem.fixed_permutation
            if self.rcpsp_schedule is not None:
                self.standardised_permutation = self.generate_permutation_from_schedule()
                self.rcpsp_permutation = deepcopy(self.standardised_permutation)
                self._schedule_to_recompute = False

        if rcpsp_schedule is None:
            if not isinstance(problem, Aggreg_RCPSPModel):
                self.generate_schedule_from_permutation_serial_sgs()
            if isinstance(problem, RCPSP_H_Model):
                self.rcpsp_schedule = problem.rcpsp_pre_helper_correction(self)

        if self.standardised_permutation is None:
            if not isinstance(problem, Aggreg_RCPSPModel):
                self.standardised_permutation = self.generate_permutation_from_schedule()

    def change_problem(self, new_problem: Problem):
        self.__init__(problem=new_problem,
                      rcpsp_permutation=self.rcpsp_permutation,
                      rcpsp_modes=self.rcpsp_modes)

    def __setattr__(self, key, value):
        super.__setattr__(self, key, value)
        if key == "rcpsp_permutation":
            self._schedule_to_recompute = True

    def copy(self):
        return RCPSPSolution(problem=self.problem,
                             rcpsp_permutation=deepcopy(self.rcpsp_permutation),
                             rcpsp_modes=deepcopy(self.rcpsp_modes),
                             rcpsp_schedule=self.rcpsp_schedule,
                             rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
                             standardised_permutation=self.standardised_permutation)

    def lazy_copy(self):
        return RCPSPSolution(problem=self.problem,
                             rcpsp_permutation=self.rcpsp_permutation,
                             rcpsp_modes=self.rcpsp_modes,
                             rcpsp_schedule=self.rcpsp_schedule,
                             rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
                             standardised_permutation=self.standardised_permutation)

    def __str__(self):
        if self.rcpsp_schedule is None:
            sched_str = 'None'
        else:
            sched_str = str(self.rcpsp_schedule)
        val = "RCPSP solution (rcpsp_schedule): " + sched_str
        print('type: ', type(val))
        return val

    def generate_permutation_from_schedule(self):
        sorted_task = [i - 2 for i in sorted(self.rcpsp_schedule,
                                             key=lambda x: self.rcpsp_schedule[x]["start_time"])]
        sorted_task.remove(-1)
        sorted_task.remove(max(sorted_task))
        return sorted_task

    def compute_mean_resource_reserve(self):
        if not self.rcpsp_schedule_feasible:
            return 0.
        last_activity = max(list(self.rcpsp_schedule.keys()))
        makespan = self.rcpsp_schedule[last_activity]['end_time']
        resource_avail_in_time = {}
        for res in list(self.problem.resources.keys()):
            if isinstance(self.problem, RCPSPModelCalendar):
                resource_avail_in_time[res] = self.problem.resources[res][:makespan+1]
            else:
                resource_avail_in_time[res] = np.full(makespan, self.problem.resources[res], dtype=int).tolist()
        for act_id in self.problem.mode_details.keys():
            # if act_id not in self.rcpsp_schedule:
            #     continue
            if (act_id != 1) and (act_id != self.problem.n_jobs+2):
                start_time = self.rcpsp_schedule[act_id]['start_time']
                end_time = self.rcpsp_schedule[act_id]['end_time']
                mode = self.rcpsp_modes[act_id-2]
                # print('act_id: ', act_id)
                # print('start_time: ', start_time)
                # print('end_time: ', end_time)
                # print('mode: ', mode)
                for t in range(start_time, end_time):
                    for res in resource_avail_in_time.keys():
                        resource_avail_in_time[res][t] -= self.problem.mode_details[act_id][mode][res]  # 17
                        if res in self.problem.non_renewable_resources and t == end_time:
                            for tt in range(end_time + 1, makespan):
                                resource_avail_in_time[res][tt] -= \
                                self.problem.mode_details[act_id][mode][res]
        # print('resource_avail_in_time: ', resource_avail_in_time)
        mean_avail = {}
        for res in list(self.problem.resources.keys()):
            mean_avail[res] = np.mean(resource_avail_in_time[res])

        mean_resource_reserve = np.mean([mean_avail[res] / self.problem.resources[res] for res in list(self.problem.resources.keys())])
        # print('mean_avail: ', mean_avail)
        # print('original_avail: ', self.problem.resources)
        # print('val: ', mean_resource_reserve)
        return mean_resource_reserve

    def generate_schedule_from_permutation_serial_sgs(self):
        perm = deepcopy(self.rcpsp_permutation)
        activity_end_times = {}

        unfeasible_non_renewable_resources = False
        new_horizon = self.problem.horizon * self.problem.horizon_multiplier

        # 1, 2
        resource_avail_in_time = {}
        for res in list(self.problem.resources.keys()):
            if  isinstance(self.problem, RCPSPModelCalendar):
                resource_avail_in_time[res] = self.problem.resources[res][:new_horizon+1]
            else:
                resource_avail_in_time[res] = np.full(new_horizon, self.problem.resources[res], dtype=int).tolist()
        # 3
        minimum_starting_time = {}
        for act in list(self.problem.successors.keys()):
            minimum_starting_time[act] = 0

        perm_extended = [x+2 for x in perm]
        perm_extended.insert(0, 1)
        perm_extended.append(self.problem.n_jobs + 2)
        # print('perm_extended: ', perm_extended)
        modes_extended = deepcopy(self.rcpsp_modes)
        modes_extended.insert(0, 1)
        modes_extended.append(1)
        # print('pre-modes_extended: ', modes_extended)

        # fix modes in case specified mode not in mode details for the activites
        for i in range(len(modes_extended)):
            # print(list(self.problem.mode_details[i + 1].keys()))

            if modes_extended[i] not in list(self.problem.mode_details[i+1].keys()):
                modes_extended[i] = 1
                if i != 0 and i != len(modes_extended)-1:
                    self.rcpsp_modes[i-1] = modes_extended[i]

        # print('modes_extended: ', modes_extended)
        # print('start SGS')
        while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
            # print('perm_extended: ', perm_extended)
            # get first activity in perm with precedences respected
            for id in perm_extended:
                respected = True
                for pred in self.problem.successors.keys():
                    if id in self.problem.successors[pred] and pred in perm_extended:
                        respected = False
                        break
                if respected:
                    act_id = id
                    break
            # print('next act_id respecting precedences :', act_id)
            # for act_id in perm_extended:  # 4
            current_min_time = minimum_starting_time[act_id]  # 5
            # print('current_min_time_0: ', current_min_time)
            valid = False  # 6
            while not valid:  # 7
                # print('current_min_time: ', current_min_time)
                valid = True  # 8
                for t in range(current_min_time,
                               current_min_time
                               +self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']):  # 9
                    # print(act_id, t)
                    for res in resource_avail_in_time.keys():  # 10
                        # if t == 4:
                        #     print('res--', res)
                            # print('t--', t)
                            # print('horizon', new_horizon)
                            # print('resource_avail_in_time[res][t]: ', resource_avail_in_time[res][t])
                            # print('self.problem.mode_details[act_id][modes_extended[act_id-1]][res]: ', self.problem.mode_details[act_id][modes_extended[act_id-1]][res])
                        if t < new_horizon:
                            if resource_avail_in_time[res][t] < self.problem.mode_details[act_id][modes_extended[act_id-1]][res]:  # 11
                                valid = False  # 12
                        else:
                            unfeasible_non_renewable_resources = True
                if not valid:  # 13
                    current_min_time += 1  # 14
            if not unfeasible_non_renewable_resources:
                # print('current_min_time: ', current_min_time)
                # print('in mode: ', modes_extended[act_id-1])
                # print('(mode details - duration: ', self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration'])
                end_t = current_min_time + self.problem.mode_details[act_id][modes_extended[act_id - 1]]['duration']-1
                # print('end_t: ', end_t)
                for t in range(current_min_time, current_min_time + self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']):  # 15
                    for res in resource_avail_in_time.keys():  # 16
                        resource_avail_in_time[res][t] -= self.problem.mode_details[act_id][modes_extended[act_id-1]][res]  # 17
                        if res in self.problem.non_renewable_resources and t == end_t:
                            for tt in range(end_t+1, new_horizon):
                                resource_avail_in_time[res][tt] -= self.problem.mode_details[act_id][modes_extended[act_id - 1]][res]
                                if resource_avail_in_time[res][tt] < 0:
                                    unfeasible_non_renewable_resources = True
                                    # print('resource', res, 'exhausted', resource_avail_in_time[res][tt], 'initial avail: ', self.problem.resources[res])
                        # print(res, ' resource_avail_in_time[res]: ', resource_avail_in_time[res])

                activity_end_times[act_id] = current_min_time + self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']  # 18
                perm_extended.remove(act_id)
                # print('scheduled to complete at: ', activity_end_times[act_id])
                for s in self.problem.successors[act_id]:  # 19
                    minimum_starting_time[s] = max(minimum_starting_time[s], activity_end_times[act_id])  # 20

        # print('activity_end_times: ', activity_end_times)
        self.rcpsp_schedule = {}
        for act_id in activity_end_times:
            self.rcpsp_schedule[act_id] = {}
            self.rcpsp_schedule[act_id]['start_time'] = activity_end_times[act_id] - self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']
            self.rcpsp_schedule[act_id]['end_time'] = activity_end_times[act_id]
        if unfeasible_non_renewable_resources:
            self.rcpsp_schedule_feasible = False
            last_act_id = list(self.problem.successors.keys())[-1]
            if last_act_id not in self.rcpsp_schedule.keys():
                self.rcpsp_schedule[last_act_id] = {}
                self.rcpsp_schedule[last_act_id]['start_time'] = 99999999
                self.rcpsp_schedule[last_act_id]['end_time'] = 9999999
        else:
            self.rcpsp_schedule_feasible = True
        self._schedule_to_recompute = False

    # TODO: Check that this new SGS can be used instead of the previous function, then remove first version
    def generate_schedule_from_permutation_serial_sgs_2(self, current_t, completed_tasks, scheduled_tasks_start_times):
        perm = deepcopy(self.rcpsp_permutation)
        activity_end_times = {}

        unfeasible_non_renewable_resources = False
        new_horizon = self.problem.horizon * self.problem.horizon_multiplier

        # 1, 2
        resource_avail_in_time = {}
        for res in list(self.problem.resources.keys()):
            if isinstance(self.problem, RCPSPModelCalendar):
                resource_avail_in_time[res] = self.problem.resources[res][:new_horizon+1]
            else:
                resource_avail_in_time[res] = np.full(new_horizon, self.problem.resources[res], dtype=int).tolist()
        # 3
        minimum_starting_time = {}
        for act in list(self.problem.successors.keys()):
            if act in list(scheduled_tasks_start_times.keys()):
                minimum_starting_time[act] = scheduled_tasks_start_times[act]
            else:
                minimum_starting_time[act] = current_t

        perm_extended = [x+2 for x in perm]
        perm_extended.insert(0, 1)
        perm_extended.append(self.problem.n_jobs + 2)
        # print('perm_extended: ', perm_extended)
        modes_extended = deepcopy(self.rcpsp_modes)
        modes_extended.insert(0, 1)
        modes_extended.append(1)
        # print('pre-modes_extended: ', modes_extended)

        all_activities = [x for x in list(self.problem.resources.keys()) if x not in list(completed_tasks)]
        # print('all_activities: ', all_activities)
        perm_extended = [x for x in perm_extended if x not in list(completed_tasks)]
        # print('perm_extended - removed completed_tasks: ', perm_extended)

        # fix modes in case specified mode not in mode details for the activites
        for i in range(len(modes_extended)):
            # print(list(self.problem.mode_details[i + 1].keys()))

            if modes_extended[i] not in list(self.problem.mode_details[i+1].keys()):
                modes_extended[i] = 1
                if i != 0 and i != len(modes_extended)-1:
                    self.rcpsp_modes[i-1] = modes_extended[i]

        # print('modes_extended: ', modes_extended)
        # print('start SGS')
        while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
            # print('perm_extended: ', perm_extended)
            # get first activity in perm with precedences respected
            for id in perm_extended:
                respected = True
                for pred in self.problem.successors.keys():
                    if id in self.problem.successors[pred] and pred in perm_extended:
                        respected = False
                        break
                if respected:
                    act_id = id
                    break
            # print('next act_id respecting precedences :', act_id)
            # for act_id in perm_extended:  # 4
            current_min_time = minimum_starting_time[act_id]  # 5
            # print('current_min_time_0: ', current_min_time)
            valid = False  # 6
            while not valid:  # 7
                # print('current_min_time: ', current_min_time)
                valid = True  # 8
                for t in range(current_min_time,
                               current_min_time
                               +self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']):  # 9
                    # print(act_id, t)
                    for res in resource_avail_in_time.keys():  # 10
                        # if t == 4:
                        #     print('res--', res)
                            # print('t--', t)
                            # print('horizon', new_horizon)
                            # print('resource_avail_in_time[res][t]: ', resource_avail_in_time[res][t])
                            # print('self.problem.mode_details[act_id][modes_extended[act_id-1]][res]: ', self.problem.mode_details[act_id][modes_extended[act_id-1]][res])
                        if t < new_horizon:
                            if resource_avail_in_time[res][t] < self.problem.mode_details[act_id][modes_extended[act_id-1]][res]:  # 11
                                valid = False  # 12
                        else:
                            unfeasible_non_renewable_resources = True
                if not valid:  # 13
                    current_min_time += 1  # 14
            if not unfeasible_non_renewable_resources:
                # print('current_min_time: ', current_min_time)
                # print('in mode: ', modes_extended[act_id-1])
                # print('(mode details - duration: ', self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration'])
                end_t = current_min_time + self.problem.mode_details[act_id][modes_extended[act_id - 1]]['duration']-1
                # print('end_t: ', end_t)
                for t in range(current_min_time, current_min_time + self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']):  # 15
                    for res in resource_avail_in_time.keys():  # 16
                        resource_avail_in_time[res][t] -= self.problem.mode_details[act_id][modes_extended[act_id-1]][res]  # 17
                        if res in self.problem.non_renewable_resources and t == end_t:
                            for tt in range(end_t+1, new_horizon):
                                resource_avail_in_time[res][tt] -= self.problem.mode_details[act_id][modes_extended[act_id - 1]][res]
                                if resource_avail_in_time[res][tt] < 0:
                                    unfeasible_non_renewable_resources = True
                                    # print('resource', res, 'exhausted', resource_avail_in_time[res][tt], 'initial avail: ', self.problem.resources[res])
                        # print(res, ' resource_avail_in_time[res]: ', resource_avail_in_time[res])

                activity_end_times[act_id] = current_min_time + self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']  # 18
                perm_extended.remove(act_id)
                # print('scheduled to complete at: ', activity_end_times[act_id])
                for s in self.problem.successors[act_id]:  # 19
                    minimum_starting_time[s] = max(minimum_starting_time[s], activity_end_times[act_id])  # 20

        # print('activity_end_times: ', activity_end_times)
        self.rcpsp_schedule = {}
        for act_id in activity_end_times:
            self.rcpsp_schedule[act_id] = {}
            self.rcpsp_schedule[act_id]['start_time'] = activity_end_times[act_id] - self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']
            self.rcpsp_schedule[act_id]['end_time'] = activity_end_times[act_id]
        for act_id in completed_tasks:
            self.rcpsp_schedule[act_id] = {}
            self.rcpsp_schedule[act_id]['start_time'] = completed_tasks[act_id].start
            self.rcpsp_schedule[act_id]['end_time'] = completed_tasks[act_id].end
        if unfeasible_non_renewable_resources:
            self.rcpsp_schedule_feasible = False
            last_act_id = list(self.problem.successors.keys())[-1]
            if last_act_id not in self.rcpsp_schedule.keys():
                self.rcpsp_schedule[last_act_id] = {}
                self.rcpsp_schedule[last_act_id]['start_time'] = 99999999
                self.rcpsp_schedule[last_act_id]['end_time'] = 9999999
        else:
            self.rcpsp_schedule_feasible = True
        self._schedule_to_recompute = False

    def __hash__(self):
        return hash((tuple(self.rcpsp_permutation), tuple(self.rcpsp_modes)))

    def __eq__(self, other):
        return self.rcpsp_permutation == other.rcpsp_permutation and self.rcpsp_modes == other.rcpsp_modes
    # def generate_schedule_from_permutation_serial_sgs(self):
    #     perm = self.rcpsp_permutation
    #     activity_end_times = {}
    #
    #     # 1, 2
    #     resource_avail_in_time = {}
    #     for res in list(self.problem.resources.keys()):
    #         resource_avail_in_time[res] = np.full(self.problem.horizon, self.problem.resources[res], dtype=int).tolist()
    #
    #     # 3
    #     minimum_starting_time = {}
    #     for act in list(self.problem.successors.keys()):
    #         minimum_starting_time[act-1] = 0
    #
    #     perm_extended = [x+2 for x in perm]
    #     perm_extended.insert(0,1)
    #     perm_extended.append(self.problem.n_jobs + 2)
    #     # print('perm_extended: ', perm_extended)
    #
    #     modes_extended = self.rcpsp_modes
    #     modes_extended.insert(0,1)
    #     modes_extended.append(1)
    #     # print('modes_extended: ', modes_extended)
    #
    #     for act_id in perm_extended:  # 4
    #         current_min_time = minimum_starting_time[act_id]  # 5
    #         valid = False  # 6
    #         while not valid:  # 7
    #             valid = True  # 8
    #             for t in range(current_min_time, current_min_time+self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']):  # 9
    #                 # print(act_id, t)
    #                 for res in resource_avail_in_time.keys():  # 10
    #                     if resource_avail_in_time[res][t] < self.problem.mode_details[act_id][modes_extended[act_id-1]][res]:  # 11
    #                         valid = False  # 12
    #                 if not valid:  # 13
    #                     current_min_time += 1  # 14
    #         for t in range(current_min_time, current_min_time + self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']):  # 15
    #             for res in resource_avail_in_time.keys():  # 16
    #                 resource_avail_in_time[res][t] -= self.problem.mode_details[act_id][modes_extended[act_id-1]][res]  # 17
    #         activity_end_times[act_id] = current_min_time + self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']  # 18
    #         for s in self.problem.successors[act_id]:  # 19
    #             minimum_starting_time[s] = activity_end_times[act_id]  # 20
    #
    #     # print('activity_end_times: ', activity_end_times)
    #     self.rcpsp_schedule = {}
    #     for act_id in activity_end_times:
    #         self.rcpsp_schedule[act_id] = {}
    #         self.rcpsp_schedule[act_id]['start_time'] = activity_end_times[act_id] - self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration']
    #         self.rcpsp_schedule[act_id]['end_time'] = activity_end_times[act_id]


class PartialSolution:
    def __init__(self,
                 task_mode: Dict[int, int]=None,
                 start_times: Dict[int, int]=None,
                 end_times: Dict[int, int]=None,
                 partial_permutation: List[int]=None,
                 list_partial_order: List[List[int]]=None):
        self.task_mode = task_mode
        self.start_times = start_times
        self.end_times = end_times
        self.partial_permutation = partial_permutation
        self.list_partial_order = list_partial_order
        # one element in self.list_partial_order is a list [l1, l2, l3]
        # indicating that l1 should be started before l1, and  l2 before l3 for example


class RCPSPModel(Problem):
    sgs: ScheduleGenerationScheme
    resources: Dict[str, int]  # {resource_name: number_of_resource}
    non_renewable_resources: List[str]  # e.g. [resource_name3, resource_name4]
    n_jobs: int  # excluding dummy activities Start (0) and End (n)
    # possible_modes: Dict[int, List[int]]  # {task_id: list_of_mode_ids}
    mode_details: Dict[int, Dict[int, Dict[str, int]]]  # e.g. {job_id: {mode_id: {resource_name1: number_of_resources_needed, resource_name2: ...}}
                                                        # one key being "duration"
    successors: Dict[int, List[int]]  # {task_id: list of successor task ids}
    def __init__(self,
                 resources: Dict[str, int],
                 non_renewable_resources: List[str],
                 mode_details: Dict[int, Dict[int, Dict[str, int]]],
                 successors: Dict[int, List[int]],
                 horizon,
                 horizon_multiplier=1,
                 name_task: Dict[int, str]=None):
        self.resources = resources
        self.resources_list = list(self.resources.keys())
        self.non_renewable_resources = non_renewable_resources
        self.mode_details = mode_details
        self.successors = successors
        self.n_jobs = len(self.mode_details.keys()) - 2
        self.horizon = horizon
        self.horizon_multiplier = horizon_multiplier
        self.name_task = name_task
        if name_task is None:
            self.name_task = {x: str(x) for x in self.mode_details}

    def is_rcpsp_multimode(self):
        if max([len(self.mode_details[key1].keys()) for key1 in self.mode_details.keys()]) > 1:
            return True
        else:
            return False

    def compute_graph(self)->Graph:
        nodes = [(n, {mode: self.mode_details[n][mode]["duration"]
                      for mode in self.mode_details[n]})
                 for n in range(1, self.n_jobs+3)]
        edges = []
        for n in self.successors:
            for succ in self.successors[n]:
                edges += [(n, succ, {})]
        return Graph(nodes, edges, False)

    # @abstractmethod
    def evaluate_function(self, rcpsp_sol: RCPSPSolution):
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        last_activity = max(list(rcpsp_sol.rcpsp_schedule.keys()))
        makespan = rcpsp_sol.rcpsp_schedule[last_activity]['end_time']
        obj_mean_resource_reserve = rcpsp_sol.compute_mean_resource_reserve()
        return makespan, obj_mean_resource_reserve

    @abstractmethod
    def evaluate_from_encoding(self, int_vector, encoding_name):
        ...

    def evaluate(self, rcpsp_sol: RCPSPSolution) -> Dict[str, float]:
        obj_makespan, obj_mean_resource_reserve = self.evaluate_function(rcpsp_sol)
        return {'makespan': obj_makespan, 'mean_resource_reserve': obj_mean_resource_reserve}

    def evaluate_mobj(self, rcpsp_sol: RCPSPSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(rcpsp_sol))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]):
        return TupleFitness(np.array([-dict_values["makespan"],
                                      dict_values["mean_resource_reserve"]]),
                            2)

    def satisfy(self, rcpsp_sol: RCPSPSolution)->bool:
        if rcpsp_sol.rcpsp_schedule_feasible is False:
            print('Schedule flagged as infeasible when generated')
            return False
        else:
            modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
            modes_extended.insert(0, 1)
            modes_extended.append(1)

            # Check for resource violation at each time step
            last_activity = max(list(rcpsp_sol.rcpsp_schedule.keys()))
            makespan = rcpsp_sol.rcpsp_schedule[last_activity]['end_time']
            for t in range(makespan):
                resource_usage = {}
                for res in self.resources.keys():
                    resource_usage[res] = 0
                for act_id in list(self.successors.keys()):
                    start = rcpsp_sol.rcpsp_schedule[act_id]['start_time']
                    end = rcpsp_sol.rcpsp_schedule[act_id]['end_time']
                    mode = modes_extended[act_id-1]
                    # print(act_id)
                    for res in self.resources.keys():#self.mode_details[act_id][mode]:
                        if start <= t and t < end:
                            # print('res: ', res)
                            # print('adding usage from act', act_id)
                            # print('mode:', mode)
                            # print('self.mode_details[act_id][mode][res]: ', self.mode_details[act_id][mode][res])
                            resource_usage[res] += self.mode_details[act_id][mode][res]

                for res in self.resources.keys():
                    if resource_usage[res] > self.resources[res]:
                        print('Time step resource violation: time: ', t, 'res', res,
                              'res_usage: ', resource_usage[res], 'res_avail: ', self.resources[res])
                        return False

            # Check for non-renewable resource violation
            for res in self.non_renewable_resources:
                usage = 0
                for act_id in list(self.successors.keys()):
                    mode = modes_extended[act_id-1]
                    usage += self.mode_details[act_id][mode][res]
                if usage > self.resources[res]:
                    print('Non-renewable resource violation: act_id: ', act_id, 'res', res, 'res_usage: ', resource_usage[res], 'res_avail: ', self.resources[res])
                    return False

            # Check precedences / successors
            for act_id in list(self.successors.keys()):
                for succ_id in self.successors[act_id]:
                    start_succ = rcpsp_sol.rcpsp_schedule[succ_id]['start_time']
                    end_pred = rcpsp_sol.rcpsp_schedule[act_id]['end_time']
                    if start_succ < end_pred:
                        print('Precedence relationship broken: ', act_id, 'end at ', end_pred, 'while ', succ_id, 'start at', start_succ)
                        return False

            return True

    def __str__(self):
        val = "RCPSP model"
        return val

    def get_solution_type(self):
        return RCPSPSolution

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {}
        dict_register["rcpsp_permutation"] = {"name": "rcpsp_permutation",
                                              "type": [TypeAttribute.PERMUTATION,
                                                       TypeAttribute.PERMUTATION_RCPSP],
                                              "range": range(self.n_jobs),
                                              "n": self.n_jobs}

        max_number_modes = max([len(list(self.mode_details[x].keys())) for x in self.mode_details.keys()])
        # print('max_number_modes: ', max_number_modes)
        dict_register["rcpsp_modes"] = {"name": "rcpsp_modes",
                                        "type": [TypeAttribute.LIST_INTEGER],
                                        "n": self.n_jobs,
                                        "arrity": max_number_modes}

        mode_arrity = [len(list(self.mode_details[list(self.mode_details.keys())[i]].keys()))
                       for i in range(1, len(self.mode_details.keys())-1)]
        dict_register["rcpsp_modes_arrity_fix"] = {"name": "rcpsp_modes",
                                                   "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY],
                                                   "n": self.n_jobs,
                                                   "arrities": mode_arrity}

        # TODO: New encoding with both permutation and modes in the same encoding
        #  (if this can be handled by DEAP ? Look at the initES example
        #  at https://deap.readthedocs.io/en/master/tutorials/basic/part1.html)

        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"makespan": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
                          "mean_resource_reserve": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}}
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.SINGLE,
                                 dict_objective_to_doc=dict_objective)

    def compute_resource_consumption(self, rcpsp_sol: RCPSPSolution):
        modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
        modes_extended.insert(0, 1)
        modes_extended.append(1)
        last_activity = max(rcpsp_sol.rcpsp_schedule)
        makespan = rcpsp_sol.rcpsp_schedule[last_activity]['end_time']
        consumptions = np.zeros((len(self.resources), makespan+1))
        for act_id in rcpsp_sol.rcpsp_schedule:
            for ir in range(len(self.resources)):
                use_ir = self.mode_details[act_id][modes_extended[act_id - 1]][self.resources_list[ir]]
                consumptions[ir, rcpsp_sol.rcpsp_schedule[act_id]["start_time"]+1:rcpsp_sol.rcpsp_schedule[act_id]["end_time"]+1] += use_ir
        return consumptions

    def plot_ressource_view(self, rcpsp_sol: RCPSPSolution):
        consumption = self.compute_resource_consumption(rcpsp_sol=rcpsp_sol)
        fig, ax = plt.subplots(nrows=len(self.resources_list), sharex=True)
        for i in range(len(self.resources_list)):
            ax[i].axhline(y=self.resources[self.resources_list[i]], label=self.resources_list[i])
            ax[i].plot(consumption[i, :])
            ax[i].legend()

    def copy(self):
        return RCPSPModel(resources=self.resources,
                          non_renewable_resources=self.non_renewable_resources,
                          mode_details=deepcopy(self.mode_details),
                          successors=deepcopy(self.successors),
                          horizon=self.horizon,
                          horizon_multiplier=self.horizon_multiplier)

    def get_dummy_solution(self):
        sol = RCPSPSolution(problem=self,
                            rcpsp_permutation=list(range(self.n_jobs)),
                            rcpsp_modes=[1 for i in range(self.n_jobs)])
        return sol

# Variable quantity of ressource.
# TODO :


class RCPSPModelCalendar(RCPSPModel):
    # TODO
    def evaluate_from_encoding(self, int_vector, encoding_name):
        # return 0
        if encoding_name == 'rcpsp_permutation':
            single_mode_list = [1 for i in range(self.n_jobs)]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes= single_mode_list)
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    resources: Dict[str, List[int]]

    def __init__(self, resources: Dict[str, List[int]],
                 non_renewable_resources: List[str],
                 mode_details: Dict[int, Dict[int, Dict[str, int]]],
                 successors: Dict[int, List[int]], horizon,
                 horizon_multiplier=1,
                 name_task: Dict[int, str] = None,
                 calendar_details: Dict[str, List[List[int]]]=None,
                 name_ressource_to_index: Dict[str, int]=None):
        super().__init__(resources,
                         non_renewable_resources,
                         mode_details,
                         successors,
                         horizon,
                         horizon_multiplier,
                         name_task)
        self.calendar_details = calendar_details
        self.name_ressource_to_index = name_ressource_to_index

    def copy(self):
        return RCPSPModelCalendar(resources={w: deepcopy(self.resources[w])
                                             for w in self.resources},
                                  non_renewable_resources=self.non_renewable_resources,
                                  mode_details=deepcopy(self.mode_details),
                                  successors=deepcopy(self.successors),
                                  horizon=self.horizon,
                                  horizon_multiplier=self.horizon_multiplier,
                                  calendar_details=deepcopy(self.calendar_details),
                                  name_ressource_to_index=self.name_ressource_to_index)

    def satisfy(self, rcpsp_sol: RCPSPSolution)->bool:
        if rcpsp_sol.rcpsp_schedule_feasible is False:
            print('Schedule flagged as infeasible when generated')
            return False
        else:
            modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
            modes_extended.insert(0, 1)
            modes_extended.append(1)
            # Check for resource violation at each time step
            last_activity = max(list(rcpsp_sol.rcpsp_schedule.keys()))
            makespan = rcpsp_sol.rcpsp_schedule[last_activity]['end_time']
            for t in range(makespan):
                resource_usage = {}
                for res in self.resources.keys():
                    resource_usage[res] = 0
                for act_id in list(self.successors.keys()):
                    start = rcpsp_sol.rcpsp_schedule[act_id]['start_time']
                    end = rcpsp_sol.rcpsp_schedule[act_id]['end_time']
                    mode = modes_extended[act_id-1]
                    # print(act_id)
                    for res in self.resources.keys():#self.mode_details[act_id][mode]:
                        if start <= t and t < end:
                            # print('res: ', res)
                            # print('adding usage from act', act_id)
                            # print('mode:', mode)
                            # print('self.mode_details[act_id][mode][res]: ', self.mode_details[act_id][mode][res])
                            resource_usage[res] += self.mode_details[act_id][mode][res]

                for res in self.resources.keys():
                    if resource_usage[res] > self.resources[res][t]:
                        print('Time step resource violation: time: ', t, 'res', res,
                              'res_usage: ', resource_usage[res], 'res_avail: ', self.resources[res])
                        return False

            # Check for non-renewable resource violation
            for res in self.non_renewable_resources:
                usage = 0
                for act_id in list(self.successors.keys()):
                    mode = modes_extended[act_id-1]
                    usage += self.mode_details[act_id][mode][res]
                if usage > self.resources[res][0]:
                    print('Non-renewable resource violation: act_id: ', act_id, 'res', res, 'res_usage: ', resource_usage[res], 'res_avail: ', self.resources[res])
                    return False

            # Check precedences / successors
            for act_id in list(self.successors.keys()):
                for succ_id in self.successors[act_id]:
                    start_succ = rcpsp_sol.rcpsp_schedule[succ_id]['start_time']
                    end_pred = rcpsp_sol.rcpsp_schedule[act_id]['end_time']
                    if start_succ < end_pred:
                        print('Precedence relationship broken: ', act_id, 'end at ', end_pred, 'while ', succ_id, 'start at', start_succ)
                        return False

            return True


class SingleModeRCPSPModel(RCPSPModel):
    def __init__(self, resources,
                 non_renewable_resources,
                 mode_details,
                 successors,
                 horizon,
                 horizon_multiplier=1):
        RCPSPModel.__init__(self, resources=resources,
                            non_renewable_resources=non_renewable_resources,
                            mode_details=mode_details,
                            successors=successors,
                            horizon=horizon,
                            horizon_multiplier=horizon_multiplier)

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'rcpsp_permutation':
            single_mode_list = [1 for i in range(self.n_jobs)]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes= single_mode_list)
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def copy(self):
        return SingleModeRCPSPModel(resources=self.resources,
                                    non_renewable_resources=self.non_renewable_resources,
                                    mode_details=deepcopy(self.mode_details),
                                    successors=deepcopy(self.successors),
                                    horizon=self.horizon,
                                    horizon_multiplier=self.horizon_multiplier)


class MultiModeRCPSPModel(RCPSPModel):
    fixed_modes: List[int]  # {task_id: mode_id} -
                            # Used to fix temporarily the task modes while an algorithm is searching the space of permutations
    fixed_permutation: Union[List[int], np.array]  # - Used to fix temporarily the permutation while an algorithm is searching the space of modes

    def __init__(self, resources, non_renewable_resources, mode_details, successors, horizon, horizon_multiplier=1):
        RCPSPModel.__init__(self, resources=resources,
                            non_renewable_resources=non_renewable_resources,
                            mode_details=mode_details,
                            successors=successors,
                            horizon=horizon,
                            horizon_multiplier=horizon_multiplier)
        self.fixed_modes = None
        self.fixed_permutation = None

    def set_fixed_attributes(self, encoding_str: str, sol: RCPSPSolution):
        att = self.get_attribute_register().dict_attribute_to_type[encoding_str]['name']
        if att == 'rcpsp_modes':
            self.set_fixed_modes(sol.rcpsp_modes)
            print('self.fixed_modes:', self.fixed_modes)
        elif att == 'rcpsp_permutation':
            self.set_fixed_permutation(sol.rcpsp_permutation)
            print('self.fixed_permutation:', self.fixed_permutation)

    def set_fixed_modes(self, fixed_modes):
        self.fixed_modes = fixed_modes

    def set_fixed_permutation(self, fixed_permutation):
        self.fixed_permutation = fixed_permutation

    # TODO: Should any evaluate_from_encoding function (in any problem) take a list of encoding as input ? - MRCPSP is an example of a problem with 2 encodings (priorities + modes). Although this will not be compatible with the basic DEAP GA implemented so far. We would need to add a multi-representation GA, possibly calling the DEAP single representation GA alternating between representations - If keeping the current single representation handling in evaluate_encoding, we need to change the variables corresponding to encosing_name only and keep the other one unchanged, then evaluate - this can work
    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'rcpsp_permutation':
            # change the permutation in the solution with int_vector and set the modes with self.fixed_modes
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes=self.fixed_modes)
        elif encoding_name == 'rcpsp_modes':
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = [x+1 for x in int_vector]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=self.fixed_permutation,
                                      rcpsp_modes=modes_corrected)
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def copy(self):
        mm = MultiModeRCPSPModel(resources=self.resources,
                                 non_renewable_resources=self.non_renewable_resources,
                                 mode_details=deepcopy(self.mode_details),
                                 successors=deepcopy(self.successors),
                                 horizon=self.horizon,
                                 horizon_multiplier=self.horizon_multiplier)
        mm.fixed_permutation = self.fixed_permutation
        mm.fixed_modes = self.fixed_modes
        return mm

    def get_dummy_solution(self):
        sol = RCPSPSolution(problem=self,
                            rcpsp_permutation=list(range(self.n_jobs)),
                            rcpsp_modes=[1 for i in range(self.n_jobs)])
        return sol


class Aggreg_RCPSPModel(RobustProblem, RCPSPModel):
    list_problems: List[RCPSPModel]

    def __init__(self, list_problem: List[RCPSPModel],
                 method_aggregating: MethodAggregating):
        RobustProblem.__init__(self, list_problem=list_problem, method_aggregating=method_aggregating)
        self.horizon = list_problem[0].horizon
        self.horizon_multiplier = list_problem[0].horizon_multiplier
        self.resources = list_problem[0].resources
        self.successors = list_problem[0].successors
        self.n_jobs = list_problem[0].n_jobs
        self.mode_details = list_problem[0].mode_details
        self.resources_list = list_problem[0].resources_list

    def get_dummy_solution(self):
        a: RCPSPSolution = self.list_problem[0].get_dummy_solution()
        a._schedule_to_recompute = True
        return a

    def get_unique_rcpsp_model(self)->RCPSPModel:
        # Create a unique rcpsp instance coherent with the aggregating method.
        model = self.list_problem[0].copy()
        for job in model.mode_details:
            for mode in model.mode_details[job]:
                for res in model.mode_details[job][mode]:
                    rs = np.array([self.list_problem[i].mode_details[job][mode][res]
                                   for i in range(self.nb_problem)])
                    agg = int(self.agg_vec(rs))
                    model.mode_details[job][mode][res] = agg
        return model

    def evaluate_from_encoding(self, int_vector, encoding_name):
        fits = [self.list_problem[i].evaluate_from_encoding(int_vector,
                                                            encoding_name)
                for i in range(self.nb_problem)]
        keys = fits[0].keys()
        aggreg = {}
        for k in keys:
            vals = np.array([fit[k] for fit in fits])
            aggreg[k] = self.agg_vec(vals)
        return aggreg

    def evaluate(self, variable: Solution):
        fits = []
        for i in range(self.nb_problem):
            var: RCPSPSolution = variable.lazy_copy()
            var.rcpsp_schedule = None
            var._schedule_to_recompute = True
            var.problem = self.list_problem[i]
            fit = self.list_problem[i].evaluate(var)
            fits += [fit]
        keys = fits[0].keys()
        aggreg = {}
        for k in keys:
            vals = np.array([fit[k] for fit in fits])
            aggreg[k] = self.agg_vec(vals)
        return aggreg


class MethodBaseRobustification(Enum):
    AVERAGE = 0
    WORST_CASE = 1
    BEST_CASE = 2
    PERCENTILE = 3
    SAMPLE = 4


class MethodRobustification:
    method_base: MethodBaseRobustification
    percentile: float

    def __init__(self,
                 method_base: MethodBaseRobustification,
                 percentile: float=50):
        self.method_base = method_base
        self.percentile = percentile


def create_poisson_laws_duration(rcpsp_model: RCPSPModel, range_around_mean=3):
    poisson_dict = {}
    source = 1
    sink = rcpsp_model.n_jobs+2
    for job in rcpsp_model.mode_details:
        poisson_dict[job] = {}
        for mode in rcpsp_model.mode_details[job]:
            poisson_dict[job][mode] = {}
            duration = rcpsp_model.mode_details[job][mode]["duration"]
            if job in {source, sink}:
                poisson_dict[job][mode]["duration"] = (duration, duration, duration)
            else:
                min_duration = max(1, duration-range_around_mean)
                max_duration = duration+range_around_mean
                poisson_dict[job][mode]["duration"] = (min_duration, duration, max_duration)
    return poisson_dict


def create_poisson_laws_resource(rcpsp_model: RCPSPModel, range_around_mean=1):
    poisson_dict = {}
    source = 1
    sink = rcpsp_model.n_jobs+2
    limit_resource = rcpsp_model.resources
    resources = rcpsp_model.resources_list
    resources_non_renewable = rcpsp_model.non_renewable_resources
    for job in rcpsp_model.mode_details:
        poisson_dict[job] = {}
        for mode in rcpsp_model.mode_details[job]:
            poisson_dict[job][mode] = {}
            for resource in rcpsp_model.mode_details[job][mode]:
                if resource == "duration":
                    continue
                if resource in resources_non_renewable:
                    continue
                resource_consumption = rcpsp_model.mode_details[job][mode][resource]
                if job in {source, sink}:
                    poisson_dict[job][mode][resource] = (resource_consumption, resource_consumption, resource_consumption)
                else:
                    min_rc = max(0,
                                 resource_consumption-range_around_mean)
                    max_rc = min(resource_consumption+range_around_mean,
                                 limit_resource[resource])
                    poisson_dict[job][mode][resource] = (min_rc, resource_consumption, max_rc)
    return poisson_dict


def create_poisson_laws(base_rcpsp_model: RCPSPModel,
                        range_around_mean_resource: int=1,
                        range_around_mean_duration: int=3,
                        do_uncertain_resource: bool=True,
                        do_uncertain_duration: bool=True):
    poisson_laws = tree()
    if do_uncertain_duration:
        poisson_laws_duration = create_poisson_laws_duration(base_rcpsp_model,
                                                             range_around_mean=range_around_mean_duration)
        for job in poisson_laws_duration:
            for mode in poisson_laws_duration[job]:
                for res in poisson_laws_duration[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_duration[job][mode][res]
    if do_uncertain_resource:
        poisson_laws_resource = create_poisson_laws_resource(base_rcpsp_model,
                                                             range_around_mean=range_around_mean_resource)
        for job in poisson_laws_resource:
            for mode in poisson_laws_resource[job]:
                for res in poisson_laws_resource[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_resource[job][mode][res]
    return poisson_laws


class UncertainRCPSPModel:
    def __init__(self,
                 base_rcpsp_model: RCPSPModel,
                 poisson_laws: Dict[int,
                                    Dict[int, Dict[str, Tuple[int, int, int]]]],
                 uniform_law=True # min, mean, max
                 ):
        self.base_rcpsp_model = base_rcpsp_model
        self.poisson_laws = poisson_laws
        self.probas = {}
        for activity in poisson_laws:
            self.probas[activity] = {}
            for mode in poisson_laws[activity]:
                self.probas[activity][mode] = {}
                for detail in poisson_laws[activity][mode]:
                    min_, mean_, max_ = poisson_laws[activity][mode][detail]
                    if uniform_law:
                        rv = randint(low=min_, high=max_+1)
                    else:
                        rv = poisson(mean_)
                    self.probas[activity][mode][detail] = {"value": np.arange(min_, max_+1, 1),
                                                           "proba": np.zeros((max_-min_+1))}
                    for k in range(len(self.probas[activity][mode][detail]["value"])):
                        self.probas[activity][mode][detail]["proba"][k] = \
                            rv.pmf(self.probas[activity][mode][detail]["value"][k])
                    self.probas[activity][mode][detail]["proba"] /= np.sum(self.probas[activity][mode][detail]["proba"])
                    self.probas[activity][mode][detail]["prob-distribution"] = rv_discrete(name=str(activity)+"-"
                                                                                                +str(mode)+"-"
                                                                                                +str(detail),
                                                                                           values=(self.probas[activity][mode][detail]["value"],
                                                                                                   self.probas[
                                                                                                       activity][mode][
                                                                                                       detail]["proba"]))

    def create_rcpsp_model(self, method_robustification: MethodRobustification):
        model = self.base_rcpsp_model.copy()
        for activity in self.probas:
            if activity in {self.base_rcpsp_model.n_jobs + 2, 1}:
                continue
            for mode in self.probas[activity]:
                for detail in self.probas[activity][mode]:
                    if method_robustification.method_base == MethodBaseRobustification.AVERAGE:
                        model.mode_details[activity][mode][detail] = \
                            int(self.probas[activity][mode][detail]["prob-distribution"].mean())
                    if method_robustification.method_base == MethodBaseRobustification.WORST_CASE:
                        model.mode_details[activity][mode][detail] = \
                            self.probas[activity][mode][detail]["prob-distribution"].support()[1]
                    if method_robustification.method_base == MethodBaseRobustification.BEST_CASE:
                        model.mode_details[activity][mode][detail] = \
                            self.probas[activity][mode][detail]["prob-distribution"].support()[0]
                    if method_robustification.method_base == MethodBaseRobustification.PERCENTILE:
                        model.mode_details[activity][mode][detail] = \
                            max(int(self.probas[activity][mode][detail]["prob-distribution"].isf(q=1-
                                                                                                 method_robustification.percentile/100)), 1)
                    if method_robustification.method_base == MethodBaseRobustification.SAMPLE:
                        model.mode_details[activity][mode][detail] = \
                            self.probas[activity][mode][detail]["prob-distribution"].rvs(size=1)[0]
        return model


class RCPSP_H_Model(SingleModeRCPSPModel):
    def __init__(self,
                 base_rcpsp_model: SingleModeRCPSPModel,
                 pre_helper_activities: Dict,
                 post_helper_activities: Dict):
        RCPSPModel.__init__(self, resources=base_rcpsp_model.resources,
                            non_renewable_resources=base_rcpsp_model.non_renewable_resources,
                            mode_details=base_rcpsp_model.mode_details,
                            successors=base_rcpsp_model.successors,
                            horizon=base_rcpsp_model.horizon,
                            horizon_multiplier=base_rcpsp_model.horizon_multiplier,
                            )
        self.pre_helper_activities = pre_helper_activities
        self.post_helper_activities = post_helper_activities
        # self.base_rcpsp_model = base_rcpsp_model

    def evaluate(self, rcpsp_sol: RCPSPSolution) -> Dict[str, float]:
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        rcpsp_sol.rcpsp_schedule = self.rcpsp_pre_helper_correction(rcpsp_sol)
        obj_makespan, obj_mean_resource_reserve = self.evaluate_function(rcpsp_sol)
        cumulated_helper_gap = 0
        for main_act_id in self.pre_helper_activities.keys():
            pre_gap = rcpsp_sol.rcpsp_schedule[main_act_id]['start_time'] - \
                      rcpsp_sol.rcpsp_schedule[self.pre_helper_activities[main_act_id][0]]['end_time']
            post_gap = rcpsp_sol.rcpsp_schedule[self.post_helper_activities[main_act_id][0]]['start_time'] - \
                       rcpsp_sol.rcpsp_schedule[main_act_id]['end_time']
            cumulated_helper_gap += pre_gap + post_gap

        return {'makespan': obj_makespan,
                'mean_resource_reserve': obj_mean_resource_reserve,
                'cumulated_helper_gap': cumulated_helper_gap}

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'rcpsp_permutation':
            single_mode_list = [1 for i in range(self.n_jobs)]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes=single_mode_list)
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def rcpsp_pre_helper_correction(self, rcpsp_sol: RCPSPSolution):
        corrected_sol = rcpsp_sol.copy()

        # sort pre_helper activities by start time decreasing
        pre_helper_ids = []
        pre_helper_starts = []
        for main_id in self.pre_helper_activities:
            pre_helper_ids.append(self.pre_helper_activities[main_id][0])
            pre_helper_starts.append(rcpsp_sol.rcpsp_schedule[self.pre_helper_activities[main_id][0]]['start_time'])
        # print('pre_helper_ids: ', pre_helper_ids)
        # print('pre_helper_starts: ', pre_helper_starts)
        sorted_pre_helper_ids = [x for _, x in sorted(zip(pre_helper_starts, pre_helper_ids), reverse=True)]
        # print('sorted_pre_helper_ids: ', sorted_pre_helper_ids)

        # for each pre_helper, try to start as late as possible
        for id in sorted_pre_helper_ids:
            # print('id: ',id)
            # print('original_start: ', corrected_sol.rcpsp_schedule[id]['start_time'])
            # print('self.successors[id]: ', self.successors[id])
            # Latest possible cannot be later than the earliest start of its successors
            all_successor_starts = [corrected_sol.rcpsp_schedule[s_id]['start_time'] for s_id in self.successors[id]]
            # print('all_successor_starts: ', all_successor_starts)
            latest_end = min(all_successor_starts)
            # print('initial latest_end: ',latest_end)
            duration = (corrected_sol.rcpsp_schedule[id]['end_time'] - corrected_sol.rcpsp_schedule[id]['start_time'])
            latest_start = latest_end - duration
            # print('initial latest_start:', latest_start)

            # print('self.compute_resource_consumption(): ', self.compute_resource_consumption(corrected_sol))
            # Then iteratively check if the latest time is suitable resource-wise
            # if not try earlier

            # first copy the resource consumption array and remove consumption of the pre_helper activity
            consumption = np.copy(self.compute_resource_consumption(corrected_sol))
            # print('self.resources: ', self.resources)
            for i in range(len(list(self.resources.keys()))):
                res_str = list(self.resources.keys())[i]
                # print('res_str: ', res_str)
                for t in range(corrected_sol.rcpsp_schedule[id]['start_time'], corrected_sol.rcpsp_schedule[id]['end_time']):
                    # print('t: ', t)
                    consumption[i,t+1] -= self.mode_details[id][1][res_str]

            # print('consumption -2: ', consumption)

            # then start trying iteratively to fit the pre_helper activity as late as possible
            stop = False
            while not stop:
                all_good = True
                for t in range(latest_start, latest_start+duration):
                    # print('t: ',t)
                    for i in range(len(list(self.resources.keys()))):
                        res_str = list(self.resources.keys())[i]
                        if consumption[i,t+1] + self.mode_details[id][1][res_str] > self.resources[res_str]:
                            all_good = False
                            break
                if all_good:
                    corrected_sol.rcpsp_schedule[id]['start_time'] = latest_start
                    corrected_sol.rcpsp_schedule[id]['end_time'] = latest_start+duration
                    # print('Corrected start: ',corrected_sol.rcpsp_schedule[id]['start_time'])
                    # print('Corrected end: ', corrected_sol.rcpsp_schedule[id]['end_time'])
                    stop = True
                else:
                    latest_start -= 1

        # print(' ---------- ')
        return corrected_sol.rcpsp_schedule


class MRCPSP_H_Model(MultiModeRCPSPModel):
    def __init__(self,
                 base_rcpsp_model: MultiModeRCPSPModel,
                 pre_helper_activities: Dict,
                 post_helper_activities: Dict
                 ):
        RCPSPModel.__init__(self, resources=base_rcpsp_model.resources,
                            non_renewable_resources=base_rcpsp_model.non_renewable_resources,
                            mode_details=base_rcpsp_model.mode_details,
                            successors=base_rcpsp_model.successors,
                            horizon=base_rcpsp_model.horizon,
                            horizon_multiplier=base_rcpsp_model.horizon_multiplier,
                            )
        self.pre_helper_activities = pre_helper_activities
        self.post_helper_activities = post_helper_activities

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'rcpsp_permutation':
            # change the permutation in the solution with int_vector and set the modes with self.fixed_modes
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=int_vector,
                                      rcpsp_modes=self.fixed_modes)
        elif encoding_name == 'rcpsp_modes':
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = [x+1 for x in int_vector]
            rcpsp_sol = RCPSPSolution(problem=self,
                                      rcpsp_permutation=self.fixed_permutation,
                                      rcpsp_modes=modes_corrected)
        objectives = self.evaluate(rcpsp_sol)
        return objectives






