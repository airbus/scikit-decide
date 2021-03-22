# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution, RCPSPModel, RCPSPModelCalendar
from typing import List, Union
from copy import deepcopy
import numpy as np
import scipy.stats


def compute_resource_consumption(rcpsp_model: RCPSPModel,
                                 rcpsp_sol: RCPSPSolution,
                                 list_resources: List[Union[int, str]]=None,
                                 future_view=True):
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    last_activity = max(rcpsp_sol.rcpsp_schedule)
    makespan = rcpsp_sol.rcpsp_schedule[last_activity]['end_time']
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    consumptions = np.zeros((len(list_resources), makespan + 1))
    for act_id in rcpsp_sol.rcpsp_schedule:
        for ir in range(len(list_resources)):
            use_ir = rcpsp_model.mode_details[act_id][modes_extended[act_id - 1]][list_resources[ir]]
            if future_view:
                consumptions[ir, rcpsp_sol.rcpsp_schedule[act_id]["start_time"] + 1:rcpsp_sol.rcpsp_schedule[act_id][
                                                                                        "end_time"] + 1] += use_ir
            else:
                consumptions[ir, rcpsp_sol.rcpsp_schedule[act_id]["start_time"]:rcpsp_sol.rcpsp_schedule[act_id]["end_time"]] += use_ir

    return consumptions, np.arange(0, makespan+1, 1)


def compute_nice_resource_consumption(rcpsp_model: RCPSPModel, rcpsp_sol: RCPSPSolution,
                                      list_resources: List[Union[int, str]] = None):
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    c_future, times = compute_resource_consumption(rcpsp_model, rcpsp_sol,
                                                   list_resources=list_resources,
                                                   future_view=True)
    c_past, times = compute_resource_consumption(rcpsp_model, rcpsp_sol,
                                                 list_resources=list_resources,
                                                 future_view=False)
    merged_times = {i: [] for i in range(len(list_resources))}
    merged_cons = {i: [] for i in range(len(list_resources))}
    for r in range(len(list_resources)):
        for index_t in range(len(times)):
            merged_times[r] += [times[index_t], times[index_t]]
            merged_cons[r] += [c_future[r, index_t], c_past[r, index_t]]
    for r in merged_times:
        merged_times[r] = np.array(merged_times[r])
        merged_cons[r] = np.array(merged_cons[r])
    return merged_times, merged_cons


def compute_schedule_per_resource_individual(rcpsp_model: RCPSPModel,
                                             rcpsp_sol: RCPSPSolution,
                                             resource_types_to_consider: List[str]=None,
                                             verbose=False):
    nb_ressources = len(rcpsp_model.resources_list)
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    if resource_types_to_consider is None:
        resources = rcpsp_model.resources_list
    else:
        resources = resource_types_to_consider
    sorted_task_by_start = sorted(rcpsp_sol.rcpsp_schedule,
                                  key=lambda x: 100000*rcpsp_sol.rcpsp_schedule[x]["start_time"]+x)
    sorted_task_by_end = sorted(rcpsp_sol.rcpsp_schedule,
                                key=lambda x: 100000*rcpsp_sol.rcpsp_schedule[x]["end_time"]+x)
    max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["end_time"]
    min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[0]]["start_time"]
    print("Min time ", min_time)
    print("Max time ", max_time)
    with_calendar = isinstance(rcpsp_model, RCPSPModelCalendar)

    array_ressource_usage = {resources[i]:
                            {"activity":
                             np.zeros((max_time-min_time+1,
                                       max(rcpsp_model.resources[resources[i]])
                                       if with_calendar else rcpsp_model.resources[resources[i]])),
                             "binary_activity":
                             np.zeros((max_time - min_time + 1,
                                      max(rcpsp_model.resources[resources[i]])
                                      if with_calendar else rcpsp_model.resources[resources[i]])),
                             "total_activity":
                             np.zeros(max(rcpsp_model.resources[resources[i]])
                                      if with_calendar else rcpsp_model.resources[resources[i]]),
                             "activity_last_n_hours":
                             np.zeros((max_time-min_time+1,
                                       max(rcpsp_model.resources[resources[i]])
                                       if with_calendar else rcpsp_model.resources[resources[i]])),
                             "boxes_time": []
                             }
                             for i in range(len(resources))}
    total_time = max_time-min_time+1
    nhour = int(min(8, total_time/2-1))
    index_to_time = {i: min_time+i for i in range(max_time-min_time+1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}

    for activity in sorted_task_by_start:
        mode = modes_extended[activity-1]
        start_time = rcpsp_sol.rcpsp_schedule[activity]["start_time"]
        end_time = rcpsp_sol.rcpsp_schedule[activity]["end_time"]
        if end_time == start_time:
            continue
        resources_needed = {r: rcpsp_model.mode_details[activity][mode][r]
                            for r in resources}
        for r in resources_needed:
            if r not in array_ressource_usage:
                continue
            rneeded = resources_needed[r]
            if not with_calendar:
                range_interest = range(array_ressource_usage[r]["activity"].shape[1])
            else:
                # try:
                #     range_interest = [x for x in range(len(rcpsp_model.calendar_details[r])) if
                #                       rcpsp_model.calendar_details[r][x][time_to_index[start_time]] == 1]
                # except:
                range_interest = range(rcpsp_model.resources[r][time_to_index[start_time]])
            while rneeded > 0:
                # availables_people_r = [i for i in range(array_ressource_usage[r]["activity"].shape[1])
                #                        if array_ressource_usage[r]["activity"][time_to_index[start_time], i] == 0]
                availables_people_r = [i for i in range_interest
                                       if array_ressource_usage[r]["activity"][time_to_index[start_time], i] == 0]
                if verbose:
                    print(len(availables_people_r), " people available : ")
                if len(availables_people_r) > 0:
                    resource = min(availables_people_r,
                                   key=lambda x: array_ressource_usage[r]["total_activity"][x])
                    # greedy choice,
                    # the one who worked the less until now.
                    array_ressource_usage[r]["activity"][time_to_index[start_time]:time_to_index[end_time], resource] \
                        = activity
                    array_ressource_usage[r]["binary_activity"][time_to_index[start_time]:time_to_index[end_time], resource] \
                        = 1
                    array_ressource_usage[r]["total_activity"][resource] += (end_time-start_time)
                    array_ressource_usage[r]["activity_last_n_hours"][:, resource] = np.convolve(array_ressource_usage[r]["binary_activity"][:, resource],
                                                                                                 np.array([1]*nhour+[0]+[0]*nhour),
                                                                                                 mode="same")
                    array_ressource_usage[r]["boxes_time"] += [[(resource-0.25, start_time+0.01, activity),
                                                                (resource-0.25, end_time-0.01, activity),
                                                                (resource+0.25, end_time-0.01, activity),
                                                                (resource+0.25, start_time+0.01, activity),
                                                                (resource-0.25, start_time+0.01, activity)]]
                    # for plot purposes.
                    rneeded -= 1
                else:
                    print("r_needed ", rneeded)
                    print("Ressource needed : ", resources_needed)
                    print("ressource : ", r)
                    print("activity : ", activity)
                    print("Problem, can't build schedule")
                    print(array_ressource_usage[r]["activity"])
                    rneeded = 0

    return array_ressource_usage


# TODO: Check if the scipy version of KTD is the most meaningful for what we want to use it for (ktd between -1 and 1)
def kendall_tau_similarity(rcpsp_sols: (RCPSPSolution, RCPSPSolution)):
    sol1 = rcpsp_sols[0]
    sol2 = rcpsp_sols[1]

    perm1 = sol1.generate_permutation_from_schedule()
    perm2 = sol2.generate_permutation_from_schedule()

    ktd, p_value = scipy.stats.kendalltau(perm1, perm2)
    return ktd


def all_diff_start_time(rcpsp_sols: (RCPSPSolution, RCPSPSolution)):
    sol1 = rcpsp_sols[0]
    sol2 = rcpsp_sols[1]
    diffs = {}
    for act_id in sol1.rcpsp_schedule.keys():
        diff = sol1.rcpsp_schedule[act_id]['start_time'] - sol2.rcpsp_schedule[act_id]['start_time']
        diffs[act_id] = diff
    return diffs
