from skdecide.builders.discrete_optimization.generic_tools.graph_api import Graph
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution, RCPSPModel, RCPSPModelCalendar
from typing import List, Union
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as pp
from matplotlib.collections import PatchCollection
import matplotlib.cm
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


def plot_ressource_view(rcpsp_model: RCPSPModel,
                        rcpsp_sol: RCPSPSolution,
                        list_resource: List[Union[int, str]]=None,
                        title_figure="",
                        fig=None,
                        ax=None):
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    with_calendar = isinstance(rcpsp_model, RCPSPModelCalendar)
    if list_resource is None:
        list_resource = rcpsp_model.resources_list
    if ax is None:
        fig, ax = plt.subplots(nrows=len(list_resource),
                               figsize=(10, 5),
                               sharex=True)
        fig.suptitle(title_figure)
    polygons_ax = {i: [] for i in range(len(list_resource))}
    labels_ax = {i: [] for i in range(len(list_resource))}
    sorted_activities = sorted(rcpsp_sol.rcpsp_schedule, key=lambda x: rcpsp_sol.rcpsp_schedule[x]["start_time"])
    for j in sorted_activities:
        time_start = rcpsp_sol.rcpsp_schedule[j]["start_time"]
        time_end = rcpsp_sol.rcpsp_schedule[j]["end_time"]
        for i in range(len(list_resource)):
            cons = rcpsp_model.mode_details[j][modes_extended[j-1]][list_resource[i]]
            if cons == 0:
                continue
            bound = rcpsp_model.resources[list_resource[i]] if not with_calendar \
                    else max(rcpsp_model.resources[list_resource[i]])
            for k in range(0, bound):
                polygon = Polygon([(time_start, k), (time_end, k), (time_end, k+cons),
                                   (time_start, k+cons), (time_start, k)])
                areas = [p.intersection(polygon).area for p in polygons_ax[i]]
                if len(areas) == 0 or max(areas) == 0:
                    polygons_ax[i].append(polygon)
                    labels_ax[i].append(j)
                    break
    for i in range(len(list_resource)):
        patches = []
        for polygon in polygons_ax[i]:
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords))
        p = PatchCollection(patches, cmap=matplotlib.cm.get_cmap('Blues'),
                            alpha=0.4)
        ax[i].add_collection(p)
    merged_times, merged_cons = compute_nice_resource_consumption(rcpsp_model, rcpsp_sol,
                                                                  list_resources=list_resource)
    for i in range(len(list_resource)):
        ax[i].plot(merged_times[i], merged_cons[i], color="r", linewidth=2,
                   label="Consumption "+str(list_resource[i]), zorder=1)
        if not with_calendar:
            ax[i].axhline(y=rcpsp_model.resources[list_resource[i]], linestyle="--",
                          label="Limit : "+str(list_resource[i]), zorder=0)
        else:
            ax[i].plot(merged_times[i],
                       [rcpsp_model.resources[list_resource[i]][m]
                        for m in merged_times[i]], linestyle="--",
                       label="Limit : " + str(list_resource[i]), zorder=0)
        ax[i].legend(fontsize=5)
        lims = ax[i].get_xlim()
        ax[i].set_xlim([lims[0], 1.*lims[1]])
    return fig


def plot_task_gantt(rcpsp_model: RCPSPModel,
                    rcpsp_sol: RCPSPSolution,
                    fig=None,
                    ax=None,
                    current_t=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,
                               figsize=(10, 5))
        ax.set_title("Gantt Task")
    tasks = sorted(rcpsp_model.mode_details.keys())
    nb_task = len(tasks)
    sorted_task_by_start = sorted(rcpsp_sol.rcpsp_schedule,
                                  key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["start_time"] + x)
    sorted_task_by_end = sorted(rcpsp_sol.rcpsp_schedule,
                                key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["end_time"] + x)
    max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["end_time"]
    min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[0]]["start_time"]
    patches = []
    for j in range(nb_task):
        nb_colors = len(tasks)//2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        box = [(j-0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
                           (j-0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
                           (j+0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
                           (j+0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
                           (j-0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"])]
        polygon = Polygon([(b[1], b[0]) for b in box])
        x, y = polygon.exterior.xy
        ax.plot(x, y, zorder=-1, color="b")
        patches.append(pp(xy=polygon.exterior.coords,
                          facecolor=colors((j - 1) % nb_colors)))

    p = PatchCollection(patches,
                        match_original=True,
                        #cmap=matplotlib.cm.get_cmap('Blues'),
                        alpha=0.4)
    ax.add_collection(p)
    ax.set_xlim((min_time, max_time))
    ax.set_ylim((-0.5, nb_task))
    ax.set_yticks(range(nb_task))
    ax.set_yticklabels(tuple(["Task "+str(tasks[j]) for j in range(nb_task)]),
                       fontdict={"size": 7})
    return fig



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


def plot_resource_individual_gantt(rcpsp_model: RCPSPModel,
                                   rcpsp_sol: RCPSPSolution,
                                   resource_types_to_consider: List[str]=None,
                                   title_figure="",
                                   fig=None,
                                   ax=None,
                                   current_t=None):
    array_ressource_usage = compute_schedule_per_resource_individual(rcpsp_model,
                                                                     rcpsp_sol,
                                                                     resource_types_to_consider=
                                                                     resource_types_to_consider)
    sorted_task_by_start = sorted(rcpsp_sol.rcpsp_schedule,
                                  key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["start_time"] + x)
    sorted_task_by_end = sorted(rcpsp_sol.rcpsp_schedule,
                                key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["end_time"] + x)
    max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["end_time"]
    min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[0]]["start_time"]
    for key in list(array_ressource_usage.keys()):
        if np.sum(array_ressource_usage[key]["total_activity"]) == 0:
            array_ressource_usage.pop(key)
    resources_list = list(array_ressource_usage.keys())
    # fig, ax = plt.subplots(len(array_ressource_usage),
    #                        figsize=(10, 5))
    # for i in range(len(array_ressource_usage)):
    #     ax[i].imshow(array_ressource_usage[resources_list[i]]["binary_activity"].T)
    if fig is None or ax is None:
        fig, ax = plt.subplots(len(array_ressource_usage),
                               figsize=(10, 5))
        fig.suptitle(title_figure)
        if len(array_ressource_usage) == 1:
            ax = [ax]

    for i in range(len(resources_list)):
        patches = []
        nb_colors = len(sorted_task_by_start)//2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        for boxe in array_ressource_usage[resources_list[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords,
                              facecolor=colors((activity-1) % nb_colors)))
        p = PatchCollection(patches,
                            match_original=True,
                            #cmap=matplotlib.cm.get_cmap('Blues'),
                            alpha=0.4)
        ax[i].add_collection(p)
        ax[i].set_title(resources_list[i])
        ax[i].set_xlim((min_time, max_time))
        try:
            ax[i].set_ylim((-0.5, rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticks(range(rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticklabels(tuple([j for j in range(rcpsp_model.resources[resources_list[i]])]),
                                  fontdict={"size": 7})
        except:
            m = max(rcpsp_model.resources[resources_list[i]])
            ax[i].set_ylim((-0.5, m))
            ax[i].set_yticks(range(m))
            ax[i].set_yticklabels(tuple([j for j in range(m)]),
                                  fontdict={"size": 7})

        ax[i].grid(True)
        if current_t is not None:
            ax[i].axvline(x=current_t, label='pyplot vertical line', color='r', ls='--')
    return fig


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


def compute_graph_rcpsp(rcpsp_model: RCPSPModel):
    nodes = [(n, {mode: rcpsp_model.mode_details[n][mode]["duration"]
                  for mode in rcpsp_model.mode_details[n]})
             for n in range(1, rcpsp_model.n_jobs + 3)]
    edges = []
    for n in rcpsp_model.successors:
        for succ in rcpsp_model.successors[n]:
            dict_transition = {mode: rcpsp_model.mode_details[n][mode]["duration"]
                               for mode in rcpsp_model.mode_details[n]}
            min_duration = min(dict_transition.values())
            max_duration = max(dict_transition.values())
            dict_transition["min_duration"] = min_duration
            dict_transition["max_duration"] = max_duration
            dict_transition["minus_min_duration"] = -min_duration
            dict_transition["minus_max_duration"] = -max_duration
            dict_transition["link"] = 1
            edges += [(n, succ, dict_transition)]
    return Graph(nodes, edges, False)





