from matplotlib.font_manager import FontProperties

from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, \
    MS_RCPSPSolution_Variant, MS_RCPSPSolution, MS_RCPSPSolution_Variant
from typing import List
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as pp
from matplotlib.collections import PatchCollection
import matplotlib.cm


def compute_schedule_per_resource_individual(rcpsp_model: MS_RCPSPModel,
                                             rcpsp_sol: MS_RCPSPSolution,
                                             verbose=False):
    modes = rcpsp_sol.modes
    sorted_task_by_start = sorted(rcpsp_sol.schedule,
                                  key=lambda x: 100000*rcpsp_sol.schedule[x]["start_time"]+x)
    sorted_task_by_end = sorted(rcpsp_sol.schedule,
                                key=lambda x: 100000*rcpsp_sol.schedule[x]["end_time"]+x)
    max_time = rcpsp_sol.schedule[sorted_task_by_end[-1]]["end_time"]
    min_time = rcpsp_sol.schedule[sorted_task_by_end[0]]["start_time"]
    print("Min time ", min_time)
    print("Max time ", max_time)
    employee_usage = {employee:
                          {"activity": np.zeros((max_time-min_time+1)),
                           "binary_activity": np.zeros((max_time-min_time+1)),
                           "total_activity": 0,
                           "boxes_time": []}
                      for employee in rcpsp_model.employees}
    total_time = max_time-min_time+1
    index_to_time = {i: min_time+i for i in range(max_time-min_time+1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}
    sorted_employees = list(sorted(rcpsp_model.employees))
    for activity in sorted_task_by_start:
        mode = modes[activity]
        start_time = rcpsp_sol.schedule[activity]["start_time"]
        end_time = rcpsp_sol.schedule[activity]["end_time"]
        for employee in rcpsp_sol.employee_usage.get(activity, {}):
            employee_usage[employee]["activity"][time_to_index[start_time]:time_to_index[end_time]] \
                = activity
            employee_usage[employee]["binary_activity"][time_to_index[start_time]:time_to_index[end_time]] \
                = 1
            employee_usage[employee]["total_activity"] += (end_time - start_time)
            index_employee = sorted_employees.index(employee)
            employee_usage[employee]["boxes_time"] += [[(index_employee - 0.25, start_time + 0.01, activity),
                                                        (index_employee - 0.25, end_time - 0.01, activity),
                                                        (index_employee + 0.25, end_time - 0.01, activity),
                                                        (index_employee + 0.25, start_time + 0.01, activity),
                                                        (index_employee - 0.25, start_time + 0.01, activity)]]
    return employee_usage


def plot_resource_individual_gantt(rcpsp_model: MS_RCPSPModel,
                                   rcpsp_sol: MS_RCPSPSolution,
                                   title_figure="",
                                   fig=None,
                                   ax=None,
                                   current_t=None):
    array_ressource_usage = compute_schedule_per_resource_individual(rcpsp_model,
                                                                     rcpsp_sol)
    sorted_task_by_start = sorted(rcpsp_sol.schedule,
                                  key=lambda x: 100000 * rcpsp_sol.schedule[x]["start_time"] + x)
    sorted_task_by_end = sorted(rcpsp_sol.schedule,
                                key=lambda x: 100000 * rcpsp_sol.schedule[x]["end_time"] + x)
    max_time = rcpsp_sol.schedule[sorted_task_by_end[-1]]["end_time"]
    min_time = rcpsp_sol.schedule[sorted_task_by_end[0]]["start_time"]
    sorted_employees = list(sorted(rcpsp_model.employees))
    # fig, ax = plt.subplots(len(array_ressource_usage),
    #                        figsize=(10, 5))
    # for i in range(len(array_ressource_usage)):
    #     ax[i].imshow(array_ressource_usage[resources_list[i]]["binary_activity"].T)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,
                               figsize=(12, 6))
        fig.suptitle(title_figure)
    position_label = {}
    for i in range(len(sorted_employees)):
        patches = []
        nb_colors = len(sorted_task_by_start)//2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        for boxe in array_ressource_usage[sorted_employees[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            x, y = polygon.exterior.xy
            ax.plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords,
                              facecolor=colors((activity-1) % nb_colors)))
            activity = boxe[0][2]
            if abs(boxe[0][1] - boxe[1][1]) >= 0.4:
                # (resource - 0.25, start_time + 0.01, activity),
                # (resource - 0.25, end_time - 0.01, activity),
                # (resource + 0.25, end_time - 0.01, activity),
                # (resource + 0.25, start_time + 0.01, activity),
                # (resource - 0.25, start_time + 0.01, activity)
                center = (sum([b[1] for b in boxe[:4]]) / 4 - 0.4, sum(b[0] for b in boxe[:4]) / 4)
                if activity not in position_label:
                    position_label[activity] = center
                position_label[activity] = max(center, position_label[activity])
        p = PatchCollection(patches,
                            match_original=True,
                            #cmap=matplotlib.cm.get_cmap('Blues'),
                            alpha=0.4)
        ax.add_collection(p)
        ax.set_xlim((min_time, max_time))
        ax.set_ylim((-0.5, len(sorted_employees)))
        ax.set_yticks(range(len(sorted_employees)))
        ax.set_yticklabels(tuple(sorted_employees),
                              fontdict={"size": 7})
        for activity in position_label:
            ax.annotate(activity,
                        xy=position_label[activity],
                        # textcoords="offset points",
                        font_properties=FontProperties(size=6,
                                                       weight="bold"),
                        verticalalignment='center',
                        horizontalalignment="left",
                        color="k",
                        clip_on=True)
        ax.grid(True)
        if current_t is not None:
            ax.axvline(x=current_t, label='pyplot vertical line', color='r', ls='--')
    return fig
