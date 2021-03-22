from copy import deepcopy
from typing import List, Union

import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pp
from skdecide.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution, RCPSPModelCalendar
from skdecide.discrete_optimization.rcpsp.rcpsp_utils import compute_nice_resource_consumption, \
    compute_schedule_per_resource_individual
from collections import namedtuple


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def area_intersection(a: Rectangle, b: Rectangle):
    # Solution picked here, to avoid shapely.
    # https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    return 0


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
                # polygon = Polygon([(time_start, k), (time_end, k), (time_end, k+cons),
                #                    (time_start, k+cons), (time_start, k)])
                polygon = Rectangle(xmin=time_start,
                                    ymin=k,
                                    xmax=time_end,
                                    ymax=k+cons)
                areas = [area_intersection(polygon, p) for p in polygons_ax[i]]
                if len(areas) == 0 or max(areas) == 0:
                    polygons_ax[i].append(polygon)
                    labels_ax[i].append(j)
                    break
    for i in range(len(list_resource)):
        patches = []
        for polygon in polygons_ax[i]:
            x = [polygon.xmin, polygon.xmax, polygon.xmax, polygon.xmin, polygon.xmin]
            y = [polygon.ymin, polygon.ymin, polygon.ymax, polygon.ymax, polygon.ymin]
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=[(xx, yy) for xx, yy in zip(x, y)]))
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
        # polygon = Polygon([(b[1], b[0]) for b in box])
        x = [xy[1] for xy in box]
        y = [xy[0] for xy in box]
        ax.plot(x, y, zorder=-1, color="b")
        patches.append(pp(xy=[(xx[1], xx[0]) for xx in box],
                          facecolor=colors((j - 1) % nb_colors)))

    p = PatchCollection(patches,
                        match_original=True,
                        # cmap=matplotlib.cm.get_cmap('Blues'),
                        alpha=0.4)
    ax.add_collection(p)
    ax.set_xlim((min_time, max_time))
    ax.set_ylim((-0.5, nb_task))
    ax.set_yticks(range(nb_task))
    ax.set_yticklabels(tuple(["Task "+str(tasks[j]) for j in range(nb_task)]),
                       fontdict={"size": 7})
    return fig


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
            # polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            # x, y = polygon.exterior.xy
            x = [xy[1] for xy in boxe]
            y = [xy[0] for xy in boxe]
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=[(xx[1], xx[0]) for xx in boxe],
                              facecolor=colors((activity-1) % nb_colors)))
        p = PatchCollection(patches,
                            match_original=True,
                            # cmap=matplotlib.cm.get_cmap('Blues'),
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
