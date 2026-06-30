# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Hashable
from typing import Union

from discrete_optimization.generic_tasks_tools.calendar_resource import (
    convert_calendar_to_availability_intervals,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp_multiskill.problem import (
    Employee,
    MultiskillRcpspProblem,
    SkillDetail,
    VariantMultiskillRcpspProblem,
)

from skdecide.builders.domain.scheduling.scheduling_domains import (
    MultiModeMultiSkillRCPSP,
    MultiModeMultiSkillRCPSPCalendar,
    MultiModeRCPSP,
    MultiModeRCPSPCalendar,
    MultiModeRCPSPWithCost,
    SchedulingDomain,
    SchedulingObjectiveEnum,
    SingleModeRCPSP,
    SingleModeRCPSP_Stochastic_Durations,
    SingleModeRCPSPCalendar,
    State,
)
from skdecide.hub.domain.rcpsp.rcpsp_sk import (
    MRCPSP,
    MSRCPSP,
    RCPSP,
    MRCPSPCalendar,
    MSRCPSPCalendar,
)

DOSchedulingProblem = (
    RcpspProblem | MultiskillRcpspProblem | GenericSchedulingImplProblem
)


def from_last_state_to_solution(
    state: State, domain: SchedulingDomain
) -> RcpspSolution:
    """Transform a scheduling state into a RcpspSolution
    This function reads the schedule from the state object and transform it back to a discrete-optimization solution
    object.
    """
    modes = [state.tasks_mode.get(j, 1) for j in sorted(domain.get_tasks_ids())]
    modes = modes[1:-1]
    schedule = {
        p.value.id: {"start_time": p.value.start, "end_time": p.value.end}
        for p in state.tasks_complete_details
    }
    return RcpspSolution(
        problem=build_do_domain(domain),
        rcpsp_permutation=None,
        rcpsp_modes=modes,
        rcpsp_schedule=schedule,
    )


def build_do_domain(
    scheduling_domain: Union[
        SingleModeRCPSP,
        SingleModeRCPSPCalendar,
        MultiModeRCPSP,
        MultiModeRCPSPWithCost,
        MultiModeRCPSPCalendar,
        MultiModeMultiSkillRCPSP,
        MultiModeMultiSkillRCPSPCalendar,
        SingleModeRCPSP_Stochastic_Durations,
        SchedulingDomain,
    ],
) -> DOSchedulingProblem:
    """Transform the scheduling domain (from scikit-decide) into a discrete-optimization problem.

    This only works for scheduling template given in the type docstring.
    """
    if isinstance(scheduling_domain, SingleModeRCPSP):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        resource_consumptions = {}
        for task in modes_details:
            resource_consumptions[task] = {}
            for mode in modes_details[task]:
                resource_consumptions[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    resource_consumptions[task][mode][r] = modes_details[task][
                        mode
                    ].get_resource_need_at_time(r, time=0)  # should be constant anyway
                resource_consumptions[task][mode]["duration"] = (
                    scheduling_domain.get_task_duration(task=task, mode=mode)
                )
        return RcpspProblem(
            resources={
                r: scheduling_domain.get_original_quantity_resource(r)
                for r in scheduling_domain.get_resource_types_names()
            },
            non_renewable_resources=[
                r
                for r in scheduling_domain.get_resource_renewability()
                if not scheduling_domain.get_resource_renewability()[r]
            ],
            mode_details=resource_consumptions,
            successors=scheduling_domain.get_successors(),
            horizon=scheduling_domain.get_max_horizon(),
        )
    elif isinstance(scheduling_domain, SingleModeRCPSP_Stochastic_Durations):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        resource_consumptions = {}
        for task in modes_details:
            resource_consumptions[task] = {}
            for mode in modes_details[task]:
                resource_consumptions[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    resource_consumptions[task][mode][r] = modes_details[task][
                        mode
                    ].get_resource_need_at_time(r, time=0)  # should be constant anyway
                resource_consumptions[task][mode]["duration"] = (
                    scheduling_domain.sample_task_duration(task=task, mode=mode)
                )
        return RcpspProblem(
            resources={
                r: scheduling_domain.get_original_quantity_resource(r)
                for r in scheduling_domain.get_resource_types_names()
            },
            non_renewable_resources=[
                r
                for r in scheduling_domain.get_resource_renewability()
                if not scheduling_domain.get_resource_renewability()[r]
            ],
            mode_details=resource_consumptions,
            successors=scheduling_domain.get_successors(),
            horizon=scheduling_domain.get_max_horizon(),
        )
    elif isinstance(scheduling_domain, (MultiModeRCPSP, MultiModeRCPSPWithCost)):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        resource_consumptions = {}
        for task in modes_details:
            resource_consumptions[task] = {}
            for mode in modes_details[task]:
                resource_consumptions[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    resource_consumptions[task][mode][r] = modes_details[task][
                        mode
                    ].get_resource_need_at_time(r, time=0)  # should be constant anyway
                resource_consumptions[task][mode]["duration"] = (
                    scheduling_domain.get_task_duration(task=task, mode=mode)
                )
        return RcpspProblem(
            resources={
                r: scheduling_domain.get_original_quantity_resource(r)
                for r in scheduling_domain.get_resource_types_names()
            },
            non_renewable_resources=[
                r
                for r in scheduling_domain.get_resource_renewability()
                if not scheduling_domain.get_resource_renewability()[r]
            ],
            mode_details=resource_consumptions,
            successors=scheduling_domain.get_successors(),
            horizon=scheduling_domain.get_max_horizon(),
        )
    elif isinstance(
        scheduling_domain, (MultiModeRCPSPCalendar, SingleModeRCPSPCalendar)
    ):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        resource_consumptions = {}
        for task in modes_details:
            resource_consumptions[task] = {}
            for mode in modes_details[task]:
                resource_consumptions[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    resource_consumptions[task][mode][r] = modes_details[task][
                        mode
                    ].get_resource_need_at_time(r, time=0)  # should be constant anyway
                resource_consumptions[task][mode]["duration"] = (
                    scheduling_domain.get_task_duration(task=task, mode=mode)
                )
        horizon = scheduling_domain.get_max_horizon()
        return RcpspProblem(
            resources={
                r: [
                    scheduling_domain.get_quantity_resource(r, time=t)
                    for t in range(horizon)
                ]
                for r in scheduling_domain.get_resource_types_names()
            },
            non_renewable_resources=[
                r
                for r in scheduling_domain.get_resource_renewability()
                if not scheduling_domain.get_resource_renewability()[r]
            ],
            mode_details=resource_consumptions,
            successors=scheduling_domain.get_successors(),
            horizon=scheduling_domain.get_max_horizon(),
        )
    elif isinstance(
        scheduling_domain, (MultiModeMultiSkillRCPSP, MultiModeMultiSkillRCPSPCalendar)
    ):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        skills_set = set()
        resource_consumptions = {}
        for task in modes_details:
            resource_consumptions[task] = {}
            for mode in modes_details[task]:
                resource_consumptions[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    resource_consumptions[task][mode][r] = modes_details[task][
                        mode
                    ].get_resource_need_at_time(r, time=0)  # should be constant anyway
                skills = scheduling_domain.get_skills_of_task(task=task, mode=mode)
                for s in skills:
                    resource_consumptions[task][mode][s] = skills[s]
                    skills_set.add(s)
                resource_consumptions[task][mode]["duration"] = (
                    scheduling_domain.get_task_duration(task=task, mode=mode)
                )
        horizon = scheduling_domain.get_max_horizon()
        employees_dict = {}
        employees = scheduling_domain.get_resource_units_names()
        sorted_employees = sorted(employees)
        for employee, i in zip(sorted_employees, range(len(sorted_employees))):
            skills = scheduling_domain.get_skills_of_resource(resource=employee)
            skills_details = {
                r: SkillDetail(skill_value=skills[r], efficiency_ratio=0, experience=0)
                for r in skills
            }
            employees_dict[i] = Employee(
                dict_skill=skills_details,
                calendar_employee=[
                    bool(scheduling_domain.get_quantity_resource(employee, time=t))
                    for t in range(horizon + 1)
                ],
            )

        return VariantMultiskillRcpspProblem(
            skills_set=scheduling_domain.get_skills_names(),
            resources_set=set(scheduling_domain.get_resource_types_names()),
            non_renewable_resources=set(
                [
                    r
                    for r in scheduling_domain.get_resource_renewability()
                    if not scheduling_domain.get_resource_renewability()[r]
                ]
            ),
            resources_availability={
                r: [
                    scheduling_domain.get_quantity_resource(r, time=t)
                    for t in range(horizon + 1)
                ]
                for r in scheduling_domain.get_resource_types_names()
            },
            employees=employees_dict,
            mode_details=resource_consumptions,
            successors=scheduling_domain.get_successors(),
            horizon=horizon,
            sink_task=max(scheduling_domain.get_tasks_ids()),
            source_task=min(scheduling_domain.get_tasks_ids()),
            one_unit_per_task_max=False,
        )
    else:
        horizon = scheduling_domain.get_max_horizon()
        resources = scheduling_domain.get_resource_types_names()
        renewability = scheduling_domain.get_resource_renewability()
        modes_details = scheduling_domain.get_tasks_modes()
        durations_per_mode = {
            task: {
                mode: scheduling_domain.sample_task_duration(task=task, mode=mode)
                for mode in task_modes_details
            }
            for task, task_modes_details in modes_details.items()
        }
        resource_consumptions = {}
        for task in modes_details:
            resource_consumptions[task] = {}
            for mode in modes_details[task]:
                resource_consumptions[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    resource_consumptions[task][mode][r] = modes_details[task][
                        mode
                    ].get_resource_need_at_time(
                        resource_name=r, time=0
                    )  # should be constant anyway
                skills = scheduling_domain.get_skills_of_task(task=task, mode=mode)
                for s in skills:
                    resource_consumptions[task][mode][s] = skills[s]
        non_renewable_resources = {
            r: scheduling_domain.sample_quantity_resource(resource=r, time=0)
            for r in resources
            if not renewability[r]
        }
        skills = scheduling_domain.get_skills_names()
        unary_resources = set(scheduling_domain.get_resource_units_names())
        if any(not renewability[r] for r in unary_resources):
            raise NotImplementedError(
                "Discrete-optimization cannot yet managed non-renewable unary resources."
            )
        unary_resources_skills = {
            resource: skills_details
            for resource, skills_details in scheduling_domain.get_all_resources_skills().items()
            if resource in unary_resources
        }
        unary_resources_availabilities = {
            r: [
                (start, end)
                for start, end, _ in convert_calendar_to_availability_intervals(
                    calendar=[
                        scheduling_domain.sample_quantity_resource(r, time=t)
                        for t in range(horizon)
                    ],
                    horizon=horizon,
                )
            ]
            for r in scheduling_domain.get_resource_types_names()
            if r not in skills and r not in non_renewable_resources
        }
        non_skill_cumulative_resources = {
            r: convert_calendar_to_availability_intervals(
                calendar=[
                    scheduling_domain.sample_quantity_resource(r, time=t)
                    for t in range(horizon)
                ],
                horizon=horizon,
            )
            for r in scheduling_domain.get_resource_types_names()
            if r not in skills
            and r not in non_renewable_resources
            and r not in unary_resources
        }
        successors = scheduling_domain.get_successors()
        time_windows = {
            task: (tw.earliest_start, tw.earliest_end, tw.latest_start, tw.latest_end)
            for task, tw in scheduling_domain.get_time_window().items()
        }
        end_to_start_min_time_lags = []
        start_to_end_min_time_lags = []
        for task1, timelags in scheduling_domain.get_time_lags().items():
            for task2, timelag in timelags.items():
                min_offset = timelag.minimum_time_lag
                max_offset = timelag.maximum_time_lag
                if min_offset is not None:
                    end_to_start_min_time_lags.append((task1, task2, min_offset))
                if max_offset is not None:
                    start_to_end_min_time_lags.append((task2, task1, -max_offset))

        # cost: use only integer costs
        mode_costs: dict[Hashable, dict[int, int]] = {
            task: {mode: int(cost) for mode, cost in task_mode_costs.items()}
            for task, task_mode_costs in scheduling_domain.get_mode_costs().items()
        }
        resource_cost_per_time_unit = (
            scheduling_domain.get_resource_cost_per_time_unit()
        )
        unary_resource_costs: dict[Hashable, dict[int, dict[str, int]]] = {
            task: {
                mode: {
                    unary_resource: duration
                    * resource_cost_per_time_unit[unary_resource]
                    for unary_resource in unary_resources
                }
                for mode, duration in task_durations_per_mode.items()
            }
            for task, task_durations_per_mode in durations_per_mode.items()
        }
        # objectives
        sk_objectives = scheduling_domain.get_objectives()
        do_objectives = []
        if SchedulingObjectiveEnum.MAKESPAN in sk_objectives:
            do_objectives.append((Objective.MAKESPAN, -1))
        if SchedulingObjectiveEnum.COST in sk_objectives:
            do_objectives.append((Objective.COST, -1))

        return GenericSchedulingImplProblem(
            horizon=horizon,
            durations_per_mode=durations_per_mode,
            resource_consumptions=resource_consumptions,
            successors=successors,
            non_renewable_resources=non_renewable_resources,
            skills=skills,
            unary_resources=unary_resources,
            unary_resources_skills=unary_resources_skills,
            unary_resources_availabilities=unary_resources_availabilities,
            non_skill_cumulative_resources=non_skill_cumulative_resources,
            time_windows=time_windows,
            start_to_end_min_time_lags=start_to_end_min_time_lags,
            end_to_start_min_time_lags=end_to_start_min_time_lags,
            objective=do_objectives,
            mode_costs=mode_costs,
            unary_resource_costs=unary_resource_costs,
        )


def build_sk_domain(
    rcpsp_do_domain: Union[MultiskillRcpspProblem, RcpspProblem],
    varying_ressource: bool = False,
) -> Union[RCPSP, MSRCPSP, MRCPSP, MSRCPSPCalendar]:
    """Build a scheduling domain (scikit-decide) from a discrete-optimization problem"""
    if (
        isinstance(rcpsp_do_domain, RcpspProblem)
        and rcpsp_do_domain.is_varying_resource()
    ):
        if varying_ressource:
            my_domain = MRCPSPCalendar(
                resource_names=rcpsp_do_domain.resources_list,
                tasks_mode=rcpsp_do_domain.mode_details,
                successors=rcpsp_do_domain.successors,
                max_horizon=rcpsp_do_domain.horizon,
                resource_availability=rcpsp_do_domain.resources,
                resource_renewable={
                    r: r not in rcpsp_do_domain.non_renewable_resources
                    for r in rcpsp_do_domain.resources_list
                },
            )
            return my_domain
        # Even if the DO domain is a calendar one... ignore it.
        else:
            my_domain = MRCPSP(
                resource_names=rcpsp_do_domain.resources_list,
                tasks_mode=rcpsp_do_domain.mode_details,
                successors=rcpsp_do_domain.successors,
                max_horizon=rcpsp_do_domain.horizon,
                resource_availability={
                    r: max(rcpsp_do_domain.resources[r])
                    for r in rcpsp_do_domain.resources
                },
                resource_renewable={
                    r: r not in rcpsp_do_domain.non_renewable_resources
                    for r in rcpsp_do_domain.resources_list
                },
            )
        return my_domain

    if (
        isinstance(rcpsp_do_domain, RcpspProblem)
        and not rcpsp_do_domain.is_rcpsp_multimode()
    ):
        my_domain = RCPSP(
            resource_names=rcpsp_do_domain.resources_list,
            tasks_mode=rcpsp_do_domain.mode_details,
            successors=rcpsp_do_domain.successors,
            max_horizon=rcpsp_do_domain.horizon,
            resource_availability=rcpsp_do_domain.resources,
            resource_renewable={
                r: r not in rcpsp_do_domain.non_renewable_resources
                for r in rcpsp_do_domain.resources_list
            },
        )
        return my_domain

    elif (
        isinstance(rcpsp_do_domain, RcpspProblem)
        and rcpsp_do_domain.is_rcpsp_multimode()
    ):
        my_domain = MRCPSP(
            resource_names=rcpsp_do_domain.resources_list,
            tasks_mode=rcpsp_do_domain.mode_details,
            successors=rcpsp_do_domain.successors,
            max_horizon=rcpsp_do_domain.horizon,
            resource_availability=rcpsp_do_domain.resources,
            resource_renewable={
                r: r not in rcpsp_do_domain.non_renewable_resources
                for r in rcpsp_do_domain.resources_list
            },
        )
        return my_domain

    elif isinstance(rcpsp_do_domain, MultiskillRcpspProblem):
        if not varying_ressource:
            resource_type_names = list(rcpsp_do_domain.resources_list)
            resource_skills = {r: {} for r in resource_type_names}
            resource_availability = {
                r: rcpsp_do_domain.resources_availability[r][0]
                for r in rcpsp_do_domain.resources_availability
            }
            resource_renewable = {
                r: r not in rcpsp_do_domain.non_renewable_resources
                for r in rcpsp_do_domain.resources_list
            }
            resource_unit_names = []
            for employee in rcpsp_do_domain.employees:
                resource_unit_names += [str(employee)]
                resource_skills[resource_unit_names[-1]] = {}
                resource_availability[resource_unit_names[-1]] = 1
                resource_renewable[resource_unit_names[-1]] = True
                for s in rcpsp_do_domain.employees[employee].dict_skill:
                    resource_skills[resource_unit_names[-1]][s] = (
                        rcpsp_do_domain.employees[employee].dict_skill[s].skill_value
                    )

            return MSRCPSP(
                skills_names=list(rcpsp_do_domain.skills_set),
                resource_unit_names=resource_unit_names,
                resource_type_names=resource_type_names,
                resource_skills=resource_skills,
                tasks_mode=rcpsp_do_domain.mode_details,
                successors=rcpsp_do_domain.successors,
                max_horizon=rcpsp_do_domain.horizon,
                resource_availability=resource_availability,
                resource_renewable=resource_renewable,
            )
        else:
            resource_type_names = list(rcpsp_do_domain.resources_list)
            resource_skills = {r: {} for r in resource_type_names}
            resource_availability = {
                r: rcpsp_do_domain.resources_availability[r]
                for r in rcpsp_do_domain.resources_availability
            }
            resource_renewable = {
                r: r not in rcpsp_do_domain.non_renewable_resources
                for r in rcpsp_do_domain.resources_list
            }
            resource_unit_names = []
            for employee in rcpsp_do_domain.employees:
                resource_unit_names += [str(employee)]
                resource_skills[resource_unit_names[-1]] = {}
                resource_availability[resource_unit_names[-1]] = [
                    1 if x else 0
                    for x in rcpsp_do_domain.employees[employee].calendar_employee
                ]
                resource_renewable[resource_unit_names[-1]] = True
                for s in rcpsp_do_domain.employees[employee].dict_skill:
                    resource_skills[resource_unit_names[-1]][s] = (
                        rcpsp_do_domain.employees[employee].dict_skill[s].skill_value
                    )

            return MSRCPSPCalendar(
                skills_names=list(rcpsp_do_domain.skills_set),
                resource_unit_names=resource_unit_names,
                resource_type_names=resource_type_names,
                resource_skills=resource_skills,
                tasks_mode=rcpsp_do_domain.mode_details,
                successors=rcpsp_do_domain.successors,
                max_horizon=rcpsp_do_domain.horizon,
                resource_availability=resource_availability,
                resource_renewable=resource_renewable,
            )
