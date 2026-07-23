from __future__ import annotations

import logging
from collections import defaultdict
from typing import Union

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspProblem

from skdecide.builders.domain.scheduling.scheduling_domains import (
    SchedulingObjectiveEnum,
)
from skdecide.hub.domain.scheduling.as_do_generic_scheduling import AsDOSchedulingDomain
from skdecide.hub.domain.scheduling.rcpsp import (
    MRCPSP,
    MSRCPSP,
    RCPSP,
    MRCPSPCalendar,
    MSRCPSPCalendar,
    RCPSPCalendar,
)

logger = logging.getLogger(__name__)


def build_sk_domain(
    do_problem: Union[
        MultiskillRcpspProblem, RcpspProblem, GenericSchedulingImplProblem
    ],
    varying_ressource: bool = True,
) -> Union[
    RCPSP,
    MSRCPSP,
    MRCPSP,
    MSRCPSPCalendar,
    RCPSPCalendar,
    MRCPSPCalendar,
    AsDOSchedulingDomain,
]:
    """Build a scheduling scikit-decide domain from a discrete-optimization problem.

    # Parameters
    do_problem:
    varying_ressource: if False, discard calendars of resources for rcpsp and rcpsp_multiskill

    """
    task2id = {task: task_id for task_id, task in enumerate(do_problem.tasks_list)}
    successors = {
        task2id[task]: [task2id[next_task] for next_task in next_tasks]
        for task, next_tasks in do_problem.successors.items()
    }
    horizon = do_problem.horizon

    if isinstance(do_problem, RcpspProblem):
        tasks_mode = {
            task2id[task]: task_mode_details
            for task, task_mode_details in do_problem.mode_details.items()
        }
        resource_names = do_problem.resources_list
        resource_renewable = {
            r: r not in do_problem.non_renewable_resources
            for r in do_problem.resources_list
        }
        if do_problem.is_varying_resource() and varying_ressource:
            resource_availability: dict[str, list[int]] = do_problem.resources
            if do_problem.is_rcpsp_multimode():
                return MRCPSPCalendar(
                    resource_names=resource_names,
                    tasks_mode=tasks_mode,
                    successors=successors,
                    max_horizon=horizon,
                    resource_availability=resource_availability,
                    resource_renewable=resource_renewable,
                )
            else:
                return RCPSPCalendar(
                    resource_names=resource_names,
                    tasks_mode=tasks_mode,
                    successors=successors,
                    max_horizon=horizon,
                    resource_availability=resource_availability,
                    resource_renewable=resource_renewable,
                )
        else:
            resource_max_capacity: dict[str, int]
            if do_problem.is_varying_resource():
                resource_max_capacity = {
                    r: max(do_problem.resources[r]) for r in do_problem.resources
                }
            else:
                resource_max_capacity = do_problem.resources
            if do_problem.is_rcpsp_multimode():
                return MRCPSP(
                    resource_names=resource_names,
                    tasks_mode=tasks_mode,
                    successors=successors,
                    max_horizon=horizon,
                    resource_availability=resource_max_capacity,
                    resource_renewable=resource_renewable,
                )
            else:
                return RCPSP(
                    resource_names=resource_names,
                    tasks_mode=tasks_mode,
                    successors=successors,
                    max_horizon=horizon,
                    resource_availability=resource_max_capacity,
                    resource_renewable=resource_renewable,
                )

    elif isinstance(do_problem, MultiskillRcpspProblem):
        skills_names = list(do_problem.skills_set)
        resource_type_names = list(do_problem.resources_list)
        resource_skills = {r: {} for r in resource_type_names}
        resource_renewable = {
            r: r not in do_problem.non_renewable_resources
            for r in do_problem.resources_list
        }
        resource_unit_names = []
        for employee in do_problem.employees:
            employee_name = str(employee)
            resource_unit_names.append(employee_name)
            resource_skills[employee_name] = {}
            resource_renewable[employee_name] = True
            for s in do_problem.employees[employee].dict_skill:
                resource_skills[employee_name][s] = (
                    do_problem.employees[employee].dict_skill[s].skill_value
                )
        tasks_mode = {
            task2id[task]: task_mode_details
            for task, task_mode_details in do_problem.mode_details.items()
        }
        if not varying_ressource:
            resource_max_capacity: dict[str, int] = {
                r: do_problem.resources_availability[r][0]
                for r in do_problem.resources_availability
            } | {unary_resource: 1 for unary_resource in resource_unit_names}

            return MSRCPSP(
                skills_names=skills_names,
                resource_unit_names=resource_unit_names,
                resource_type_names=resource_type_names,
                resource_skills=resource_skills,
                tasks_mode=tasks_mode,
                successors=successors,
                max_horizon=horizon,
                resource_availability=resource_max_capacity,
                resource_renewable=resource_renewable,
            )
        else:
            resource_availability = do_problem.resources_availability | {
                str(employee): [
                    int(value) for value in employee_details.calendar_employee
                ]
                for employee, employee_details in do_problem.employees.items()
            }

            return MSRCPSPCalendar(
                skills_names=skills_names,
                resource_unit_names=resource_unit_names,
                resource_type_names=resource_type_names,
                resource_skills=resource_skills,
                tasks_mode=tasks_mode,
                successors=successors,
                max_horizon=horizon,
                resource_availability=resource_availability,
                resource_renewable=resource_renewable,
            )
    elif isinstance(do_problem, GenericSchedulingImplProblem):
        duration_per_modes = {
            task2id[task]: durations
            for task, durations in do_problem.durations_per_mode.items()
        }
        resource_consumptions = {
            task2id[task]: consos
            for task, consos in do_problem.resource_consumptions.items()
        }

        sk_objectives_weights = defaultdict(int)
        for objective, weight in do_problem.weighted_objectives:
            if objective == Objective.MAKESPAN:
                sk_objectives_weights[SchedulingObjectiveEnum.MAKESPAN] += weight
            elif objective == Objective.COST:
                sk_objectives_weights[SchedulingObjectiveEnum.COST] += weight
            else:
                raise NotImplementedError()
        sk_objectives = []
        for sk_objective, weight in sk_objectives_weights.items():
            if weight not in (0, -1):
                raise NotImplementedError(
                    "Can only translate trivial objective weights"
                )
            if weight == -1:
                sk_objectives.append(sk_objective)

        time_windows = {
            task2id[task]: tw for task, tw in do_problem.time_windows.items()
        }
        end_to_start_min_time_lags = [
            (task2id[t1], task2id[t2], offset)
            for t1, t2, offset in do_problem.end_to_start_min_time_lags
        ]
        end_to_start_max_time_lags = [
            (task2id[t2], task2id[t1], -offset)
            for t1, t2, offset in do_problem.start_to_end_min_time_lags
        ]
        mode_costs = {
            task2id[task]: costs for task, costs in do_problem.mode_costs.items()
        }
        unary_resource_costs = {
            task2id[task]: costs
            for task, costs in do_problem.unary_resource_costs.items()
        }

        # drop calendars if necessary:
        if varying_ressource:
            unary_resources_availabilities = do_problem.unary_resources_availabilities
            non_skill_cumulative_resources = do_problem.non_skill_cumulative_resources
        else:
            unary_resources_availabilities = {}
            non_skill_cumulative_resources = {
                resource: do_problem.get_resource_max_capacity(resource)
                for resource in do_problem.non_skill_cumulative_resources_list
            }

        # warnings for constraints not handled in skdecide
        if len(do_problem.no_overlap_sets) > 0:
            logger.warning(
                "No overlap constraints not taken into account by AsDOSchedulingDomain"
            )
        if (
            len(do_problem.end_to_end_min_time_lags)
            + len(do_problem.start_to_start_min_time_lags)
        ) > 0:
            logger.warning(
                "End to end and start_to start time lags constraints not taken into account by AsDOSchedulingDomain"
            )

        return AsDOSchedulingDomain(
            horizon=horizon,
            durations_per_mode=duration_per_modes,
            resource_consumptions=resource_consumptions,
            successors=successors,
            unary_resources=list(do_problem.unary_resources),
            unary_resources_skills=do_problem.unary_resources_skills,
            unary_resources_availabilities=unary_resources_availabilities,
            skills=do_problem.skills,
            non_skill_cumulative_resources=non_skill_cumulative_resources,
            non_renewable_resources=do_problem.non_renewable_resources,
            time_windows=time_windows,
            end_to_start_min_time_lags=end_to_start_min_time_lags,
            end_to_start_max_time_lags=end_to_start_max_time_lags,
            mode_costs=mode_costs,
            unary_resource_costs=unary_resource_costs,
            objective=sk_objectives,
        )
    else:
        raise NotImplementedError()
