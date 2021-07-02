# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union

from skdecide.builders.domain.scheduling.scheduling_domains import SchedulingDomain, \
    SingleModeRCPSP, SingleModeRCPSPCalendar, MultiModeRCPSP, MultiModeRCPSPCalendar,\
    MultiModeMultiSkillRCPSPCalendar, MultiModeMultiSkillRCPSP, MultiModeRCPSPWithCost, State, \
    SingleModeRCPSP_Stochastic_Durations
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP, MRCPSP, MSRCPSP, MRCPSPCalendar, MSRCPSPCalendar
from skdecide.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, SingleModeRCPSPModel, \
    MultiModeRCPSPModel, RCPSPModelCalendar, RCPSPSolution
from skdecide.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, \
    MS_RCPSPModel_Variant, Employee, \
    SkillDetail


def from_last_state_to_solution(state: State, domain: SchedulingDomain):
    modes = [state.tasks_mode.get(j, 1) for j in sorted(domain.get_tasks_ids()) ]
    modes = modes[1:-1]
    schedule = {j: {"start_time": state.tasks_details[j].start,
                    "end_time": state.tasks_details[j].end}
                for j in state.tasks_details}
    return RCPSPSolution(problem=build_do_domain(domain),
                         rcpsp_permutation=None,
                         rcpsp_modes=modes,
                         rcpsp_schedule=schedule)


def build_do_domain(scheduling_domain: Union[SingleModeRCPSP,
                                             SingleModeRCPSPCalendar,
                                             MultiModeRCPSP,
                                             MultiModeRCPSPWithCost,
                                             MultiModeRCPSPCalendar,
                                             MultiModeMultiSkillRCPSP,
                                             MultiModeMultiSkillRCPSPCalendar,
                                             SingleModeRCPSP_Stochastic_Durations]):
    if isinstance(scheduling_domain, SingleModeRCPSP):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        mode_details_do = {}
        for task in modes_details:
            mode_details_do[task] = {}
            for mode in modes_details[task]:
                mode_details_do[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    mode_details_do[task][mode][r] = modes_details[task][mode].get_resource_need_at_time(r, time=0) # should be constant anyway
                mode_details_do[task][mode]["duration"] = scheduling_domain.get_task_duration(task=task, mode=mode)
        return SingleModeRCPSPModel(resources={r: scheduling_domain.get_original_quantity_resource(r)
                                               for r in scheduling_domain.get_resource_types_names()} ,
                                    non_renewable_resources=[r
                                                             for r in scheduling_domain.get_resource_renewability()
                                                             if not scheduling_domain.get_resource_renewability()[r]],
                                    mode_details=mode_details_do,
                                    successors=scheduling_domain.get_successors(),
                                    horizon=scheduling_domain.get_max_horizon(),
                                    horizon_multiplier=1)
    if isinstance(scheduling_domain, SingleModeRCPSP_Stochastic_Durations):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        mode_details_do = {}
        for task in modes_details:
            mode_details_do[task] = {}
            for mode in modes_details[task]:
                mode_details_do[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    mode_details_do[task][mode][r] = modes_details[task][mode].get_resource_need_at_time(r, time=0) # should be constant anyway
                mode_details_do[task][mode]["duration"] = scheduling_domain.sample_task_duration(task=task, mode=mode)
        return SingleModeRCPSPModel(resources={r: scheduling_domain.get_original_quantity_resource(r)
                                               for r in scheduling_domain.get_resource_types_names()} ,
                                    non_renewable_resources=[r
                                                             for r in scheduling_domain.get_resource_renewability()
                                                             if not scheduling_domain.get_resource_renewability()[r]],
                                    mode_details=mode_details_do,
                                    successors=scheduling_domain.get_successors(),
                                    horizon=scheduling_domain.get_max_horizon(),
                                    horizon_multiplier=1)
    if isinstance(scheduling_domain, (MultiModeRCPSP, MultiModeRCPSPWithCost)):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        mode_details_do = {}
        for task in modes_details:
            mode_details_do[task] = {}
            for mode in modes_details[task]:
                mode_details_do[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    mode_details_do[task][mode][r] = modes_details[task][mode].get_resource_need_at_time(r, time=0) # should be constant anyway
                mode_details_do[task][mode]["duration"] = scheduling_domain.get_task_duration(task=task, mode=mode)
        return MultiModeRCPSPModel(resources={r: scheduling_domain.get_original_quantity_resource(r)
                                               for r in scheduling_domain.get_resource_types_names()} ,
                                   non_renewable_resources=[r
                                                            for r in scheduling_domain.get_resource_renewability()
                                                            if not scheduling_domain.get_resource_renewability()[r]],
                                   mode_details=mode_details_do,
                                   successors=scheduling_domain.get_successors(),
                                   horizon=scheduling_domain.get_max_horizon(),
                                   horizon_multiplier=1)
    if isinstance(scheduling_domain, (MultiModeRCPSPCalendar, SingleModeRCPSPCalendar)):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        mode_details_do = {}
        for task in modes_details:
            mode_details_do[task] = {}
            for mode in modes_details[task]:
                mode_details_do[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    mode_details_do[task][mode][r] = modes_details[task][mode].get_resource_need_at_time(r,
                                                                                                         time=0)  # should be constant anyway
                mode_details_do[task][mode]["duration"] = scheduling_domain.get_task_duration(task=task, mode=mode)
        horizon = scheduling_domain.get_max_horizon()
        return RCPSPModelCalendar(resources={r: [scheduling_domain.get_quantity_resource(r, time=t)
                                                 for t in range(horizon)]
                                             for r in scheduling_domain.get_resource_types_names()},
                                  non_renewable_resources=[r
                                                           for r in scheduling_domain.get_resource_renewability()
                                                           if not scheduling_domain.get_resource_renewability()[r]],
                                  mode_details=mode_details_do,
                                  successors=scheduling_domain.get_successors(),
                                  horizon=scheduling_domain.get_max_horizon(),
                                  horizon_multiplier=1)
    if isinstance(scheduling_domain, (MultiModeMultiSkillRCPSP, MultiModeMultiSkillRCPSPCalendar)):
        modes_details = scheduling_domain.get_tasks_modes().copy()
        skills_set = set()
        mode_details_do = {}
        for task in modes_details:
            mode_details_do[task] = {}
            for mode in modes_details[task]:
                mode_details_do[task][mode] = {}
                for r in modes_details[task][mode].get_ressource_names():
                    mode_details_do[task][mode][r] = modes_details[task][mode].get_resource_need_at_time(r,
                                                                                                         time=0)  # should be constant anyway
                skills = scheduling_domain.get_skills_of_task(task=task, mode=mode)
                for s in skills:
                    mode_details_do[task][mode][s] = skills[s]
                    skills_set.add(s)
                mode_details_do[task][mode]["duration"] = scheduling_domain.get_task_duration(task=task, mode=mode)
        horizon = scheduling_domain.get_max_horizon()
        employees_dict = {}
        employees = scheduling_domain.get_resource_units_names()
        sorted_employees = sorted(employees)
        print(sorted_employees)
        for employee, i in zip(sorted_employees, range(len(sorted_employees))):
            skills = scheduling_domain.get_skills_of_resource(resource=employee)
            skills_details = {r: SkillDetail(skill_value=skills[r],
                                             efficiency_ratio=0,
                                             experience=0)
                              for r in skills}
            employees_dict[i] = Employee(dict_skill=skills_details,
                                         calendar_employee=[bool(scheduling_domain.get_quantity_resource(employee,
                                                                                                         time=t))
                                                            for t in range(horizon+1)])

        return MS_RCPSPModel_Variant(skills_set=scheduling_domain.get_skills_names(),
                                     resources_set=set(scheduling_domain.get_resource_types_names()),
                                     non_renewable_resources=set([r
                                                                  for r in scheduling_domain.get_resource_renewability()
                                                                  if not scheduling_domain.get_resource_renewability()[r]]),
                                     resources_availability={r: [scheduling_domain.get_quantity_resource(r, time=t)
                                                                 for t in range(horizon+1)]
                                                             for r in scheduling_domain.get_resource_types_names()},
                                     employees=employees_dict,
                                     employees_availability=[sum([scheduling_domain.get_quantity_resource(employee,
                                                                                                          time=t)
                                                                  for employee in employees]) for t in range(horizon+1)],
                                     mode_details=mode_details_do,
                                     successors=scheduling_domain.get_successors(),
                                     horizon=horizon,
                                     horizon_multiplier=1,
                                     sink_task=max(scheduling_domain.get_tasks_ids()),
                                     source_task=min(scheduling_domain.get_tasks_ids()),
                                     one_unit_per_task_max=False)
        # TODO : for imopse this should be True


def build_sk_domain(rcpsp_do_domain: Union[MS_RCPSPModel,
                                           SingleModeRCPSPModel,
                                           MultiModeRCPSP,
                                           RCPSPModelCalendar],
                    varying_ressource: bool=False):
    if isinstance(rcpsp_do_domain, RCPSPModelCalendar):
        if varying_ressource:
            my_domain = MRCPSPCalendar(resource_names=rcpsp_do_domain.resources_list,
                                       task_ids=sorted(rcpsp_do_domain.mode_details.keys()),
                                       tasks_mode=rcpsp_do_domain.mode_details,
                                       successors=rcpsp_do_domain.successors,
                                       max_horizon=rcpsp_do_domain.horizon,
                                       resource_availability=rcpsp_do_domain.resources,
                                       resource_renewable={r: r not in rcpsp_do_domain.non_renewable_resources
                                                           for r in rcpsp_do_domain.resources_list})
            return my_domain
        # Even if the DO domain is a calendar one... ignore it.
        else:
            my_domain = MRCPSP(resource_names=rcpsp_do_domain.resources_list,
                               task_ids=sorted(rcpsp_do_domain.mode_details.keys()),
                               tasks_mode=rcpsp_do_domain.mode_details,
                               successors=rcpsp_do_domain.successors,
                               max_horizon=rcpsp_do_domain.horizon,
                               resource_availability={r: max(rcpsp_do_domain.resources[r])
                                                      for r in rcpsp_do_domain.resources},
                               resource_renewable={r: r not in rcpsp_do_domain.non_renewable_resources
                                                   for r in rcpsp_do_domain.resources_list})
        return my_domain

    if isinstance(rcpsp_do_domain, SingleModeRCPSPModel):
        my_domain = RCPSP(resource_names=rcpsp_do_domain.resources_list,
                          task_ids=sorted(rcpsp_do_domain.mode_details.keys()),
                          tasks_mode=rcpsp_do_domain.mode_details,
                          successors=rcpsp_do_domain.successors,
                          max_horizon=rcpsp_do_domain.horizon,
                          resource_availability=rcpsp_do_domain.resources,
                          resource_renewable={r: r not in rcpsp_do_domain.non_renewable_resources
                                              for r in rcpsp_do_domain.resources_list})
        return my_domain

    elif isinstance(rcpsp_do_domain, (MultiModeRCPSPModel, RCPSPModel)):
         my_domain = MRCPSP(resource_names=rcpsp_do_domain.resources_list,
                            task_ids=sorted(rcpsp_do_domain.mode_details.keys()),
                            tasks_mode=rcpsp_do_domain.mode_details,
                            successors=rcpsp_do_domain.successors,
                            max_horizon=rcpsp_do_domain.horizon,
                            resource_availability=rcpsp_do_domain.resources,
                            resource_renewable={r: r not in rcpsp_do_domain.non_renewable_resources
                                                for r in rcpsp_do_domain.resources_list})
         return my_domain

    elif isinstance(rcpsp_do_domain, MS_RCPSPModel):
        if not varying_ressource:
            resource_type_names = list(rcpsp_do_domain.resources_list)
            resource_skills = {r: {} for r in resource_type_names}
            resource_availability = {r: rcpsp_do_domain.resources_availability[r][0]
                                     for r in rcpsp_do_domain.resources_availability}
            resource_renewable = {r: r not in rcpsp_do_domain.non_renewable_resources
                                  for r in rcpsp_do_domain.resources_list}
            resource_unit_names = []
            for employee in rcpsp_do_domain.employees:
                resource_unit_names += [str(employee)]
                resource_skills[resource_unit_names[-1]] = {}
                resource_availability[resource_unit_names[-1]] = 1
                resource_renewable[resource_unit_names[-1]] = True
                for s in rcpsp_do_domain.employees[employee].dict_skill:
                    resource_skills[resource_unit_names[-1]][s] = rcpsp_do_domain.employees[employee].dict_skill[
                        s].skill_value

            return MSRCPSP(skills_names=list(rcpsp_do_domain.skills_set),
                           resource_unit_names=resource_unit_names,
                           resource_type_names=resource_type_names,
                           resource_skills=resource_skills,
                           task_ids=sorted(rcpsp_do_domain.mode_details.keys()),
                           tasks_mode=rcpsp_do_domain.mode_details,
                           successors=rcpsp_do_domain.successors,
                           max_horizon=rcpsp_do_domain.horizon,
                           resource_availability=resource_availability,
                           resource_renewable=resource_renewable)
        else:
            resource_type_names = list(rcpsp_do_domain.resources_list)
            resource_skills = {r: {} for r in resource_type_names}
            resource_availability = {r: rcpsp_do_domain.resources_availability[r]
                                     for r in rcpsp_do_domain.resources_availability}
            resource_renewable = {r: r not in rcpsp_do_domain.non_renewable_resources
                                  for r in rcpsp_do_domain.resources_list}
            resource_unit_names = []
            for employee in rcpsp_do_domain.employees:
                resource_unit_names += [str(employee)]
                resource_skills[resource_unit_names[-1]] = {}
                resource_availability[resource_unit_names[-1]] = [1 if x else 0
                                                                  for x in
                                                                  rcpsp_do_domain.employees[employee].calendar_employee]
                resource_renewable[resource_unit_names[-1]] = True
                for s in rcpsp_do_domain.employees[employee].dict_skill:
                    resource_skills[resource_unit_names[-1]][s] = rcpsp_do_domain.employees[employee].dict_skill[
                        s].skill_value

            return MSRCPSPCalendar(skills_names=list(rcpsp_do_domain.skills_set),
                                   resource_unit_names=resource_unit_names,
                                   resource_type_names=resource_type_names,
                                   resource_skills=resource_skills,
                                   task_ids=sorted(rcpsp_do_domain.mode_details.keys()),
                                   tasks_mode=rcpsp_do_domain.mode_details,
                                   successors=rcpsp_do_domain.successors,
                                   max_horizon=rcpsp_do_domain.horizon,
                                   resource_availability=resource_availability,
                                   resource_renewable=resource_renewable)


