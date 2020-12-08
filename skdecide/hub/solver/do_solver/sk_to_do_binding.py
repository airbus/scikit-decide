from typing import Union

from skdecide.builders.scheduling.scheduling_domains import SchedulingDomain, \
    SingleModeRCPSP, SingleModeRCPSPCalendar, MultiModeRCPSP, MultiModeRCPSPCalendar,\
    MultiModeMultiSkillRCPSPCalendar, MultiModeMultiSkillRCPSP, MultiModeRCPSPWithCost, State
from skdecide.solvers import Solver, DeterministicPolicies
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, SingleModeRCPSPModel, \
    MultiModeRCPSPModel, RCPSPModelCalendar, RCPSPSolution
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant, Employee, \
    SkillDetail


def from_last_state_to_solution(state: State, domain: SchedulingDomain):
    modes = [state.tasks_mode[j] for j in sorted(domain.get_tasks_ids())]
    modes = modes[1:-1]
    schedule = {j: {"start_time": state.tasks_details[j].start,
                    "end_time": state.tasks_details[j].end}
                for j in state.tasks_details}
    return RCPSPSolution(problem=build_do_domain(domain),
                         rcpsp_permutation=None, rcpsp_modes=modes, rcpsp_schedule=schedule)


def build_do_domain(scheduling_domain: Union[SingleModeRCPSP,
                                             SingleModeRCPSPCalendar,
                                             MultiModeRCPSP,
                                             MultiModeRCPSPWithCost,
                                             MultiModeRCPSPCalendar,
                                             MultiModeMultiSkillRCPSP,
                                             MultiModeMultiSkillRCPSPCalendar]):
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
    if isinstance(scheduling_domain, (MultiModeMultiSkillRCPSP, MultiModeRCPSPCalendar)):
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




