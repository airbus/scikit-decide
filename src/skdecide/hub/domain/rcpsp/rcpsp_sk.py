# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Optional

from skdecide import DiscreteDistribution, Distribution
from skdecide.builders.domain.scheduling.modes import (
    ConstantModeConsumption,
    ModeConsumption,
)
from skdecide.builders.domain.scheduling.scheduling_domains import (
    MultiModeMultiSkillRCPSP,
    MultiModeMultiSkillRCPSPCalendar,
    MultiModeRCPSP,
    MultiModeRCPSP_Stochastic_Durations,
    MultiModeRCPSPCalendar,
    MultiModeRCPSPCalendar_Stochastic_Durations,
    SchedulingObjectiveEnum,
    SingleMode,
)


class D(MultiModeRCPSP):
    pass


class MRCPSP(D):
    """Multimode RCPSP domain

    # Attributes
     resource_names: list of resource names
    task_ids: list of tasks ids
    tasks_mode: dictionary giving details of resource consumption and duration for each tasks/modes :
      format is the following : {task_id: {mode_1: {"duration": 2, "res_1": 1}, mode_2: {"duration": 3, "res_2": 2}}}
    successors: dictionary of precedence constraint:
      format is the following {task_id: list[task_id]}, where the values are the list of successor task of a given task_id
    max_horizon: the max horizon for scheduling
    resource_availability: for each resource, gives its (constant) capacity
    resource_renewable: for each resource, indicates if it's renewable or not
    """

    def __init__(
        self,
        tasks_mode: dict[int, dict[int, dict[str, int]]],
        max_horizon: int,
        successors: Optional[dict[int, list[int]]] = None,
        resource_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, int]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
    ):
        if successors is None:
            self.successors: dict[int, list[int]] = {}
        else:
            self.successors = successors
        if resource_names is None:
            self.resource_names: list[str] = []
        else:
            self.resource_names = resource_names
        if resource_availability is None:
            self.resource_availability: dict[str, int] = {}
        else:
            self.resource_availability = resource_availability
        if resource_renewable is None:
            self.resource_renewable: dict[str, bool] = {}
        else:
            self.resource_renewable = resource_renewable
        self.task_ids = list(tasks_mode)
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.duration_dict = {}
        for task in tasks_mode:
            self.task_mode_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                for r in tasks_mode[task][mode]:
                    if r in self.resource_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [
                            tasks_mode[task][mode][r]
                        ]
                self.duration_dict[task][mode] = tasks_mode[task][mode]["duration"]
        self.max_horizon = max_horizon
        self.initialize_domain()

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_resource_renewability(self) -> dict[str, bool]:
        return self.resource_renewable

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return self.task_ids

    def _get_task_duration(
        self, task: int, mode: int = 1, progress_from: float = 0.0
    ) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_types_names(self) -> list[str]:
        return self.resource_names

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class D(MultiModeRCPSPCalendar):
    pass


class MRCPSPCalendar(D):
    """Multimode RCPSP with calendars domain

    # Attributes
    resource_names: list of resource names
    task_ids: list of tasks ids
    tasks_mode: dictionary giving details of resource consumption and duration for each tasks/modes :
      format is the following : {task_id: {mode_1: {"duration": 2, "res_1": 1}, mode_2: {"duration": 3, "res_2": 2}}}
    successors: dictionary of precedence constraint:
      format is the following {task_id: list[task_id]}, where the values are the list of successor task of a given task_id
    max_horizon: the max horizon for scheduling
    resource_availability: for each resource, gives its capacity through time as a list of integer
    resource_renewable: for each resource, indicates if it's renewable or not
    """

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        return self.resource_availability[resource][time]

    def __init__(
        self,
        tasks_mode: dict[int, dict[int, dict[str, int]]],
        max_horizon: int,
        successors: Optional[dict[int, list[int]]] = None,
        resource_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, list[int]]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
    ):
        if successors is None:
            self.successors: dict[int, list[int]] = {}
        else:
            self.successors = successors
        if resource_names is None:
            self.resource_names: list[str] = []
        else:
            self.resource_names = resource_names
        if resource_availability is None:
            self.resource_availability: dict[str, list[int]] = {}
        else:
            self.resource_availability = resource_availability
        if resource_renewable is None:
            self.resource_renewable: dict[str, bool] = {}
        else:
            self.resource_renewable = resource_renewable
        self.task_ids = list(tasks_mode)
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.duration_dict = {}
        for task in tasks_mode:
            self.task_mode_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                for r in tasks_mode[task][mode]:
                    if r in self.resource_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [
                            tasks_mode[task][mode][r]
                        ]
                self.duration_dict[task][mode] = tasks_mode[task][mode]["duration"]
        self.max_horizon = max_horizon
        self.original_resource_availability = {
            r: max(self.resource_availability[r]) for r in self.resource_availability
        }
        self.initialize_domain()

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_resource_renewability(self) -> dict[str, bool]:
        return self.resource_renewable

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return self.task_ids

    def _get_task_duration(
        self, task: int, mode: int = 1, progress_from: float = 0.0
    ) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.original_resource_availability[resource]

    def _get_resource_types_names(self) -> list[str]:
        return self.resource_names

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class RCPSP(MRCPSP, SingleMode):
    """Monomode RCPSP domain

    # Attributes
    resource_names: list of resource names
    task_ids: list of tasks ids
    tasks_mode: dictionary giving details of resource consumption and duration for each tasks :
      format is the following : {task_id: {1: {"duration": 2, "res_1": 1}}, only 1 mode in this template
    successors: dictionary of precedence constraint:
      format is the following {task_id: list[task_id]}, where the values are the list of successor task of a given task_id
    max_horizon: the max horizon for scheduling
    resource_availability: for each resource, gives its constant capacity
    resource_renewable: for each resource, indicates if it's renewable or not
    """

    def __init__(
        self,
        tasks_mode: dict[int, dict[int, dict[str, int]]],
        max_horizon: int,
        successors: Optional[dict[int, list[int]]] = None,
        resource_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, int]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
    ):
        super().__init__(
            resource_names=resource_names,
            tasks_mode=tasks_mode,
            successors=successors,
            max_horizon=max_horizon,
            resource_availability=resource_availability,
            resource_renewable=resource_renewable,
        )
        self.tasks_modes_rcpsp = {
            t: self.task_mode_dict[t][1] for t in self.task_mode_dict
        }

    def _get_tasks_mode(self) -> dict[int, ModeConsumption]:
        return self.tasks_modes_rcpsp


class RCPSPCalendar(MRCPSPCalendar, SingleMode):
    """Monomode RCPSP with calendars domain

    # Attributes
    resource_names: list of resource names
    task_ids: list of tasks ids
    tasks_mode: dictionary giving details of resource consumption and duration for each tasks :
      format is the following : {task_id: {1: {"duration": 2, "res_1": 1}}, only 1 mode in this template
    successors: dictionary of precedence constraint:
      format is the following {task_id: list[task_id]}, where the values are the list of successor task of a given task_id
    max_horizon: the max horizon for scheduling
    resource_availability: for each resource, gives its capacity through time
    resource_renewable: for each resource, indicates if it's renewable or not
    """

    def __init__(
        self,
        tasks_mode: dict[int, dict[int, dict[str, int]]],
        max_horizon: int,
        successors: Optional[dict[int, list[int]]] = None,
        resource_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, list[int]]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
    ):
        super().__init__(
            resource_names=resource_names,
            tasks_mode=tasks_mode,
            successors=successors,
            max_horizon=max_horizon,
            resource_availability=resource_availability,
            resource_renewable=resource_renewable,
        )
        self.tasks_modes_rcpsp = {
            t: self.task_mode_dict[t][1] for t in self.task_mode_dict
        }

    def _get_tasks_mode(self) -> dict[int, ModeConsumption]:
        return self.tasks_modes_rcpsp


class Stochastic_RCPSP(MultiModeRCPSP_Stochastic_Durations):
    """Stochastic RCPSP

    # Attributes
    resource_names: list of resource names
    task_ids: list of tasks ids
    tasks_mode: dictionary giving details of resource consumption and duration for each tasks :
      format is the following : {task_id: {1: {"res_1": 1}, 2: {"res_1": 2}}
    duration_distribution: dictionary giving distribution of task duration function of mode.
    successors: dictionary of precedence constraint:
      format is the following {task_id: list[task_id]}, where the values are the list of successor task of a given task_id
    max_horizon: the max horizon for scheduling
    resource_availability: for each resource, gives its constant capacity
    resource_renewable: for each resource, indicates if it's renewable or not
    """

    def __init__(
        self,
        duration_distribution: dict[int, dict[int, DiscreteDistribution]],
        max_horizon: int,
        tasks_mode: Optional[dict[int, dict[int, ModeConsumption]]] = None,
        successors: Optional[dict[int, list[int]]] = None,
        resource_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, int]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
    ):
        if successors is None:
            self.successors: dict[int, list[int]] = {}
        else:
            self.successors = successors
        if resource_names is None:
            self.resource_names: list[str] = []
        else:
            self.resource_names = resource_names
        if resource_availability is None:
            self.resource_availability: dict[str, int] = {}
        else:
            self.resource_availability = resource_availability
        if resource_renewable is None:
            self.resource_renewable: dict[str, bool] = {}
        else:
            self.resource_renewable = resource_renewable
        if tasks_mode is None:
            self.task_mode_dict: dict[int, dict[int, ModeConsumption]] = {}
        else:
            self.task_mode_dict = tasks_mode
        self.task_ids = list(duration_distribution)
        self.duration_distribution = duration_distribution
        self.max_horizon = max_horizon

        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return self.task_ids

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: int = 1,
        progress_from: float = 0.0,
        multivariate_settings: Optional[dict[str, int]] = None,
    ) -> Distribution:
        return self.duration_distribution[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_types_names(self) -> list[str]:
        return self.resource_names


def build_stochastic_from_deterministic(rcpsp: MRCPSP, task_to_noise: set[int] = None):
    if task_to_noise is None:
        task_to_noise = set(rcpsp.get_tasks_ids())
    duration_distribution = {}
    for task_id in rcpsp.get_tasks_ids():
        duration_distribution[task_id] = {}
        for mode in rcpsp.get_task_modes(task_id=task_id):
            duration = rcpsp.get_task_duration(task=task_id, mode=mode)
            if duration == 0 or task_id not in task_to_noise:
                distrib = DiscreteDistribution(values=[(duration, 1)])
            else:
                n = 10
                distrib = DiscreteDistribution(
                    values=[
                        (max(1, duration + i), 1 / (2 * n + 1))
                        for i in range(-n, n + 1)
                    ]
                )
            duration_distribution[task_id][mode] = distrib

    return Stochastic_RCPSP(
        resource_names=rcpsp.get_resource_types_names(),
        tasks_mode=rcpsp.get_tasks_modes(),  # ressource
        duration_distribution=duration_distribution,
        successors=rcpsp.successors,
        max_horizon=rcpsp.max_horizon * 2,
        resource_availability=rcpsp.resource_availability,
        resource_renewable=rcpsp.resource_renewable,
    )


def build_n_determinist_from_stochastic(srcpsp: Stochastic_RCPSP, nb_instance: int):
    instances = []
    for i in range(nb_instance):
        modes = srcpsp.get_tasks_modes()
        modes_for_rcpsp = {
            task: {
                mode: {
                    r: modes[task][mode].get_resource_need_at_time(r, 0)
                    for r in modes[task][mode].get_ressource_names()
                }
                for mode in modes[task]
            }
            for task in modes
        }
        for t in modes_for_rcpsp:
            for m in modes_for_rcpsp[t]:
                duration = srcpsp.sample_task_duration(task=t, mode=m)
                modes_for_rcpsp[t][m]["duration"] = duration

        resource_availability_dict = {}
        for r in srcpsp.get_resource_types_names():
            resource_availability_dict[r] = srcpsp.get_original_quantity_resource(r)

        instances += [
            MRCPSP(
                resource_names=srcpsp.get_resource_types_names(),
                tasks_mode=modes_for_rcpsp,  # ressource
                successors=srcpsp.successors,
                # max_horizon=srcpsp.max_horizon,
                max_horizon=srcpsp.get_max_horizon(),
                # resource_availability=srcpsp.resource_availability,
                resource_availability=resource_availability_dict,
                resource_renewable=srcpsp.get_resource_renewability(),
                # resource_renewable=srcpsp.resource_renewable
            )
        ]
    return instances


class D(MultiModeRCPSPCalendar_Stochastic_Durations):
    pass


class SMRCPSPCalendar(D):
    """Stochastic RCPSP With calendars domain

    # Attributes
    resource_names: list of resource names
    task_ids: list of tasks ids
    tasks_mode: dictionary giving details of resource consumption and duration for each tasks :
      format is the following : {task_id: {1: {"res_1": 1}, 2: {"res_1": 2}}
    duration_distribution: dictionary giving distribution of task duration function of mode.
    successors: dictionary of precedence constraint:
      format is the following {task_id: list[task_id]}, where the values are the list of successor task of a given task_id
    max_horizon: the max horizon for scheduling
    resource_availability: for each resource, gives its variable capacity
    resource_renewable: for each resource, indicates if it's renewable or not
    """

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: int = 1,
        progress_from: float = 0.0,
        multivariate_settings: Optional[dict[str, int]] = None,
    ) -> Distribution:
        return self.duration_distribution[task][mode]

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        return self.resource_availability[resource][time]

    def __init__(
        self,
        duration_distribution: dict[int, dict[int, DiscreteDistribution]],
        max_horizon: int,
        tasks_mode: Optional[dict[int, dict[int, ModeConsumption]]] = None,
        successors: Optional[dict[int, list[int]]] = None,
        resource_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, list[int]]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
    ):
        if successors is None:
            self.successors: dict[int, list[int]] = {}
        else:
            self.successors = successors
        if resource_names is None:
            self.resource_names: list[str] = []
        else:
            self.resource_names = resource_names
        if resource_availability is None:
            self.resource_availability: dict[str, list[int]] = {}
        else:
            self.resource_availability = resource_availability
        if resource_renewable is None:
            self.resource_renewable: dict[str, bool] = {}
        else:
            self.resource_renewable = resource_renewable
        if tasks_mode is None:
            self.task_mode_dict: dict[int, dict[int, ModeConsumption]] = {}
        else:
            self.task_mode_dict = tasks_mode
        self.task_ids = list(duration_distribution)
        self.duration_distribution = duration_distribution
        self.max_horizon = max_horizon
        self.original_resource_availability = {
            r: max(self.resource_availability[r]) for r in self.resource_availability
        }
        self.initialize_domain()

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_resource_renewability(self) -> dict[str, bool]:
        return self.resource_renewable

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return self.task_ids

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        # return self.resource_availability[resource]
        return self.original_resource_availability[resource]

    def _get_resource_types_names(self) -> list[str]:
        return self.resource_names

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class D(MultiModeMultiSkillRCPSP):
    pass


class MSRCPSP(D):
    """Multi-skill RCPSP domain

    # Attributes
    skills_names: list of skills id
    resource_unit_names: list of unitary skilled resource
    resource_type_names: list of cumulative resource
    resource_skills: for each resource_unit and skills store the skill level
    others : see classical RCPSP doc
    """

    def __init__(
        self,
        tasks_mode: dict[int, dict[int, dict[str, int]]],
        max_horizon: int,
        successors: Optional[dict[int, list[int]]] = None,
        resource_type_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, int]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
        skills_names: Optional[list[str]] = None,
        resource_unit_names: Optional[list[str]] = None,
        resource_skills: Optional[dict[str, dict[str, Any]]] = None,
    ):
        if skills_names is None:
            self.skills_set: set[str] = set()
        else:
            self.skills_set = set(skills_names)
        if successors is None:
            self.successors: dict[int, list[int]] = {}
        else:
            self.successors = successors
        if resource_type_names is None:
            self.resource_type_names: list[str] = []
        else:
            self.resource_type_names = resource_type_names
        if resource_unit_names is None:
            self.resource_unit_names: list[str] = []
        else:
            self.resource_unit_names = resource_unit_names
        if resource_skills is None:
            self.resource_skills: dict[str, dict[str, Any]] = {}
        else:
            self.resource_skills = resource_skills
        if resource_availability is None:
            self.resource_availability: dict[str, int] = {}
        else:
            self.resource_availability = resource_availability
        if resource_renewable is None:
            self.resource_renewable: dict[str, bool] = {}
        else:
            self.resource_renewable = resource_renewable
        self.task_ids = list(tasks_mode)
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.task_skills_dict = {}
        self.duration_dict = {}
        for task in tasks_mode:
            self.task_mode_dict[task] = {}
            self.task_skills_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                self.task_skills_dict[task][mode] = {}
                for r in tasks_mode[task][mode]:
                    if r in self.resource_type_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [
                            tasks_mode[task][mode][r]
                        ]
                    if r in self.skills_set:
                        self.task_skills_dict[task][mode][r] = tasks_mode[task][mode][r]
                self.duration_dict[task][mode] = tasks_mode[task][mode]["duration"]
        self.max_horizon = max_horizon
        self.initialize_domain()

    def _get_resource_units_names(self) -> list[str]:
        """Return the names (string) of all resource units as a list."""
        return self.resource_unit_names

    def _get_resource_types_names(self) -> list[str]:
        return self.resource_type_names

    def _get_resource_type_for_unit(self) -> dict[str, str]:
        """Return a dictionary where the key is a resource unit name and the value a resource type name.
        An empty dictionary can be used if there are no resource unit matching a resource type."""
        return {}

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return self.task_ids

    def _get_task_duration(
        self, task: int, mode: int = 1, progress_from: float = 0.0
    ) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_renewability(self) -> dict[str, bool]:
        return self.resource_renewable

    def _get_all_resources_skills(self) -> dict[str, dict[str, Any]]:
        return self.resource_skills

    def _get_all_tasks_skills(self) -> dict[int, dict[int, dict[str, Any]]]:
        return self.task_skills_dict

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class D(MultiModeMultiSkillRCPSPCalendar):
    pass


class MSRCPSPCalendar(D):
    """
    Multi-skill RCPSP with calendars domain
    Defined the same as classical MSRCPSP but with variable resource availability.
    """

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        return self.resource_availability[resource][time]

    def __init__(
        self,
        tasks_mode: dict[int, dict[int, dict[str, int]]],
        max_horizon: int,
        successors: Optional[dict[int, list[int]]] = None,
        resource_type_names: Optional[list[str]] = None,
        resource_availability: Optional[dict[str, list[int]]] = None,
        resource_renewable: Optional[dict[str, bool]] = None,
        skills_names: Optional[list[str]] = None,
        resource_unit_names: Optional[list[str]] = None,
        resource_skills: Optional[dict[str, dict[str, Any]]] = None,
    ):
        if skills_names is None:
            self.skills_set: set[str] = set()
        else:
            self.skills_set = set(skills_names)
        if successors is None:
            self.successors: dict[int, list[int]] = {}
        else:
            self.successors = successors
        if resource_type_names is None:
            self.resource_type_names: list[str] = []
        else:
            self.resource_type_names = resource_type_names
        if resource_unit_names is None:
            self.resource_unit_names: list[str] = []
        else:
            self.resource_unit_names = resource_unit_names
        if resource_skills is None:
            self.resource_skills: dict[str, dict[str, Any]] = {}
        else:
            self.resource_skills = resource_skills
        if resource_availability is None:
            self.resource_availability: dict[str, list[int]] = {}
        else:
            self.resource_availability = resource_availability
        if resource_renewable is None:
            self.resource_renewable: dict[str, bool] = {}
        else:
            self.resource_renewable = resource_renewable
        self.task_ids = list(tasks_mode)
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.task_skills_dict = {}
        self.duration_dict = {}
        for task in tasks_mode:
            self.task_mode_dict[task] = {}
            self.task_skills_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                self.task_skills_dict[task][mode] = {}
                for r in tasks_mode[task][mode]:
                    if r in self.resource_type_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [
                            tasks_mode[task][mode][r]
                        ]
                    if r in self.skills_set:
                        self.task_skills_dict[task][mode][r] = tasks_mode[task][mode][r]
                self.duration_dict[task][mode] = tasks_mode[task][mode]["duration"]
        self.max_horizon = max_horizon
        self.original_resource_availability = {
            r: max(self.resource_availability[r]) for r in self.resource_availability
        }
        self.initialize_domain()

    def _get_resource_units_names(self) -> list[str]:
        """Return the names (string) of all resource units as a list."""
        return self.resource_unit_names

    def _get_resource_types_names(self) -> list[str]:
        return self.resource_type_names

    def _get_resource_type_for_unit(self) -> dict[str, str]:
        """Return a dictionary where the key is a resource unit name and the value a resource type name.
        An empty dictionary can be used if there are no resource unit matching a resource type."""
        return {}

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return self.task_ids

    def _get_task_duration(
        self, task: int, mode: int = 1, progress_from: float = 0.0
    ) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.original_resource_availability[resource]

    def _get_resource_renewability(self) -> dict[str, bool]:
        return self.resource_renewable

    def _get_all_resources_skills(self) -> dict[str, dict[str, Any]]:
        return self.resource_skills

    def _get_all_tasks_skills(self) -> dict[int, dict[int, dict[str, Any]]]:
        return self.task_skills_dict

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]
