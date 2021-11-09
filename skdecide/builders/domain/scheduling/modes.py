# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Set, Union

__all__ = ["MultiMode", "SingleMode"]


class ModeConsumption:
    def _get_resource_need_at_time(self, resource_name: str, time: int):
        """Return the resource consumption for the given resource at the given time.
        Note that the time should be the time from the start of the execution of the task (starting from 0)."""
        raise NotImplementedError

    def _get_non_zero_ressource_need_names(self, time: int):
        raise NotImplementedError

    def _get_ressource_names(self):
        raise NotImplementedError

    def get_resource_need_at_time(self, resource_name: str, time: int):
        """Return the resource consumption for the given resource at the given time.
        Note that the time should be the time from the start of the execution of the task (starting from 0)."""
        return self._get_resource_need_at_time(resource_name=resource_name, time=time)

    def get_non_zero_ressource_need_names(self, time: int):
        return self._get_non_zero_ressource_need_names(time=time)

    def get_ressource_names(self):
        return self._get_ressource_names()


class VaryingModeConsumption(ModeConsumption):
    """Defines the most generic type of mode."""

    def __init__(self, mode_dict: Dict[str, List[int]]):
        self.mode_details = mode_dict

    def _get_resource_need_at_time(self, resource_name: str, time: int):
        """Return the resource consumption for the given resource at the given time.
        Note that the time should be the time from the start of the execution of the task (starting from 0)."""
        if resource_name in self.mode_details:
            return self.mode_details[resource_name][time]
        else:
            return 0

    def _get_non_zero_ressource_need_names(self, time: int = 0):
        return [
            r for r in self.mode_details if self.get_resource_need_at_time(r, time) > 0
        ]

    def _get_ressource_names(self):
        return self.mode_details.keys()


class ConstantModeConsumption(VaryingModeConsumption):
    """Defines a mode where the resource consumption is constant throughout
    the duration of the task."""

    def __init__(self, mode_dict: Dict[str, int]):
        self.mode_details = {}
        for key in mode_dict.keys():
            # TODO i challenge this to be usefull?
            self.mode_details[key] = [mode_dict[key]]

    def get_resource_need(self, resource_name: str):
        """Return the resource consumption for the given resource."""
        return self._get_resource_need(resource_name=resource_name)

    def _get_resource_need(self, resource_name: str):
        """Return the resource consumption for the given resource."""
        return self.mode_details.get(resource_name, [0])[0]

    def _get_resource_need_at_time(self, resource_name: str, time: int):
        """Return the resource consumption for the given resource at the given time.
        Note that the time should be the time from the start of the execution of the task (starting from 0)."""
        return self._get_resource_need(resource_name=resource_name)


class MultiMode:
    """A domain must inherit this class if tasks can be done in 1 or more modes."""

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        """Return a set or dict of int = id of tasks"""
        raise NotImplementedError

    def get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self._get_tasks_ids()

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        """Return a nested dictionary where the first key is a task id and the second key is a mode id.
         The value is a Mode object defining the resource consumption.
        If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
        If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
        then ConstantModeConsumption should be used.

        E.g. with constant resource consumption
            {
                12: {
                        1: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1}),
                        2: ConstantModeConsumption({'rt_1': 0, 'rt_2': 3, 'ru_1': 1}),
                    }
            }

        E.g. with time varying resource consumption
            {
            12: {
                1: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]}),
                2: VaryingModeConsumption({'rt_1': [1,1,1,1,2,2,2], 'rt_2': [0,0,0,0,0,0,0], 'ru_1': [1,1,1,1,1,1,1]}),
                }
            }
        """
        raise NotImplementedError

    def get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self._get_tasks_modes()

    def _get_ressource_names_for_task_mode(self, task: int, mode: int):
        return self.get_tasks_modes()[task][mode].get_ressource_names()

    def get_ressource_names_for_task_mode(self, task: int, mode: int):
        return self._get_ressource_names_for_task_mode(task=task, mode=mode)

    def _get_task_modes(self, task_id: int):
        return self.get_tasks_modes()[task_id]

    def get_task_modes(self, task_id: int):
        return self._get_task_modes(task_id=task_id)

    def _get_task_consumption(
        self, task: int, mode: int, resource_name: str, time: int
    ):
        return self.get_task_modes(task)[mode].get_resource_need_at_time(
            resource_name=resource_name, time=time
        )

    def get_task_consumption(self, task: int, mode: int, resource_name: str, time: int):
        return self._get_task_consumption(
            task=task, mode=mode, resource_name=resource_name, time=time
        )


class SingleMode(MultiMode):
    """A domain must inherit this class if ALL tasks only have 1 possible execution mode."""

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        """Return a nested dictionary where the first key is a task id and the second key is a mode id.
        The value is a Mode object defining the resource consumption."""
        modes = {}
        tmp_dict = self.get_tasks_mode()
        for key in tmp_dict:
            modes[key] = {}
            modes[key][1] = tmp_dict[key]
        return modes

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        """Return a dictionary where the key is a task id and the value is a ModeConsumption object defining
        the resource consumption.
        If the domain is an instance of VariableResourceConsumption, VaryingModeConsumption objects should be used.
        If this is not the case (i.e. the domain is an instance of ConstantResourceConsumption),
        then ConstantModeConsumption should be used.

        E.g. with constant resource consumption
            {
                12: ConstantModeConsumption({'rt_1': 2, 'rt_2': 0, 'ru_1': 1})
            }

        E.g. with time varying resource consumption
            {
                12: VaryingModeConsumption({'rt_1': [2,2,2,2,3], 'rt_2': [0,0,0,0,0], 'ru_1': [1,1,1,1,1]})
            }
        """

        # TODO: Check if the test below is correct and should be placed here
        # if issubclass(type(self), VariableResourceConsumption) and\
        #         (not issubclass(type(self), DeterministicTaskDuration)):
        #     raise NotImplementedError('Use of varying resource consumption and non0-deterministic task durations '
        #                               'not supported yet')

        raise NotImplementedError

    def get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return self._get_tasks_mode()
