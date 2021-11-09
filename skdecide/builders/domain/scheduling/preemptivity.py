# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List

__all__ = ["WithPreemptivity", "WithoutPreemptivity"]

from enum import Enum


class ResumeType(Enum):
    NA = 0
    Restart = 1
    Resume = 2


class WithPreemptivity:
    """A domain must inherit this class if there exist at least 1 task that can be paused."""

    def _get_task_preemptivity(self) -> Dict[int, bool]:
        """Return a dictionary where the key is a task id and the value a boolean indicating
        if the task can be paused or stopped.
        E.g. {
                1: False
                2: True
                3: False
                4: False
                5: True
                6: False
                }
        """
        raise NotImplementedError

    def get_task_preemptivity(self) -> Dict[int, bool]:
        """Return a dictionary where the key is a task id and the value a boolean indicating
        if the task can be paused or stopped.
        E.g. {
                1: False
                2: True
                3: False
                4: False
                5: True
                6: False
                }
        """
        return self._get_task_preemptivity()

    def _get_task_resuming_type(self) -> Dict[int, ResumeType]:
        """Return a dictionary where the key is a task id and the value is of type ResumeType indicating
        if the task can be resumed (restarted from where it was paused with no time loss)
        or restarted (restarted from the start).
        E.g. {
                1: ResumeType.NA
                2: ResumeType.Resume
                3: ResumeType.NA
                4: ResumeType.NA
                5: ResumeType.Restart
                6: ResumeType.NA
                }
        """
        raise NotImplementedError

    def get_task_resuming_type(self) -> Dict[int, ResumeType]:
        """Return a dictionary where the key is a task id and the value is of type ResumeType indicating
        if the task can be resumed (restarted from where it was paused with no time loss)
        or restarted (restarted from the start).
        E.g. {
                1: ResumeType.NA
                2: ResumeType.Resume
                3: ResumeType.NA
                4: ResumeType.NA
                5: ResumeType.Restart
                6: ResumeType.NA
                }
        """
        return self._get_task_resuming_type()

    def _get_task_paused_non_renewable_resource_returned(self) -> Dict[int, bool]:
        """Return a dictionary where the key is a task id and the value is of type bool indicating
        if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
        E.g. {
                2: False  # if paused, non-renewable resource will be consumed
                5: True  # if paused, the non-renewable resource will be available again
                }
        """
        raise NotImplementedError

    def get_task_paused_non_renewable_resource_returned(self) -> Dict[int, bool]:
        """Return a dictionary where the key is a task id and the value is of type bool indicating
        if the non-renewable resources are consumed when the task is paused (False) or made available again (True).
        E.g. {
                2: False  # if paused, non-renewable resource will be consumed
                5: True  # if paused, the non-renewable resource will be available again
                }
        """
        return self._get_task_paused_non_renewable_resource_returned()


class WithoutPreemptivity(WithPreemptivity):
    """A domain must inherit this class if none of the task can be paused."""

    def _get_task_preemptivity(self) -> Dict[int, bool]:
        preemptivity = {}
        ids = self.get_tasks_ids()
        for id in ids:
            preemptivity[id] = False
        return preemptivity

    def _get_task_resuming_type(self) -> Dict[int, ResumeType]:
        resume_types = {}
        ids = self.get_tasks_ids()
        for id in ids:
            resume_types[id] = ResumeType.NA
        return resume_types

    def _get_task_paused_non_renewable_resource_returned(self) -> Dict[int, bool]:
        handling = {}
        ids = self.get_tasks_ids()
        for id in ids:
            handling[id] = True
        return handling
