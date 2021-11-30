# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

# __all__ = ['Task', 'Status']
#
#
# class Status(Enum):
#     unreleased = 0
#     released = 1
#     ongoing = 2
#     complete = 3


class Task:

    id: int
    start: int
    end: int
    sampled_duration: int
    mode: int
    paused: List[int]
    resumed: List[int]

    def __init__(self, id: int, start: int, sampled_duration: int):
        self.id = id
        self.start = start
        self.end = None
        self.sampled_duration = sampled_duration
        self.mode = None
        self.paused = []
        self.resumed = []

    def get_task_active_time(self, t: Optional[int] = None):
        tt = t
        if self.end is not None:
            tt = min(t, self.end)
        if self.start is None:
            return 0
        time_since_start = tt - self.start
        time_paused = 0
        for i in range(len(self.paused)):
            time_paused += self.resumed[i] - self.paused[i]
        total_active_time = time_since_start - time_paused
        return total_active_time

    def __str__(self):
        out = ""
        for key in sorted(self.__dict__.keys()):
            out += str(key) + ":" + str(getattr(self, key)) + ","
        return out

    # def __copy__(self):
    #     s = Task(id=self.id, status=self.status)
    #     s.start = self.start
    #     s.end = self.end
    #     s.progress = self.progress
    #     s.mode = self.mode
    #     return s
