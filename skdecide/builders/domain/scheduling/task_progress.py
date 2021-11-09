# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List, Optional, Union

__all__ = ["CustomTaskProgress", "DeterministicTaskProgress"]


class CustomTaskProgress:
    """A domain must inherit this class if the task progress is uncertain."""

    def get_task_progress(
        self,
        task: int,
        t_from: int,
        t_to: int,
        mode: Optional[int],
        sampled_duration: Optional[int] = None,
    ) -> float:
        """
        # Returns
         The task progress (float) between t_from and t_to.
        """
        return self._get_task_progress(task, t_from, t_to, mode, sampled_duration)

    def _get_task_progress(
        self,
        task: int,
        t_from: int,
        t_to: int,
        mode: Optional[int],
        sampled_duration: Optional[int] = None,
    ) -> float:
        """
        # Returns
         The task progress (float) between t_from and t_to.
        """
        raise NotImplementedError


class DeterministicTaskProgress(CustomTaskProgress):
    """A domain must inherit this class if the task progress is deterministic and can be considered as linear
    over the duration of the task."""

    def _get_task_progress(
        self,
        task: int,
        t_from: int,
        t_to: int,
        mode: Optional[int],
        sampled_duration: Optional[int] = None,
    ) -> float:
        """
        # Returns
         The task progress (float) between t_from and t_to based on the task duration
        and assuming linear progress."""
        duration = (
            self.get_latest_sampled_duration(task, mode)
            if sampled_duration is None
            else sampled_duration
        )
        if duration == 0.0:
            progress = 1
        else:
            progress = float((t_to - t_from)) / float(duration)
        return progress
