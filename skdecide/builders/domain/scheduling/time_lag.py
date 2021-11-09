# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

__all__ = [
    "TimeLag",
    "MinimumOnlyTimeLag",
    "MaximumOnlyTimeLag",
    "WithTimeLag",
    "WithoutTimeLag",
]


class TimeLag:
    """Defines a time lag with both a minimum time lag and maximum time lag."""

    def __init__(self, minimum_time_lag, maximum_time_lags):
        self.minimum_time_lag = minimum_time_lag
        self.maximum_time_lags = maximum_time_lags


class MinimumOnlyTimeLag(TimeLag):
    """Defines a minimum time lag."""

    def __init__(self, minimum_time_lag):
        self.minimum_time_lag = minimum_time_lag
        self.maximum_time_lags = self.get_max_horizon()


class MaximumOnlyTimeLag(TimeLag):
    """Defines a maximum time lag."""

    def __init__(self, maximum_time_lags):
        self.minimum_time_lag = 0
        self.maximum_time_lags = maximum_time_lags


class WithTimeLag:
    """A domain must inherit this class if there are minimum and maximum time lags between some of its tasks."""

    def get_time_lags(self) -> Dict[int, Dict[int, TimeLag]]:
        """
        Return nested dictionaries where the first key is the id of a task (int)
        and the second key is the id of another task (int).
        The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
        of the first task to the start of the second task.

        e.g.
            {
                12:{
                    15: TimeLag(5, 10),
                    16: TimeLag(5, 20),
                    17: MinimumOnlyTimeLag(5),
                    18: MaximumOnlyTimeLag(15),
                }
            }

        # Returns
        A dictionary of TimeLag objects.

        """
        return self._get_time_lags()

    def _get_time_lags(self) -> Dict[int, Dict[int, TimeLag]]:
        """
        Return nested dictionaries where the first key is the id of a task (int)
        and the second key is the id of another task (int).
        The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
        of the first task to the start of the second task.

        e.g.
            {
                12:{
                    15: TimeLag(5, 10),
                    16: TimeLag(5, 20),
                    17: MinimumOnlyTimeLag(5),
                    18: MaximumOnlyTimeLag(15),
                }
            }

        # Returns
        A dictionary of TimeLag objects.
        """
        raise NotImplementedError


class WithoutTimeLag(WithTimeLag):
    """A domain must inherit this class if there is no required time lag between its tasks."""

    def _get_time_lags(self) -> Dict[int, Dict[int, TimeLag]]:
        """
        Return nested dictionaries where the first key is the id of a task (int)
        and the second key is the id of another task (int).
        The value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end
        of the first task to the start of the second task."""
        return {}
