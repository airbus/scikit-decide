# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

__all__ = [
    "TimeWindow",
    "ClassicTimeWindow",
    "StartFromOnlyTimeWindow",
    "StartBeforeOnlyTimeWindow",
    "EndFromOnlyTimeWindow",
    "EndBeforeOnlyTimeWindow",
    "StartTimeWindow",
    "EndTimeWindow",
    "EmptyTimeWindow",
    "WithTimeWindow",
    "WithoutTimeWindow",
]


class TimeWindow:
    """Defines a time window with earliest start, latest start, earliest end and latest end only."""

    def __init__(
        self,
        earliest_start: int,
        latest_start: int,
        earliest_end: int,
        latest_end: int,
        max_horizon: int,
    ) -> None:
        self.earliest_start = earliest_start
        self.latest_start = latest_start
        self.earliest_end = earliest_end
        self.latest_end = latest_end


class ClassicTimeWindow(TimeWindow):
    """Defines a time window with earliest start and latest end only."""

    def __init__(self, earliest_start: int, latest_end: int, max_horizon: int) -> None:
        self.earliest_start = earliest_start
        self.latest_start = max_horizon
        self.earliest_end = 0
        self.latest_end = latest_end


class StartFromOnlyTimeWindow(TimeWindow):
    """Defines a time window with an earliest start only."""

    def __init__(self, earliest_start: int, max_horizon: int) -> None:
        self.earliest_start = earliest_start
        self.latest_start = max_horizon
        self.earliest_end = 0
        self.latest_end = max_horizon


class StartBeforeOnlyTimeWindow(TimeWindow):
    """Defines a time window with an latest start only."""

    def __init__(self, latest_start: int, max_horizon: int) -> None:
        self.earliest_start = 0
        self.latest_start = latest_start
        self.earliest_end = 0
        self.latest_end = max_horizon


class EndFromOnlyTimeWindow(TimeWindow):
    """Defines a time window with an earliest end only."""

    def __init__(self, earliest_end: int, max_horizon: int) -> None:
        self.earliest_start = 0
        self.latest_start = max_horizon
        self.earliest_end = earliest_end
        self.latest_end = max_horizon


class EndBeforeOnlyTimeWindow(TimeWindow):
    """Defines a time window with a latest end only."""

    def __init__(self, latest_end: int, max_horizon: int) -> None:
        self.earliest_start = 0
        self.latest_start = max_horizon
        self.earliest_end = 0
        self.latest_end = latest_end


class StartTimeWindow(TimeWindow):
    """Defines a time window with an earliest start and a latest start only."""

    def __init__(
        self, earliest_start: int, latest_start: int, max_horizon: int
    ) -> None:
        self.earliest_start = earliest_start
        self.latest_start = latest_start
        self.earliest_end = 0
        self.latest_end = max_horizon


class EndTimeWindow(TimeWindow):
    """Defines a time window with an earliest end and a latest end only."""

    def __init__(self, earliest_end: int, latest_end: int, max_horizon: int) -> None:
        self.earliest_start = 0
        self.latest_start = max_horizon
        self.earliest_end = earliest_end
        self.latest_end = latest_end


class EmptyTimeWindow(TimeWindow):
    """Defines an empty time window."""

    def __init__(self, max_horizon: int) -> None:
        self.earliest_start = 0
        self.latest_start = max_horizon
        self.earliest_end = 0
        self.latest_end = max_horizon


class WithTimeWindow:
    """A domain must inherit this class if some tasks have time windows defined."""

    def get_time_window(self) -> Dict[int, TimeWindow]:
        """
        Return a dictionary where the key is the id of a task (int)
        and the value is a TimeWindow object.
        Note that the max time horizon needs to be provided to the TimeWindow constructors
        e.g.
            {
                1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
                2: EmptyTimeWindow(self.get_max_horizon())
                3: EndTimeWindow(20, 25, self.get_max_horizon())
                4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
            }

        # Returns
        A dictionary of TimeWindow objects.
        """
        return self._get_time_window()

    def _get_time_window(self) -> Dict[int, TimeWindow]:
        """
        Return a dictionary where the key is the id of a task (int)
        and the value is a TimeWindow object.
        Note that the max time horizon needs to be provided to the TimeWindow constructors
        e.g.
            {
                1: TimeWindow(10, 15, 20, 30, self.get_max_horizon())
                2: EmptyTimeWindow(self.get_max_horizon())
                3: EndTimeWindow(20, 25, self.get_max_horizon())
                4: EndBeforeOnlyTimeWindow(40, self.get_max_horizon())
            }

        # Returns
        A dictionary of TimeWindow objects.

        """
        raise NotImplementedError


class WithoutTimeWindow(WithTimeWindow):
    """A domain must inherit this class if none of the tasks have restrictions on start times or end times."""

    def _get_time_window(self) -> Dict[int, TimeWindow]:
        """
        Return a dictionary where the key is the id of a task (int)
        and the value is a dictionary of EmptyTimeWindow object.

        # Returns
        A dictionary of TimeWindow objects.
        """
        ids = self.get_tasks_ids()
        the_dict = {}
        for id in ids:
            the_dict[id] = EmptyTimeWindow(self.get_max_horizon())
        return the_dict
