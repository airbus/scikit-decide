# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List

__all__ = [
    "WithModeCosts",
    "WithoutModeCosts",
    "WithResourceCosts",
    "WithoutResourceCosts",
]


class WithModeCosts:
    """A domain must inherit this class if there are some mode costs to consider."""

    def _get_mode_costs(
        self,
    ) -> Dict[
        int, Dict[int, float]
    ]:  # TODO: To be handled by domain (in transition cost)
        """
        Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
        and the value indicates the cost of execution the task in the mode."""
        raise NotImplementedError

    def get_mode_costs(
        self,
    ) -> Dict[
        int, Dict[int, float]
    ]:  # TODO: To be handled by domain (in transition cost)
        """
        Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode
        and the value indicates the cost of execution the task in the mode."""
        return self._get_mode_costs()


class WithoutModeCosts(WithModeCosts):
    """A domain must inherit this class if there are no mode cost to consider."""

    def _get_mode_costs(self) -> Dict[int, Dict[int, float]]:
        cost_dict = {}
        for task_id, modes in self.get_tasks_modes().items():
            cost_dict[task_id] = {mode_id: 0.0 for mode_id in modes}
        return cost_dict


class WithResourceCosts:
    """A domain must inherit this class if there are some resource costs to consider."""

    def _get_resource_cost_per_time_unit(
        self,
    ) -> Dict[str, float]:  # TODO: To be handled by domain (in transition cost)
        """
        Return a dictionary where the key is the name of a resource (str)
        and the value indicates the cost of using this resource per time unit."""
        raise NotImplementedError

    def get_resource_cost_per_time_unit(
        self,
    ) -> Dict[str, float]:  # TODO: To be handled by domain (in transition cost)
        """
        Return a dictionary where the key is the name of a resource (str)
        and the value indicates the cost of using this resource per time unit."""
        return self._get_resource_cost_per_time_unit()


class WithoutResourceCosts(WithResourceCosts):
    """A domain must inherit this class if there are no resource cost to consider."""

    def _get_resource_cost_per_time_unit(self) -> Dict[str, float]:
        cost_dict = {}
        for res in self.get_resource_types_names():
            cost_dict[res] = 0.0
        for res in self.get_resource_units_names():
            cost_dict[res] = 0.0
        return cost_dict
