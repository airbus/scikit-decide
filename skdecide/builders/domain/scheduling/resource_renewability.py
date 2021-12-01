# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

from skdecide.builders.domain.scheduling.scheduling_domains_modelling import State

__all__ = ["MixedRenewable", "RenewableOnly"]


class MixedRenewable:
    """A domain must inherit this class if the resource available are non-renewable and renewable."""

    def get_resource_renewability(self) -> Dict[str, bool]:
        """
        Return a dictionary where the key is a resource name (string)
        and the value whether this resource is renewable (True) or not (False)."""
        return self._get_resource_renewability()

    def _get_resource_renewability(self) -> Dict[str, bool]:
        """
        Return a dictionary where the key is a resource name (string)
        and the value whether this resource is renewable (True) or not (False)."""
        raise NotImplementedError

    def is_renewable(self, resource: str):
        return self.get_resource_renewability()[resource]

    def all_tasks_possible(self, state: State) -> bool:
        """Return a True is for each task there is at least one mode in which the task can be executed, given the
        resource configuration in the state provided as argument. Returns False otherwise.
        If this function returns False, the scheduling problem is unsolvable from this state.
        This is to cope with the use of non-renable resources that may lead to state from which a
        task will not be possible anymore."""
        resource_types_names = self.get_resource_types_names()
        resource_not_renewable = set(
            res
            for res, renewable in self.get_resource_renewability().items()
            if res in resource_types_names and not renewable
        )
        modes_details = self.get_tasks_modes()
        remaining_tasks = (
            state.task_ids.difference(state.tasks_complete)
            .difference(state.tasks_progress)
            .difference(state.tasks_unsatisfiable)
        )
        for task_id in remaining_tasks:
            for mode_consumption in modes_details[task_id].values():
                for res in resource_not_renewable:
                    need = mode_consumption.get_resource_need(res)
                    avail = state.resource_availability[res] - state.resource_used[res]
                    if avail - need < 0:
                        break
                else:
                    # The else-clause runs if loop completes normally, which means
                    # that we found a mode for which all resources are available, and
                    # we can exit from the loop on modes.
                    break
            else:
                # This task is not possible
                return False
        return True


class RenewableOnly(MixedRenewable):
    """A domain must inherit this class if the resource available are ALL renewable."""

    def _get_resource_renewability(self) -> Dict[str, bool]:
        """Return a dictionary where the key is a resource name (string)
        and the value whether this resource is renewable (True) or not (False)."""
        names = (
            self.get_resource_types_names() + self.get_resource_units_names()
        )  # comes from resource_handling...
        renewability = {}
        for name in names:
            renewability[name] = True
        return renewability

    def is_renewable(self, resource: str):
        return True
