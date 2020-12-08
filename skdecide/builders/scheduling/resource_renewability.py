from __future__ import annotations
from typing import List, Union, Dict
from skdecide.builders.scheduling.scheduling_domains_modelling import State

__all__ = ['MixedRenewable', 'RenewableOnly']


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
        for task_id in state.tasks_remaining:
            task_possible = True
            for mode in self.get_task_modes(task_id).keys():
                mode_possible = True
                for res in self.get_resource_types_names():
                    if not self.get_resource_renewability()[res]:
                        need = self.get_task_modes(task_id)[mode].get_resource_need(res)
                        avail = state.resource_availability[res] - state.resource_used[res]
                        if avail - need < 0:
                            mode_possible = False
                            break
                if mode_possible is True:
                    break
            if not mode_possible:
                return False
        return True


class RenewableOnly(MixedRenewable):
    """A domain must inherit this class if the resource available are ALL renewable."""

    def _get_resource_renewability(self) -> Dict[str, bool]:
        """Return a dictionary where the key is a resource name (string)
        and the value whether this resource is renewable (True) or not (False)."""
        names = self.get_resource_types_names() + self.get_resource_units_names() # comes from resource_handling...
        renewability = {}
        for name in names:
            renewability[name] = True
        return renewability

    def is_renewable(self, resource: str):
        return True
