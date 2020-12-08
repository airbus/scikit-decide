from __future__ import annotations
from typing import List, Dict, Union, Set, Any

__all__ = ['VariableResourceConsumption', 'ConstantResourceConsumption']


class VariableResourceConsumption:
    """A domain must inherit this class if the amount of resource needed by some tasks vary in time."""

    def _get_variable_resource_consumption(self) -> bool:
        """Return true if the domain has variable resource consumption,
        false if the consumption of resource does not vary in time for any of the tasks"""
        return True

    def get_variable_resource_consumption(self) -> bool:
        """Return true if the domain has variable resource consumption,
        false if the consumption of resource does not vary in time for any of the tasks"""
        return self._get_variable_resource_consumption()


class ConstantResourceConsumption(VariableResourceConsumption):
    """A domain must inherit this class if the amount of resource needed by all tasks do not vary in time."""

    def _get_variable_resource_consumption(self) -> bool:
        """Return true if the domain has variable resource consumption,
        false if the consumption of resource does not vary in time for any of the tasks"""
        return False

