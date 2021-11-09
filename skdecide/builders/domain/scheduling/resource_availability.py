# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

__all__ = [
    "UncertainResourceAvailabilityChanges",
    "DeterministicResourceAvailabilityChanges",
    "WithoutResourceAvailabilityChange",
]


class UncertainResourceAvailabilityChanges:
    """A domain must inherit this class if the availability of its resource vary in an uncertain way over time."""

    def _sample_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Sample an amount of resource availability (int) for the given resource
        (either resource type or resource unit) at the given time. This number should be the sum of the number of
        resource available at time t and the number of resource of this type consumed so far)."""
        raise NotImplementedError

    def sample_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Sample an amount of resource availability (int) for the given resource
        (either resource type or resource unit) at the given time. This number should be the sum of the number of
        resource available at time t and the number of resource of this type consumed so far)."""
        return self._sample_quantity_resource(resource=resource, time=time, **kwargs)

    def check_unique_resource_names(
        self,
    ) -> bool:  # TODO: How to enforce a call to this function when initialising a domain ?
        """Return True if there are no duplicates in resource names across both resource types
        and resource units name lists."""
        list1 = self.get_resource_types_names() + self.get_resource_units_names()
        list2 = list(set(list1))
        check_1 = len(list1) == len(list2)  # no duplicated names
        check_2 = len(list2) > 0  # at least one resource
        return check_1 and check_2


class DeterministicResourceAvailabilityChanges(UncertainResourceAvailabilityChanges):
    """A domain must inherit this class if the availability of its resource vary in a deterministic way over time."""

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        raise NotImplementedError

    def get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        return self._get_quantity_resource(resource=resource, time=time, **kwargs)

    def _sample_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Sample an amount of resource availability (int) for the given resource
        (either resource type or resource unit) at the given time. This number should be the sum of the number of
        resource available at time t and the number of resource of this type consumed so far)."""
        return self.get_quantity_resource(resource, time, **kwargs)


class WithoutResourceAvailabilityChange(DeterministicResourceAvailabilityChanges):
    """A domain must inherit this class if the availability of its resource does not vary over time."""

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        """Return the resource availability (int) for the given resource (either resource type or resource unit)."""
        raise NotImplementedError

    def get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        """Return the resource availability (int) for the given resource (either resource type or resource unit)."""
        return self._get_original_quantity_resource(resource=resource, **kwargs)

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        return self.get_original_quantity_resource(resource)

    def _sample_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Sample an amount of resource availability (int) for the given resource
        (either resource type or resource unit) at the given time. This number should be the sum of the number of
        resource available at time t and the number of resource of this type consumed so far)."""
        return self.get_original_quantity_resource(resource)
