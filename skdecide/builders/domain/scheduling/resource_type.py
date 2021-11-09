# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "WithResourceTypes",
    "WithoutResourceTypes",
    "WithResourceUnits",
    "SingleResourceUnit",
    "WithoutResourceUnit",
]


class WithResourceTypes:
    """A domain must inherit this class if some of its resources are resource types."""

    def get_resource_types_names(self) -> List[str]:
        """Return the names (string) of all resource types as a list."""
        return self._get_resource_types_names()

    def _get_resource_types_names(self) -> List[str]:
        """Return the names (string) of all resource types as a list."""
        raise NotImplementedError


class WithoutResourceTypes(WithResourceTypes):
    """A domain must inherit this class if it only uses resource types."""

    def _get_resource_types_names(self) -> List[str]:
        """Return the names (string) of all resource types as a list."""
        return []


class WithResourceUnits:
    """A domain must inherit this class if some of its resources are resource units."""

    def get_resource_units_names(self) -> List[str]:
        """Return the names (string) of all resource units as a list."""
        return self._get_resource_units_names()

    def _get_resource_units_names(self) -> List[str]:
        """Return the names (string) of all resource units as a list."""
        raise NotImplementedError

    def get_resource_type_for_unit(self) -> Dict[str, str]:
        """Return a dictionary where the key is a resource unit name and the value a resource type name.
        An empty dictionary can be used if there are no resource unit matching a resource type."""
        return self._get_resource_type_for_unit()

    def _get_resource_type_for_unit(self) -> Dict[str, str]:
        """Return a dictionary where the key is a resource unit name and the value a resource type name.
        An empty dictionary can be used if there are no resource unit matching a resource type."""
        raise NotImplementedError


class SingleResourceUnit(WithResourceUnits):
    """A domain must inherit this class if there is no allocation to be done (i.e. there is a single resource)."""

    def _get_resource_units_names(self) -> List[str]:
        return ["single_resource"]

    def _get_resource_type_for_unit(self) -> Dict[str, str]:
        return {}


class WithoutResourceUnit(SingleResourceUnit):
    """A domain must inherit this class if it only uses resource types."""

    def _get_resource_units_names(self) -> List[str]:
        return []

    def _get_resource_type_for_unit(self) -> Dict[str, str]:
        return {}
