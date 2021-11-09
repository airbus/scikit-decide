# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.core import D, Value, autocastable

__all__ = ["Rewards", "PositiveCosts"]


class Rewards:
    """A domain must inherit this class if it sends rewards (positive and/or negative)."""

    @autocastable
    def check_value(self, value: Value[D.T_value]) -> bool:
        """Check that a value is compliant with its reward specification.

        !!! tip
            This function returns always True by default because any kind of reward should be accepted at this level.

        # Parameters
        value: The value to check.

        # Returns
        True if the value is compliant (False otherwise).
        """
        return self._check_value(value)

    def _check_value(self, value: Value[D.T_value]) -> bool:
        """Check that a value is compliant with its reward specification.

        !!! tip
            This function returns always True by default because any kind of reward should be accepted at this level.

        # Parameters
        value: The value to check.

        # Returns
        True if the value is compliant (False otherwise).
        """
        return True


class PositiveCosts(Rewards):
    """A domain must inherit this class if it sends only positive costs (i.e. negative rewards).

    Having only positive costs is a required assumption for certain solvers to work, such as classical planners.
    """

    def _check_value(self, value: Value[D.T_value]) -> bool:
        """Check that a value is compliant with its cost specification (must be positive).

        !!! tip
            This function calls #PositiveCost._is_positive() to determine if a value is positive (can be overridden for
            advanced value types).

        # Parameters
        value: The value to check.

        # Returns
        True if the value is compliant (False otherwise).
        """
        return self._is_positive(value.cost)

    def _is_positive(self, cost: D.T_value) -> bool:
        """Determine if a value is positive (can be overridden for advanced value types).

        # Parameters
        cost: The cost to evaluate.

        # Returns
        True if the cost is positive (False otherwise).
        """
        return cost >= 0
