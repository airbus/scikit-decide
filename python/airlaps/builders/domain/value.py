from typing import Generic

from airlaps.core import T_value, TransitionValue

__all__ = ['RewardDomain', 'PositiveCostDomain']


class RewardDomain(Generic[T_value]):
    """A domain must inherit this class if it sends rewards (positive and/or negative)."""

    def check_value(self, value: TransitionValue[T_value]) -> bool:
        """Check that a transition value is compliant with its reward specification.

        !!! note
            This function returns always True by default because any kind of reward should be accepted at this level.

        # Parameters
        value: The transition value to check.

        # Returns
        True if the transition value is compliant (False otherwise).
        """
        return True


class PositiveCostDomain(RewardDomain[T_value]):
    """A domain must inherit this class if it sends only positive costs (i.e. negative rewards).

    Having only positive costs is a required assumption for certain solvers to work, such as classical planners.
    """

    def check_value(self, value: TransitionValue[T_value]) -> bool:
        """Check that a transition value is compliant with its cost specification (must be positive).

        !!! note
            This function calls #PositiveCostDomain._is_positive() to determine if a value is positive
            (can be overwritten for advanced value types).

        # Parameters
        value: The transition value to check.

        # Returns
        True if the transition value is compliant (False otherwise).
        """
        return self._is_positive(value.cost)

    def _is_positive(self, cost: T_value) -> bool:
        """Determine if a value is positive (can be overwritten for advanced value types).

        # Parameters
        cost: The cost to evaluate.

        # Returns
        True if the cost is positive (False otherwise).
        """
        return cost >= 0
