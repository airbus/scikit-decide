import functools
from typing import Generic, List

from airlaps.core import T_state, T_event, Constraint

__all__ = ['ConstrainedDomain']


class ConstrainedDomain(Generic[T_state, T_event]):
    """A domain must inherit this class if it has constraints."""

    @functools.lru_cache()
    def get_constraints(self) -> List[Constraint[T_state, T_event]]:
        """Get the (cached) domain constraints.

        By default, #ConstrainedDomain.get_constraints() internally calls #ConstrainedDomain._get_constraints_() the
        first time and automatically caches its value to make future calls more efficient (since the list of constraints
        is assumed to be constant).

        # Returns
        The list of constraints.
        """
        return self._get_constraints_()

    def _get_constraints_(self) -> List[Constraint[T_state, T_event]]:
        """Get the domain constraints.

        This is a helper function called by default from #ConstrainedDomain.get_constraints(), the difference being that
        the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The list of constraints.
        """
        raise NotImplementedError
