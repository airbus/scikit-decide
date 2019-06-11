import functools
from typing import Generic

from airlaps.core import T_observation, Space

__all__ = ['GoalDomain']


class GoalDomain(Generic[T_observation]):
    """A domain must inherit this class if it has formalized goals."""

    @functools.lru_cache()
    def get_goals(self) -> Space[T_observation]:
        """Get the (cached) domain goals space (finite or infinite set).

        By default, #GoalDomain.get_goals() internally calls #GoalDomain._get_goals_() the first time
        and automatically caches its value to make future calls more efficient (since the goals space is assumed to be
        constant).

        !!! warning
            Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty
            about whether the goal has been reached or not. This assumption guarantees that any policy that does not
            reach the goal with certainty incurs in infinite expected cost.[^Geffner 2013]

        [^Geffner 2013]: A Concise Introduction to Models and Methods for Automated Planning (Geffner, 2013)

        # Returns
        The goals space.
        """
        return self._get_goals_()

    def _get_goals_(self) -> Space[T_observation]:
        """Get the domain goals space (finite or infinite set).

        This is a helper function called by default from #GoalDomain.get_goals(), the difference being that
        the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The goals space.
        """
        raise NotImplementedError

    def is_goal(self, observation: T_observation) -> bool:
        """Indicate whether an observation belongs to the goals.

        !!! tip
            By default, this function is implemented using the #airlaps.core.Space.contains() function on the domain
            goals space provided by #GoalDomain.get_goals(), but it can be overwritten for faster implementations.

        # Parameters
        event: The event to consider.

        # Returns
        True if the event is an action (False otherwise).
        """
        return self.get_goals().contains(observation)
