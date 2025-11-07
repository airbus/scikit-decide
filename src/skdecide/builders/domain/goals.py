# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from typing import Union

from skdecide.core import D, Space, autocastable

__all__ = ["Goals"]


class Goals:
    """A domain must inherit this class if it has formalized goals."""

    @autocastable
    def get_goals(self) -> D.T_agent[Space[D.T_observation]]:
        """Get the (cached) domain goals space (finite or infinite set).

        By default, #Goals.get_goals() internally calls #Goals._get_goals_() the first time and automatically caches its
        value to make future calls more efficient (since the goals space is assumed to be constant).

        !!! warning
            Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty
            about whether the goal has been reached or not. This assumption guarantees that any policy that does not
            reach the goal with certainty incurs in infinite expected cost. - *Geffner, 2013: A Concise Introduction to
            Models and Methods for Automated Planning*

        # Returns
        The goals space.
        """
        return self._get_goals()

    @functools.lru_cache()
    def _get_goals(self) -> D.T_agent[Space[D.T_observation]]:
        """Get the (cached) domain goals space (finite or infinite set).

        By default, #Goals._get_goals() internally calls #Goals._get_goals_() the first time and automatically caches
        its value to make future calls more efficient (since the goals space is assumed to be constant).

        !!! warning
            Goal states are assumed to be fully observable (i.e. observation = state) so that there is never uncertainty
            about whether the goal has been reached or not. This assumption guarantees that any policy that does not
            reach the goal with certainty incurs in infinite expected cost. - *Geffner, 2013: A Concise Introduction to
            Models and Methods for Automated Planning*

        # Returns
        The goals space.
        """
        return self._get_goals_()

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        """Get the domain goals space (finite or infinite set).

        This is a helper function called by default from #Goals._get_goals(), the difference being that the result is
        not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The goals space.
        """
        raise NotImplementedError

    @autocastable
    def is_goal(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_predicate]:
        """Indicate whether an observation belongs to the goals.

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the domain
            goals space provided by #Goals.get_goals(), but it can be overridden for faster implementations.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the observation is a goal (False otherwise).
        """
        return self._is_goal(observation)

    def _is_goal(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_predicate]:
        """Indicate whether an observation belongs to the goals.

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the domain
            goals space provided by #Goals._get_goals(), but it can be overridden for faster implementations.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the observation is a goal (False otherwise).
        """
        goals = self._get_goals()
        if self.T_agent == Union:
            return goals.contains(observation)
        else:  # StrDict
            return {k: goals[k].contains(v) for k, v in observation.items()}
