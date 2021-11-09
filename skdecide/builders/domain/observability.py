# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from typing import Optional, Union

from skdecide.core import D, Distribution, SingleValueDistribution, Space, autocastable

__all__ = ["PartiallyObservable", "TransformedObservable", "FullyObservable"]


class PartiallyObservable:
    """A domain must inherit this class if it is partially observable.

    "Partially observable" means that the observation provided to the agent is computed from (but generally not equal
    to) the internal state of the domain. Additionally, according to literature, a partially observable domain must
    provide the probability distribution of the observation given a state and action.
    """

    @autocastable
    def get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
        """Get the (cached) observation space (finite or infinite set).

        By default, #PartiallyObservable.get_observation_space() internally
        calls #PartiallyObservable._get_observation_space_() the first time and automatically caches its value to make
        future calls more efficient (since the observation space is assumed to be constant).

        # Returns
        The observation space.
        """
        return self._get_observation_space()

    @functools.lru_cache()
    def _get_observation_space(self) -> D.T_agent[Space[D.T_observation]]:
        """Get the (cached) observation space (finite or infinite set).

        By default, #PartiallyObservable._get_observation_space() internally
        calls #PartiallyObservable._get_observation_space_() the first time and automatically caches its value to make
        future calls more efficient (since the observation space is assumed to be constant).

        # Returns
        The observation space.
        """
        return self._get_observation_space_()

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        """Get the observation space (finite or infinite set).

        This is a helper function called by default from #PartiallyObservable._get_observation_space(), the difference
        being that the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The observation space.
        """
        raise NotImplementedError

    @autocastable
    def is_observation(self, observation: D.T_agent[D.T_observation]) -> bool:
        """Check that an observation indeed belongs to the domain observation space.

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the domain
            observation space provided by #PartiallyObservable.get_observation_space(), but it can be overridden for
            faster implementations.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the observation belongs to the domain observation space (False otherwise).
        """
        return self._is_observation(observation)

    def _is_observation(self, observation: D.T_agent[D.T_observation]) -> bool:
        """Check that an observation indeed belongs to the domain observation space.

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the domain
            observation space provided by #PartiallyObservable._get_observation_space(), but it can be overridden for
            faster implementations.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the observation belongs to the domain observation space (False otherwise).
        """
        observation_space = self._get_observation_space()
        if self.T_agent == Union:
            return observation_space.contains(observation)
        else:  # StrDict
            return all(observation_space[k].contains(v) for k, v in observation.items())

    @autocastable
    def get_observation_distribution(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> Distribution[D.T_agent[D.T_observation]]:
        """Get the probability distribution of the observation given a state and action.

        In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
        where $O$ is the random variable of the observation.

        # Parameters
        state: The state to be observed.
        action: The last applied action (or None if the state is an initial state).

        # Returns
        The probability distribution of the observation.
        """
        return self._get_observation_distribution(state, action)

    def _get_observation_distribution(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> Distribution[D.T_agent[D.T_observation]]:
        """Get the probability distribution of the observation given a state and action.

        In mathematical terms (discrete case), given an action $a$, this function represents: $P(O|s, a)$,
        where $O$ is the random variable of the observation.

        # Parameters
        state: The state to be observed.
        action: The last applied action (or None if the state is an initial state).

        # Returns
        The probability distribution of the observation.
        """
        raise NotImplementedError


class TransformedObservable(PartiallyObservable):
    """A domain must inherit this class if it is transformed observable.

    "Transformed observable" means that the observation provided to the agent is deterministically computed from (but
    generally not equal to) the internal state of the domain.
    """

    def _get_observation_distribution(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> Distribution[D.T_agent[D.T_observation]]:
        return SingleValueDistribution(self._get_observation(state, action))

    @autocastable
    def get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        """Get the deterministic observation given a state and action.

        # Parameters
        state: The state to be observed.
        action: The last applied action (or None if the state is an initial state).

        # Returns
        The probability distribution of the observation.
        """
        return self._get_observation(state, action)

    def _get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        """Get the deterministic observation given a state and action.

        # Parameters
        state: The state to be observed.
        action: The last applied action (or None if the state is an initial state).

        # Returns
        The probability distribution of the observation.
        """
        raise NotImplementedError


class FullyObservable(TransformedObservable):
    """A domain must inherit this class if it is fully observable.

    "Fully observable" means that the observation provided to the agent is equal to the internal state of the domain.

    !!! warning
        In the case of fully observable domains, make sure that the observation type D.T_observation is equal to the
        state type D.T_state.
    """

    def _get_observation(
        self,
        state: D.T_state,
        action: Optional[D.T_agent[D.T_concurrency[D.T_event]]] = None,
    ) -> D.T_agent[D.T_observation]:
        return state
