# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools

from skdecide.core import D, Distribution, SingleValueDistribution, autocastable

__all__ = ["Initializable", "UncertainInitialized", "DeterministicInitialized"]


class Initializable:
    """A domain must inherit this class if it can be initialized."""

    @autocastable
    def reset(self) -> D.T_agent[D.T_observation]:
        """Reset the state of the environment and return an initial observation.

        By default, #Initializable.reset() provides some boilerplate code and internally calls #Initializable._reset()
        (which returns an initial state). The boilerplate code automatically stores the initial state into the #_memory
        attribute and samples a corresponding observation.

        # Returns
        An initial observation.
        """
        return self._reset()

    def _reset(self) -> D.T_agent[D.T_observation]:
        """Reset the state of the environment and return an initial observation.

        By default, #Initializable._reset() provides some boilerplate code and internally
        calls #Initializable._state_reset() (which returns an initial state). The boilerplate code automatically stores
        the initial state into the #_memory attribute and samples a corresponding observation.

        # Returns
        An initial observation.
        """
        initial_state = self._state_reset()
        self._memory = self._init_memory(initial_state)
        initial_observation = self._get_observation_distribution(initial_state).sample()
        return initial_observation

    def _state_reset(self) -> D.T_state:
        """Reset the state of the environment and return an initial state.

        This is a helper function called by default from #Initializable._reset(). It focuses on the state level, as
        opposed to the observation one for the latter.

        # Returns
        An initial state.
        """
        raise NotImplementedError


class UncertainInitialized(Initializable):
    """A domain must inherit this class if its states are initialized according to a probability distribution known as
    white-box."""

    def _state_reset(self) -> D.T_state:
        initial_state = self._get_initial_state_distribution().sample()
        return initial_state

    @autocastable
    def get_initial_state_distribution(self) -> Distribution[D.T_state]:
        """Get the (cached) probability distribution of initial states.

        By default, #UncertainInitialized.get_initial_state_distribution() internally
        calls #UncertainInitialized._get_initial_state_distribution_() the first time and automatically caches its value
        to make future calls more efficient (since the initial state distribution is assumed to be constant).

        # Returns
        The probability distribution of initial states.
        """
        return self._get_initial_state_distribution()

    @functools.lru_cache()
    def _get_initial_state_distribution(self) -> Distribution[D.T_state]:
        """Get the (cached) probability distribution of initial states.

        By default, #UncertainInitialized._get_initial_state_distribution() internally
        calls #UncertainInitialized._get_initial_state_distribution_() the first time and automatically caches its value
        to make future calls more efficient (since the initial state distribution is assumed to be constant).

        # Returns
        The probability distribution of initial states.
        """
        return self._get_initial_state_distribution_()

    def _get_initial_state_distribution_(self) -> Distribution[D.T_state]:
        """Get the probability distribution of initial states.

        This is a helper function called by default from #UncertainInitialized._get_initial_state_distribution(), the
        difference being that the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The probability distribution of initial states.
        """
        raise NotImplementedError


class DeterministicInitialized(UncertainInitialized):
    """A domain must inherit this class if it has a deterministic initial state known as white-box."""

    def _get_initial_state_distribution_(self) -> Distribution[D.T_state]:
        return SingleValueDistribution(self._get_initial_state())

    @autocastable
    def get_initial_state(self) -> D.T_state:
        """Get the (cached) initial state.

        By default, #DeterministicInitialized.get_initial_state() internally
        calls #DeterministicInitialized._get_initial_state_() the first time and automatically caches its value to make
        future calls more efficient (since the initial state is assumed to be constant).

        # Returns
        The initial state.
        """
        return self._get_initial_state()

    @functools.lru_cache()
    def _get_initial_state(self) -> D.T_state:
        """Get the (cached) initial state.

        By default, #DeterministicInitialized._get_initial_state() internally
        calls #DeterministicInitialized._get_initial_state_() the first time and automatically caches its value to make
        future calls more efficient (since the initial state is assumed to be constant).

        # Returns
        The initial state.
        """
        return self._get_initial_state_()

    def _get_initial_state_(self) -> D.T_state:
        """Get the initial state.

        This is a helper function called by default from #DeterministicInitialized._get_initial_state(), the difference
        being that the result is not cached here.

        # Returns
        The initial state.
        """
        raise NotImplementedError
