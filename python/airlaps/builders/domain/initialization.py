import functools
from copy import deepcopy
from typing import Generic

from airlaps.core import T_state, T_observation, Distribution, SingleValueDistribution

__all__ = ['InitializableDomain', 'UncertainInitializedDomain', 'DeterministicInitializedDomain']


class InitializableDomain(Generic[T_observation]):
    """A domain must inherit this class if it can be initialized."""

    def reset(self) -> T_observation:
        """Reset the state of the environment and return an initial observation.

        By default, #InitializableDomain.reset() provides some boilerplate code and internally
        calls #InitializableDomain._reset() (which returns an initial state). The boilerplate code automatically stores
        the initial state into the #_memory attribute and samples a corresponding observation.

        # Returns
        An initial observation.
        """
        initial_state = self._reset()
        # TODO: document in "get_observation_distribution" that event=None must be handled
        #  (i.e. for initial observation)
        initial_observation = self.get_observation_distribution(initial_state, None).sample()
        self._memory = self._init_memory([deepcopy(initial_state)])
        return initial_observation

    def _reset(self) -> T_state:
        """Reset the state of the environment and return an initial state.

        This is a helper function called by default from #InitializableDomain.reset(). It focuses on the state level, as
        opposed to the observation one for the latter.

        # Returns
        An initial state.
        """
        raise NotImplementedError


class UncertainInitializedDomain(InitializableDomain[T_observation], Generic[T_state, T_observation]):
    """A domain must inherit this class if its states are initialized according to a probability distribution known as
    white-box."""

    def _reset(self) -> T_state:
        initial_state = self.get_initial_state_distribution().sample()
        return initial_state

    @functools.lru_cache()
    def get_initial_state_distribution(self) -> Distribution[T_state]:
        """Get the (cached) probability distribution of initial states.

        By default, #UncertainInitializedDomain.get_initial_state_distribution() internally
        calls #UncertainInitializedDomain._get_initial_state_distribution_() the first time and automatically caches its
        value to make future calls more efficient (since the initial state distribution is assumed to be constant).

        # Returns
        The probability distribution of initial states.
        """
        return self._get_initial_state_distribution_()

    def _get_initial_state_distribution_(self) -> Distribution[T_state]:
        """Get the probability distribution of initial states.

        This is a helper function called by default from #UncertainInitializedDomain.get_initial_state_distribution(),
        the difference being that the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The probability distribution of initial states.
        """
        raise NotImplementedError


class DeterministicInitializedDomain(UncertainInitializedDomain[T_state, T_observation]):
    """A domain must inherit this class if it has a deterministic initial state known as white-box."""

    def _get_initial_state_distribution_(self) -> Distribution[T_state]:
        return SingleValueDistribution(self.get_initial_state())

    @functools.lru_cache()
    def get_initial_state(self) -> T_state:
        """Get the (cached) initial state.

        By default, #DeterministicInitializedDomain.get_initial_state() internally
        calls #DeterministicInitializedDomain._get_initial_state_() the first time and automatically caches its
        value to make future calls more efficient (since the initial state is assumed to be constant).

        # Returns
        The initial state.
        """
        return self._get_initial_state_()

    def _get_initial_state_(self) -> T_state:
        """Get the initial state.

        This is a helper function called by default from #DeterministicInitializedDomain.get_initial_state(),
        the difference being that the result is not cached here.

        # Returns
        The initial state.
        """
        raise NotImplementedError
