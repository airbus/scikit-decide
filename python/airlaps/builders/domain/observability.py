import functools
from typing import Generic

from airlaps.core import T_state, T_observation, T_event, Space, Distribution, SingleValueDistribution

__all__ = ['PartiallyObservableDomain', 'TransformedObservableDomain', 'FullyObservableDomain']


class PartiallyObservableDomain(Generic[T_state, T_observation, T_event]):
    """A domain must inherit this class if it is partially observable.

    "Partially observable" means that the observation provided to the agent is computed from (but generally not equal
    to) the internal state of the domain. Additionally, according to literature, a partially observable domain must
    provide the probability distribution of the observation given a state and event.
    """

    @functools.lru_cache()
    def get_observation_space(self) -> Space[T_observation]:
        """Get the (cached) observation space (finite or infinite set).

        By default, #PartiallyObservableDomain.get_observation_space() internally
        calls #PartiallyObservableDomain._get_observation_space() the first time and automatically caches its value to
        make future calls more efficient (since the observation space is assumed to be constant).

        # Returns
        The observation space.
        """
        return self._get_observation_space_()

    def _get_observation_space_(self) -> Space[T_observation]:
        """Get the observation space (finite or infinite set).

        This is a helper function called by default from #PartiallyObservableDomain.get_observation_space(), the
        difference being that the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The observation space.
        """
        raise NotImplementedError

    def is_observation(self, observation: T_observation) -> bool:
        """Check that an observation indeed belongs to the domain observation space.

        !!! tip
            By default, this function is implemented using the #airlaps.core.Space.contains() function on the domain
            observation space provided by #PartiallyObservableDomain.get_observation_space(), but it can be overwritten
            for faster implementations.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the observation belongs to the domain observation space (False otherwise).
        """
        return self.get_observation_space().contains(observation)

    def get_observation_distribution(self, state: T_state, event: T_event) -> Distribution[T_observation]:
        """Get the probability distribution of the observation given a state and event.

        In mathematical terms (discrete case), if the event is an action $a$, this function represents: $P(O|s, a)$,
        where $O$ is the random variable of the observation.

        # Parameters
        state: The state to be observed.
        event: The last applied event.

        # Returns
        The probability distribution of the observation.
        """
        raise NotImplementedError


class TransformedObservableDomain(PartiallyObservableDomain[T_state, T_observation, T_event]):
    """A domain must inherit this class if it is transformed observable.

    "Transformed observable" means that the observation provided to the agent is deterministically computed from (but
    generally not equal to) the internal state of the domain.
    """

    def get_observation_distribution(self, state: T_state, event: T_event) -> Distribution[T_observation]:
        return SingleValueDistribution(self.get_observation(state, event))

    def get_observation(self, state: T_state, event: T_event) -> T_observation:
        """Get the deterministic observation given a state and event.

        # Parameters
        state: The state to be observed.
        event: The last applied event.

        # Returns
        The probability distribution of the observation.
        """
        raise NotImplementedError


class FullyObservableDomain(TransformedObservableDomain[T_state, T_state, T_event], Generic[T_state, T_event]):
    """A domain must inherit this class if it is fully observable.

    "Fully observable" means that the observation provided to the agent is equal to the internal state of the domain.

    !!! warning
        In the case of fully observable domains, make sure that the observation type ~T_observation is equal to the
        state type ~T_state.
    """

    def get_observation(self, state: T_state, event: T_event) -> T_state:
        return state
